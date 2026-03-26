import json
import logging
import mimetypes
import os
import random
import threading
from datetime import datetime

from flask import (
    Flask, abort, jsonify, redirect, render_template,
    request, send_file, url_for
)

import models
from models import AudioIndex, Book, TTSStatus, db_init, get_session
from epub_parser import parse_epub, save_cover
import worker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — overridable via environment variables
# ---------------------------------------------------------------------------
BOOKS_DIR = os.environ.get('BOOKS_DIR', '/books')
AUDIOBOOKS_DIR = os.environ.get('AUDIOBOOKS_DIR', '/audiobooks')
CONFIG_DIR = os.environ.get('CONFIG_DIR', '/config')
# Covers stored on SSD (inside config volume) so the library page loads
# without spinning up the HDD. HDD only wakes when audio is requested.
COVERS_DIR = os.path.join(CONFIG_DIR, 'covers')
PREVIEWS_DIR = os.path.join(CONFIG_DIR, 'previews')
FAVOURITES_PATH = os.path.join(CONFIG_DIR, 'favourites.json')
MAX_UPLOAD_MB = int(os.environ.get('MAX_UPLOAD_MB', '50'))

# ---------------------------------------------------------------------------
# Preview job state (in-memory — intentionally not persisted)
# ---------------------------------------------------------------------------
_preview_jobs = {}      # book_id → {status, voice, speed, error}
_preview_all_jobs = {}  # book_id → {status, speed, voices: {voice: 'pending'|'generating'|'ready'|'error'}}
_preview_lock = threading.Lock()

PREVIEW_TARGET_CHARS = 700   # ~45–60 s of speech at 1× speed

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _ensure_dirs():
    for d in (BOOKS_DIR, AUDIOBOOKS_DIR, CONFIG_DIR, COVERS_DIR, PREVIEWS_DIR):
        os.makedirs(d, exist_ok=True)


def _init():
    _ensure_dirs()
    db_init(CONFIG_DIR)
    # Pass dirs to worker module
    worker.BOOKS_DIR = BOOKS_DIR
    worker.AUDIOBOOKS_DIR = AUDIOBOOKS_DIR
    worker.COVERS_DIR = COVERS_DIR
    worker.start_worker()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_book_or_404(book_id: int, session):
    book = session.get(Book, book_id)
    if book is None:
        abort(404)
    return book


def _epub_metadata(epub_path: str) -> tuple[str, str]:
    """Quick metadata extraction without full parse."""
    try:
        parsed = parse_epub(epub_path)
        return parsed.title, parsed.author
    except Exception as e:
        logger.warning("Could not extract EPUB metadata: %s", e)
        return 'Unknown Title', 'Unknown Author'


# ---------------------------------------------------------------------------
# Routes — Library
# ---------------------------------------------------------------------------

@app.route('/')
def library():
    session = get_session()
    try:
        books = session.query(Book).order_by(Book.created_at.desc()).all()
        return render_template('library.html', books=books)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Routes — Upload
# ---------------------------------------------------------------------------

@app.route('/book/upload', methods=['POST'])
def upload_book():
    if 'epub' not in request.files:
        abort(400, 'No file part')

    f = request.files['epub']
    if not f.filename:
        abort(400, 'No file selected')

    if not f.filename.lower().endswith('.epub'):
        abort(400, 'Only .epub files are accepted')

    # Save file
    safe_name = os.path.basename(f.filename)
    dest_path = os.path.join(BOOKS_DIR, safe_name)

    # Avoid collisions
    base, ext = os.path.splitext(safe_name)
    counter = 1
    while os.path.exists(dest_path):
        safe_name = f"{base}_{counter}{ext}"
        dest_path = os.path.join(BOOKS_DIR, safe_name)
        counter += 1

    f.save(dest_path)

    # Extract metadata
    title, author = _epub_metadata(dest_path)

    session = get_session()
    try:
        book = Book(
            title=title,
            author=author,
            epub_filename=safe_name,
            tts_status=TTSStatus.none,
            tts_voice='af_heart',
            tts_progress_pct=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(book)
        session.commit()

        # Try to extract and save cover immediately (saved to SSD covers dir)
        try:
            from epub_parser import parse_epub as _pe, save_cover as _sc
            parsed = _pe(dest_path)
            if parsed.cover_data:
                cover_fn = _sc(parsed.cover_data, parsed.cover_media_type or 'image/jpeg',
                               COVERS_DIR, book.id)
                if cover_fn:
                    book.cover_image_path = cover_fn
                    session.commit()
        except Exception as e:
            logger.warning("Cover extraction failed for book %d: %s", book.id, e)

        book_id = book.id
    finally:
        session.close()

    return redirect(url_for('book_detail', book_id=book_id))


# ---------------------------------------------------------------------------
# Routes — Book detail
# ---------------------------------------------------------------------------

@app.route('/book/<int:book_id>')
def book_detail(book_id):
    session = get_session()
    try:
        book = _get_book_or_404(book_id, session)
        epub_path = os.path.join(BOOKS_DIR, book.epub_filename)
        file_size_mb = None
        if os.path.exists(epub_path):
            file_size_mb = round(os.path.getsize(epub_path) / (1024 * 1024), 2)

        from tts_engine import AVAILABLE_VOICES, VOICE_LABELS
        return render_template(
            'book_detail.html',
            book=book,
            file_size_mb=file_size_mb,
            available_voices=AVAILABLE_VOICES,
            voice_labels=VOICE_LABELS,
            favourites=_load_favourites(),
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Routes — Generate
# ---------------------------------------------------------------------------

VALID_SPEEDS = {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3}

@app.route('/book/<int:book_id>/generate', methods=['POST'])
def generate_book(book_id):
    from tts_engine import AVAILABLE_VOICES
    voice = request.form.get('voice', 'af_heart')
    if voice not in AVAILABLE_VOICES:
        voice = 'af_heart'

    try:
        speed = float(request.form.get('speed', '1.0'))
        if speed not in VALID_SPEEDS:
            speed = 1.0
    except ValueError:
        speed = 1.0

    session = get_session()
    try:
        book = _get_book_or_404(book_id, session)
        book.tts_status = TTSStatus.queued
        book.tts_voice = voice
        book.tts_speed = speed
        book.tts_progress_pct = 0.0
        book.tts_error = None
        book.updated_at = datetime.utcnow()
        session.commit()
    finally:
        session.close()

    return redirect(url_for('book_detail', book_id=book_id))


# ---------------------------------------------------------------------------
# Routes — Status (polled by UI)
# ---------------------------------------------------------------------------

@app.route('/book/<int:book_id>/status')
def book_status(book_id):
    session = get_session()
    try:
        book = _get_book_or_404(book_id, session)
        return jsonify({
            'status': book.tts_status.value,
            'tts_progress_pct': book.tts_progress_pct,
            'tts_error': book.tts_error,
        })
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Routes — Player
# ---------------------------------------------------------------------------

@app.route('/book/<int:book_id>/play')
def play_book(book_id):
    session = get_session()
    try:
        book = _get_book_or_404(book_id, session)
        if book.tts_status != TTSStatus.done:
            return redirect(url_for('book_detail', book_id=book_id))
        return render_template('player.html', book=book)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Routes — Serve audio (Range header support)
# ---------------------------------------------------------------------------

@app.route('/audio/<int:book_id>.mp3')
def serve_audio(book_id):
    mp3_path = os.path.join(AUDIOBOOKS_DIR, f"{book_id}.mp3")
    if not os.path.exists(mp3_path):
        abort(404)
    return send_file(
        mp3_path,
        mimetype='audio/mpeg',
        conditional=True,   # enables Range header / 206 Partial Content
    )


# ---------------------------------------------------------------------------
# Routes — Serve index JSON
# ---------------------------------------------------------------------------

@app.route('/index/<int:book_id>.json')
def serve_index(book_id):
    json_path = os.path.join(AUDIOBOOKS_DIR, f"{book_id}.json")
    if not os.path.exists(json_path):
        abort(404)
    return send_file(json_path, mimetype='application/json')


# ---------------------------------------------------------------------------
# Routes — Serve cover image
# ---------------------------------------------------------------------------

@app.route('/cover/<path:filename>')
def serve_cover(filename):
    # Only allow files directly in COVERS_DIR (no path traversal)
    safe = os.path.basename(filename)
    path = os.path.join(COVERS_DIR, safe)
    if not os.path.exists(path):
        abort(404)
    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or 'image/jpeg')


# ---------------------------------------------------------------------------
# Routes — Delete book
# ---------------------------------------------------------------------------

@app.route('/book/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    session = get_session()
    try:
        book = _get_book_or_404(book_id, session)

        # Remove files
        epub_path = os.path.join(BOOKS_DIR, book.epub_filename)
        if os.path.exists(epub_path):
            os.remove(epub_path)

        if book.cover_image_path:
            cover_path = os.path.join(COVERS_DIR, book.cover_image_path)
            if os.path.exists(cover_path):
                os.remove(cover_path)

        mp3_path = os.path.join(AUDIOBOOKS_DIR, f"{book_id}.mp3")
        if os.path.exists(mp3_path):
            os.remove(mp3_path)

        json_path = os.path.join(AUDIOBOOKS_DIR, f"{book_id}.json")
        if os.path.exists(json_path):
            os.remove(json_path)

        session.delete(book)
        session.commit()
    finally:
        session.close()

    return jsonify({'deleted': True})


# ---------------------------------------------------------------------------
# Favourite voices
# ---------------------------------------------------------------------------

def _load_favourites() -> list:
    try:
        with open(FAVOURITES_PATH) as f:
            return json.load(f).get('voices', [])
    except FileNotFoundError:
        return []
    except Exception:
        return []


def _save_favourites(voices: list):
    with open(FAVOURITES_PATH, 'w') as f:
        json.dump({'voices': voices}, f)


@app.route('/voices/favourite', methods=['POST'])
def toggle_favourite():
    data = request.get_json(silent=True) or {}
    voice = data.get('voice') or request.form.get('voice', '')
    from tts_engine import AVAILABLE_VOICES
    if voice not in AVAILABLE_VOICES:
        abort(400)
    favs = _load_favourites()
    if voice in favs:
        favs.remove(voice)
        favourited = False
    else:
        favs.append(voice)
        favourited = True
    _save_favourites(favs)
    return jsonify({'favourited': favourited, 'favourites': favs})


# ---------------------------------------------------------------------------
# Preview generation
# ---------------------------------------------------------------------------

def _run_preview(book_id: int, epub_path: str, voice: str, speed: float, output_path: str):
    """Background thread: synthesise a short sample and write it to output_path."""
    try:
        from tts_engine import TTSEngine, SAMPLE_RATE
        from audio_utils import audio_to_mp3

        parsed = parse_epub(epub_path)
        paragraphs = parsed.paragraphs
        if not paragraphs:
            raise ValueError("No paragraphs found in EPUB")

        # Pick a random start point between 10% and 85% through the book
        # to avoid front matter and end matter
        lo = max(0, len(paragraphs) // 10)
        hi = max(lo + 1, int(len(paragraphs) * 0.85))
        start = random.randint(lo, hi - 1)

        # Accumulate paragraphs until we reach the target character count
        selected = []
        total_chars = 0
        for para in paragraphs[start:]:
            selected.append(para.text)
            total_chars += len(para.text)
            if total_chars >= PREVIEW_TARGET_CHARS:
                break

        if not selected:
            selected = [paragraphs[0].text]

        engine = TTSEngine(voice=voice, speed=speed)
        engine.load()
        chunks = [engine.synthesise(t) for t in selected]
        engine.close()

        audio_to_mp3(chunks, output_path)

        with _preview_lock:
            _preview_jobs[book_id]['status'] = 'ready'

        logger.info("Preview ready for book %d (%s, %.1fx)", book_id, voice, speed)

    except Exception as e:
        logger.error("Preview failed for book %d: %s", book_id, e)
        with _preview_lock:
            if book_id in _preview_jobs:
                _preview_jobs[book_id]['status'] = 'error'
                _preview_jobs[book_id]['error'] = str(e)


@app.route('/book/<int:book_id>/preview', methods=['POST'])
def start_preview(book_id):
    from tts_engine import AVAILABLE_VOICES
    voice = request.form.get('voice', 'af_heart')
    if voice not in AVAILABLE_VOICES:
        voice = 'af_heart'
    try:
        speed = float(request.form.get('speed', '1.0'))
        if speed not in VALID_SPEEDS:
            speed = 1.0
    except ValueError:
        speed = 1.0

    session = get_session()
    try:
        book = _get_book_or_404(book_id, session)
        epub_path = os.path.join(BOOKS_DIR, book.epub_filename)
    finally:
        session.close()

    with _preview_lock:
        if _preview_jobs.get(book_id, {}).get('status') == 'generating':
            return jsonify({'status': 'generating'})
        _preview_jobs[book_id] = {'status': 'generating', 'voice': voice, 'speed': speed}

    output_path = os.path.join(PREVIEWS_DIR, f"{book_id}_preview.mp3")
    threading.Thread(
        target=_run_preview,
        args=(book_id, epub_path, voice, speed, output_path),
        daemon=True,
    ).start()

    return jsonify({'status': 'generating'})


@app.route('/book/<int:book_id>/preview/status')
def preview_status(book_id):
    with _preview_lock:
        job = dict(_preview_jobs.get(book_id, {}))
    return jsonify({
        'status': job.get('status', 'none'),
        'error': job.get('error'),
    })


@app.route('/book/<int:book_id>/preview.mp3')
def serve_preview(book_id):
    path = os.path.join(PREVIEWS_DIR, f"{book_id}_preview.mp3")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype='audio/mpeg', conditional=True)


# ---------------------------------------------------------------------------
# Preview — per-voice file (used by preview-all)
# ---------------------------------------------------------------------------

@app.route('/book/<int:book_id>/preview/<voice>.mp3')
def serve_preview_voice(book_id, voice):
    from tts_engine import AVAILABLE_VOICES
    if voice not in AVAILABLE_VOICES:
        abort(404)
    path = os.path.join(PREVIEWS_DIR, f"{book_id}_{voice}_preview.mp3")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype='audio/mpeg', conditional=True)


# ---------------------------------------------------------------------------
# Preview all voices
# ---------------------------------------------------------------------------

def _sample_paragraphs(epub_path: str) -> list:
    """Parse EPUB and return a list of paragraph texts for the sample passage."""
    parsed = parse_epub(epub_path)
    paragraphs = parsed.paragraphs
    if not paragraphs:
        return []
    lo = max(0, len(paragraphs) // 10)
    hi = max(lo + 1, int(len(paragraphs) * 0.85))
    start = random.randint(lo, hi - 1)
    selected, total = [], 0
    for para in paragraphs[start:]:
        selected.append(para.text)
        total += len(para.text)
        if total >= PREVIEW_TARGET_CHARS:
            break
    return selected or [paragraphs[0].text]


def _run_preview_all(book_id: int, epub_path: str, speed: float):
    """
    Background thread: synthesise the same passage with every available voice,
    one at a time, updating _preview_all_jobs[book_id] as each voice completes.
    """
    from tts_engine import AVAILABLE_VOICES, TTSEngine
    from audio_utils import audio_to_mp3

    try:
        texts = _sample_paragraphs(epub_path)
        if not texts:
            raise ValueError("No paragraphs found in EPUB")
    except Exception as e:
        with _preview_lock:
            _preview_all_jobs[book_id]['status'] = 'error'
        logger.error("Preview-all failed to sample EPUB for book %d: %s", book_id, e)
        return

    for voice in AVAILABLE_VOICES:
        with _preview_lock:
            if _preview_all_jobs.get(book_id, {}).get('status') == 'cancelled':
                return
            _preview_all_jobs[book_id]['voices'][voice] = 'generating'

        output_path = os.path.join(PREVIEWS_DIR, f"{book_id}_{voice}_preview.mp3")
        try:
            engine = TTSEngine(voice=voice, speed=speed)
            engine.load()
            chunks = [engine.synthesise(t) for t in texts]
            engine.close()
            audio_to_mp3(chunks, output_path)
            with _preview_lock:
                _preview_all_jobs[book_id]['voices'][voice] = 'ready'
            logger.info("Preview-all: %s done for book %d", voice, book_id)
        except Exception as e:
            logger.error("Preview-all: %s failed for book %d: %s", voice, book_id, e)
            with _preview_lock:
                _preview_all_jobs[book_id]['voices'][voice] = 'error'

    with _preview_lock:
        _preview_all_jobs[book_id]['status'] = 'done'
    logger.info("Preview-all complete for book %d", book_id)


@app.route('/book/<int:book_id>/preview-all', methods=['POST'])
def start_preview_all(book_id):
    session = get_session()
    try:
        book = _get_book_or_404(book_id, session)
        epub_path = os.path.join(BOOKS_DIR, book.epub_filename)
    finally:
        session.close()

    try:
        speed = float(request.form.get('speed', '1.0'))
        if speed not in VALID_SPEEDS:
            speed = 1.0
    except ValueError:
        speed = 1.0

    from tts_engine import AVAILABLE_VOICES
    with _preview_lock:
        if _preview_all_jobs.get(book_id, {}).get('status') == 'generating':
            return jsonify({'status': 'generating',
                            'voices': _preview_all_jobs[book_id]['voices']})
        _preview_all_jobs[book_id] = {
            'status': 'generating',
            'speed': speed,
            'voices': {v: 'pending' for v in AVAILABLE_VOICES},
        }

    threading.Thread(
        target=_run_preview_all,
        args=(book_id, epub_path, speed),
        daemon=True,
    ).start()

    return jsonify({'status': 'generating',
                    'voices': _preview_all_jobs[book_id]['voices']})


@app.route('/book/<int:book_id>/preview-all/status')
def preview_all_status(book_id):
    with _preview_lock:
        job = dict(_preview_all_jobs.get(book_id, {}))
    if not job:
        return jsonify({'status': 'none', 'voices': {}})
    return jsonify({
        'status': job.get('status', 'none'),
        'voices': job.get('voices', {}),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    _init()
    port = int(os.environ.get('PORT', '8084'))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
