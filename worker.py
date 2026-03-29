import json
import logging
import os
import threading
import time
import traceback
from datetime import datetime

from models import AudioIndex, Book, CharacterCast, TTSStatus, get_session
from epub_parser import parse_epub, save_cover
from tts_engine import TTSEngine, SAMPLE_RATE
from audio_utils import audio_to_mp3, write_index_atomic

logger = logging.getLogger(__name__)

POLL_INTERVAL = 5  # seconds between queue checks

# Injected by app.py at startup
BOOKS_DIR = '/books'
AUDIOBOOKS_DIR = '/audiobooks'
COVERS_DIR = '/config/covers'


def _build_index(book_id: int, title: str, author: str,
                  source_para_entries: list, word_entries: list) -> dict:
    """Build the JSON index dict from paragraph metadata and word-level timing."""
    chapters = []
    seen_chapters = set()

    # Derive chapter list from source paragraphs (words may be absent for empty paras)
    for entry in source_para_entries:
        ci = entry['chapter_index']
        if ci not in seen_chapters:
            chapters.append({
                'index': ci,
                'title': entry['chapter_title'],
                'start_time': entry['start_time'],
            })
            seen_chapters.add(ci)

    duration = source_para_entries[-1]['end_time'] if source_para_entries else 0.0

    return {
        'book_id': book_id,
        'title': title,
        'author': author,
        'duration_seconds': duration,
        'chapters': chapters,
        'source_paragraphs': [
            {
                'paragraph_index': e['paragraph_index'],
                'chapter_index': e['chapter_index'],
                'chapter_title': e['chapter_title'],
                'text': e['text'],
            }
            for e in source_para_entries
        ],
        'words': [
            {
                'paragraph_index': e['paragraph_index'],
                'char_start': e['char_start'],
                'char_end':   e['char_end'],
                'start_time': e['start_time'],
                'end_time':   e['end_time'],
            }
            for e in word_entries
        ],
    }


def _process_book(book_id: int):
    mp3_path = None
    session = get_session()
    try:
        book = session.get(Book, book_id)
        if book is None:
            logger.error("Book %d not found in DB", book_id)
            return

        book.tts_status = TTSStatus.processing
        book.tts_progress_pct = 0.0
        book.tts_error = None
        book.updated_at = datetime.utcnow()
        session.commit()

        # Use character cast only when explicitly requested
        cast_data = None
        if book.tts_use_cast:
            cast = session.query(CharacterCast).filter(CharacterCast.book_id == book_id).first()
            if cast and cast.status.value == 'done' and cast.cast_json:
                cast_data = json.loads(cast.cast_json)
            else:
                logger.warning("Book %d: tts_use_cast=True but no completed cast found — using default voice", book_id)

        # Build paragraph_index → voice_id lookup from cast data
        para_speaker: dict[int, str] = {}
        if cast_data:
            char_voices = cast_data.get('characters', {})
            for seg in cast_data.get('segments', []):
                speaker = seg.get('speaker', 'NARRATOR')
                voice = (char_voices.get(speaker) or {}).get('voice') or None
                para_speaker[seg['paragraph_index']] = voice

        epub_path = os.path.join(BOOKS_DIR, book.epub_filename)
        logger.info("Processing book %d: %s", book_id, epub_path)

        # --- Parse EPUB ---
        parsed = parse_epub(epub_path)

        # Save cover if not already saved
        if parsed.cover_data and not book.cover_image_path:
            cover_filename = save_cover(
                parsed.cover_data, parsed.cover_media_type or 'image/jpeg',
                COVERS_DIR, book_id
            )
            if cover_filename:
                book.cover_image_path = cover_filename
                session.commit()

        paragraphs = parsed.paragraphs
        total = len(paragraphs)
        if total == 0:
            raise ValueError("EPUB produced no parseable paragraphs")

        # --- TTS ---
        voice       = book.tts_voice       or 'af_heart'
        speed       = float(book.tts_speed)       if book.tts_speed       is not None else 1.0
        voice_blend = book.tts_voice_blend  or None
        blend_ratio = float(book.tts_blend_ratio) if book.tts_blend_ratio is not None else 0.5
        lang_code   = book.tts_language     or None

        engine = TTSEngine(voice=voice, speed=speed,
                           voice_blend=voice_blend, blend_ratio=blend_ratio,
                           lang_code=lang_code)
        engine.load()

        audio_chunks        = []
        source_para_entries = []
        word_entries        = []
        current_time        = 0.0

        try:
            for i, para in enumerate(paragraphs):
                voice_override = para_speaker.get(para.paragraph_index) if cast_data else None
                para_audio, para_words = engine.synthesise_with_words(para.text, voice_override=voice_override)
                para_duration = len(para_audio) / SAMPLE_RATE

                source_para_entries.append({
                    'paragraph_index': para.paragraph_index,
                    'chapter_index':   para.chapter_index,
                    'chapter_title':   para.chapter_title,
                    'text':            para.text,
                    'start_time':      current_time,
                    'end_time':        current_time + para_duration,
                })

                for w in para_words:
                    word_entries.append({
                        'paragraph_index': para.paragraph_index,
                        'char_start':  w['char_start'],
                        'char_end':    w['char_end'],
                        'start_time':  current_time + w['start_time'],
                        'end_time':    current_time + w['end_time'],
                    })

                audio_chunks.append(para_audio)
                current_time += para_duration

                pct = round((i + 1) / total * 100, 1)
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    session.query(Book).filter(Book.id == book_id).update(
                        {'tts_progress_pct': pct, 'updated_at': datetime.utcnow()}
                    )
                    session.commit()
                    logger.debug("Book %d: %d/%d paragraphs (%.1f%%)", book_id, i + 1, total, pct)
        finally:
            engine.close()

        # --- Audio export ---
        mp3_filename = f"{book_id}.mp3"
        mp3_path     = os.path.join(AUDIOBOOKS_DIR, mp3_filename)
        total_duration = audio_to_mp3(audio_chunks, mp3_path)

        # --- JSON index ---
        index_filename = f"{book_id}.json"
        index_path = os.path.join(AUDIOBOOKS_DIR, index_filename)
        index_data = _build_index(book_id, parsed.title, parsed.author,
                                   source_para_entries, word_entries)
        write_index_atomic(index_data, index_path)

        # --- DB update ---
        # Remove existing AudioIndex if retrying
        existing = session.query(AudioIndex).filter(AudioIndex.book_id == book_id).first()
        if existing:
            session.delete(existing)
            session.flush()

        audio_index = AudioIndex(
            book_id=book_id,
            audio_filename=mp3_filename,
            index_json_filename=index_filename,
            duration_seconds=total_duration,
            chapter_count=len(index_data['chapters']),
            paragraph_count=len(source_para_entries),
            generated_at=datetime.utcnow(),
        )
        session.add(audio_index)

        book.tts_status = TTSStatus.done
        book.tts_progress_pct = 100.0
        book.updated_at = datetime.utcnow()
        session.commit()

        logger.info(
            "Book %d done — %.1f s audio, %d chapters, %d paragraphs, %d words",
            book_id, total_duration, len(index_data['chapters']),
            len(source_para_entries), len(word_entries)
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Book %d failed: %s\n%s", book_id, e, tb)
        if mp3_path and os.path.exists(mp3_path):
            try:
                os.remove(mp3_path)
                logger.info("Removed partial mp3 for book %d: %s", book_id, mp3_path)
            except OSError as rm_err:
                logger.warning("Could not remove partial mp3 %s: %s", mp3_path, rm_err)
        try:
            session.query(Book).filter(Book.id == book_id).update({
                'tts_status': TTSStatus.error,
                'tts_error': str(e)[:1000],
                'updated_at': datetime.utcnow(),
            })
            session.commit()
        except Exception as db_err:
            logger.error("Could not update error status for book %d: %s", book_id, db_err)
    finally:
        session.close()


def _worker_loop():
    logger.info("TTS worker started")
    while True:
        try:
            session = get_session()
            try:
                book = (
                    session.query(Book)
                    .filter(Book.tts_status == TTSStatus.queued)
                    .order_by(Book.updated_at.asc())
                    .first()
                )
                book_id = book.id if book else None
            finally:
                session.close()

            if book_id is not None:
                logger.info("Worker picked up book %d", book_id)
                _process_book(book_id)
            else:
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            logger.error("Worker loop error: %s", e)
            time.sleep(POLL_INTERVAL)


def start_worker():
    """Start the background TTS worker thread. Call once at app startup."""
    t = threading.Thread(target=_worker_loop, name='tts-worker', daemon=True)
    t.start()
    logger.info("TTS worker thread started")
    return t
