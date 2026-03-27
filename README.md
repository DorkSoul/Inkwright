# Inkwright

Self-hosted audiobook generator. Drop in your EPUB collection, pick a voice, and Inkwright converts them to MP3 audiobooks with word-level timing for synchronized text highlighting.

Built on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — a small, fast, high-quality open-weights TTS model.

---

## Features

- **EPUB → MP3** conversion with a background worker queue
- **52 voices** across 9 languages (American English, British English, Spanish, French, Hindi, Italian, Japanese, Mandarin, Brazilian Portuguese)
- **Voice blending** — mix two voices at a configurable ratio
- **Word-level timing index** stored alongside each MP3 for potential karaoke-style readers
- **Voice preview** — generate a short sample before committing to a full conversion; compare all voices side-by-side
- **Library view** — grid/list, sort by title/author/series/date/status, filter by status, group by author/series/status, bulk queue/delete
- **Mass import** — select a folder and import all supported ebooks recursively in one go
- **Series metadata** — reads Calibre and EPUB3 series tags; displayed in library groupings
- **Cover extraction** — pulls cover art from EPUB and serves it from a local cache

---

## Quick Start

### Docker Compose (recommended)

```yaml
services:
  inkwright:
    image: ghcr.io/dorksoul/inkwright:latest
    container_name: inkwright
    ports:
      - "8084:8084"
    volumes:
      - /path/to/config:/config        # DB, covers, previews, favourites
      - /path/to/books:/books          # EPUB source files
      - /path/to/audiobooks:/audiobooks  # Generated MP3s + index files
    environment:
      - BOOKS_DIR=/books
      - AUDIOBOOKS_DIR=/audiobooks
      - CONFIG_DIR=/config
      - PORT=8084
      # - SECRET_KEY=change-me         # Set a stable secret key in production
    restart: unless-stopped
```

Then open `http://localhost:8084`.

### Run from source

```bash
# Python 3.10+
pip install -r requirements.txt

export BOOKS_DIR=./books
export AUDIOBOOKS_DIR=./audiobooks
export CONFIG_DIR=./config

python app.py
```

The Kokoro model weights (~330 MB) are downloaded from Hugging Face on first use and cached in `~/.cache/huggingface/`.

---

## Usage

### Importing books

- **Single upload**: drag an EPUB onto the upload area on the Library page, or click to browse.
- **Folder import**: click "Import Folder" in the nav, select a directory. All supported files found recursively are uploaded one by one with per-file status.

Supported formats: epub, mobi, azw, azw3, fb2, lit, lrf, pdb, snb, tcr *(non-epub formats require conversion to EPUB before TTS processing)*.

### Generating audio

1. Open a book from the library.
2. Choose a voice and optionally a blend voice + ratio.
3. Set speed (0.5× – 2.0×) and language hint if needed.
4. Click **Queue for Audio**. The background worker picks it up and processes paragraphs in order, updating progress in the library.
5. When done, an audio player appears on the book page.

### Voice preview

On the book detail page, click **Preview** under a voice to generate a ~10-second sample from the book's first paragraph. Previews are cached on disk and survive page refreshes. Use **Clear preview** to remove a single preview or **Clear all previews** on the Compare page.

**Compare Voices**: generates previews for multiple voices at once. Select individual voices or filter by language with the chip buttons, then click **Generate Selected**.

---

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `BOOKS_DIR` | `/books` | Directory for EPUB source files |
| `AUDIOBOOKS_DIR` | `/audiobooks` | Output directory for MP3s and index JSON files |
| `CONFIG_DIR` | `/config` | Directory for SQLite DB, cover images, voice previews, favourites |
| `PORT` | `8084` | HTTP port |
| `SECRET_KEY` | random | Flask secret key for session signing. Set a stable value to persist sessions across restarts. |

---

## Data layout

```
/config/
  inkwright.db          # SQLite database
  covers/               # Extracted cover images (JPEG/PNG)
  previews/             # Voice preview MP3 samples

/books/
  my-book.epub          # Source EPUB files (never modified)

/audiobooks/
  42.mp3                # Full audiobook for book ID 42
  42.json               # Word-level timing index for book ID 42
```

### Index JSON format

Each generated audiobook has a companion `.json` file:

```json
{
  "book_id": 42,
  "title": "...",
  "author": "...",
  "duration_seconds": 12345.6,
  "chapters": [
    { "index": 0, "title": "Chapter 1", "start_time": 0.0 }
  ],
  "source_paragraphs": [
    { "paragraph_index": 0, "chapter_index": 0, "chapter_title": "...", "text": "..." }
  ],
  "words": [
    { "paragraph_index": 0, "char_start": 0, "char_end": 5, "start_time": 0.0, "end_time": 0.3 }
  ]
}
```

This index can be used by any client to implement word-by-word highlighting while the MP3 plays.

---

## Architecture

```
app.py          Flask web app + API routes
worker.py       Background thread — polls DB for queued books, calls TTS pipeline
models.py       SQLAlchemy models (Book, AudioIndex) + SQLite migration
epub_parser.py  ebooklib-based EPUB → paragraph list extractor
tts_engine.py   Kokoro-82M wrapper — synthesis + word-level timing
audio_utils.py  NumPy audio chunks → MP3 (pydub/ffmpeg) + atomic JSON writes
```

The TTS worker runs in a daemon thread within the same process. It polls the database every 5 seconds for books with `tts_status = queued` and processes them one at a time. Progress is written to the database every 10 paragraphs and is visible in the library.

---

## API

The web UI is the primary interface, but the JSON API is usable directly:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/books` | List all books with status |
| `POST` | `/upload` | Upload an EPUB (`Accept: application/json` returns `{book_id, title, author}`) |
| `POST` | `/book/<id>/queue` | Queue a book for TTS |
| `DELETE` | `/book/<id>` | Delete a book and its audio |
| `GET` | `/audiobooks/<id>.mp3` | Stream the generated MP3 |
| `GET` | `/audiobooks/<id>.json` | Download the timing index |
| `POST` | `/books/bulk-queue` | Queue multiple books (`{book_ids: [...]}`) |
| `POST` | `/books/bulk-delete` | Delete multiple books (`{book_ids: [...]}`) |
| `GET` | `/books/status` | Status of queued/processing books |

---

## Building from source

```bash
docker build -t inkwright .
```

The image bundles Python dependencies and downloads the Kokoro model weights at build time so the container starts immediately without a network fetch.

---

## Requirements

- Docker (recommended), or Python 3.10+ with ffmpeg installed
- ~1 GB disk for the container image (model weights included)
- A modern browser (folder import uses `webkitdirectory`)
- No GPU required — Kokoro runs on CPU

---

## License

MIT
