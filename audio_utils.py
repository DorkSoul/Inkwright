import json
import logging
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


def concatenate_chunks(chunks: list) -> np.ndarray:
    """Stack a list of float32 numpy arrays into one."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def save_wav(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a float32 numpy array to a WAV file using soundfile."""
    sf.write(path, audio, sample_rate, subtype='FLOAT')
    logger.debug("WAV written: %s (%.2f s)", path, len(audio) / sample_rate)


def wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    """
    Convert a WAV file to MP3 using ffmpeg.
    Uses VBR quality 2 (~190 kbps) for good quality at reasonable size.
    Raises subprocess.CalledProcessError on failure.
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', wav_path,
        '-codec:a', 'libmp3lame',
        '-b:a', '64k',   # CBR — byte offset maps exactly to time, enabling accurate seeking
        mp3_path,
    ]
    logger.info("Converting WAV → MP3: %s", mp3_path)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        err = result.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {err[-500:]}")
    logger.info("MP3 written: %s", mp3_path)


def write_index_atomic(data: dict, path: str) -> None:
    """
    Write a JSON index file atomically: write to .tmp, then os.rename().
    Guarantees a partial file is never served.
    """
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        logger.info("Index written atomically: %s", path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def audio_to_mp3(
    chunks: list,
    mp3_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """
    Concatenate audio chunks, write WAV to a temp file, convert to MP3,
    clean up the WAV, and return total duration in seconds.
    """
    audio = concatenate_chunks(chunks)
    duration = len(audio) / sample_rate

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = tmp.name

    try:
        save_wav(audio, wav_path, sample_rate)
        wav_to_mp3(wav_path, mp3_path)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return duration
