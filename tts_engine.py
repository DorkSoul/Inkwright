import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# Kokoro tokens are phonemic — much more granular than LLM tokens.
# Conservative estimate: ~1.3 Kokoro tokens per character of English prose.
# So 1 char ≈ 1.3 tokens  →  1 token ≈ 0.77 chars.
TOKENS_PER_CHAR = 1.3

MAX_TOKENS = 500                                    # hard cap — never exceed
TARGET_TOKENS = 175                                 # sweet-spot midpoint (100–250)

MAX_CHARS    = int(MAX_TOKENS    / TOKENS_PER_CHAR)  # ~385 chars
TARGET_CHARS = int(TARGET_TOKENS / TOKENS_PER_CHAR)  # ~135 chars

# All voices shipped with Kokoro-82M.
# af = American Female, am = American Male, bf = British Female, bm = British Male
AVAILABLE_VOICES = [
    # American Female
    'af_heart',
    'af_bella',
    'af_nicole',
    'af_sarah',
    'af_sky',
    # American Male
    'am_adam',
    'am_michael',
    # British Female
    'bf_emma',
    'bf_isabella',
    # British Male
    'bm_george',
    'bm_lewis',
]
DEFAULT_VOICE = 'af_heart'

VOICE_LABELS = {
    'af_heart':    'Heart',
    'af_bella':    'Bella',
    'af_nicole':   'Nicole',
    'af_sarah':    'Sarah',
    'af_sky':      'Sky',
    'am_adam':     'Adam',
    'am_michael':  'Michael',
    'bf_emma':     'Emma',
    'bf_isabella': 'Isabella',
    'bm_george':   'George',
    'bm_lewis':    'Lewis',
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at common boundary punctuation."""
    # Split on '. ', '! ', '? ' followed by a capital or end of string
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str) -> list[str]:
    """
    Split text into chunks that target Kokoro's sweet spot (100–250 tokens).

    Strategy:
    - Always split into sentences first.
    - Greedily group sentences together up to TARGET_CHARS.
    - Once adding the next sentence would push the current chunk past TARGET_CHARS,
      flush the current chunk and start a new one.
    - Hard ceiling: if a single sentence exceeds MAX_CHARS, hard-split it at
      MAX_CHARS boundaries so the model is never fed an oversized input.

    All chunks produced are concatenated into one index entry by the caller.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current = ''

    for sentence in sentences:
        # A single sentence that's too long — hard split it first
        if len(sentence) > MAX_CHARS:
            if current:
                chunks.append(current)
                current = ''
            for i in range(0, len(sentence), MAX_CHARS):
                chunks.append(sentence[i:i + MAX_CHARS])
            continue

        candidate = (current + ' ' + sentence).strip() if current else sentence

        if len(candidate) <= TARGET_CHARS:
            # Still within sweet spot — keep grouping
            current = candidate
        else:
            # Adding this sentence would exceed the target; flush and restart
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def _lang_code_for_voice(voice: str) -> str:
    """
    Derive the correct Kokoro lang_code from the voice prefix.
    bf_* and bm_* are British English; everything else is American English.
    Using the wrong G2P for a voice produces noticeably unnatural phonemisation.
    """
    if voice.startswith('bf_') or voice.startswith('bm_'):
        return 'b'
    return 'a'


class TTSEngine:
    """
    Wraps a Kokoro KPipeline instance.
    Instantiate once per job, then call synthesise() per paragraph.
    """

    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = 1.0):
        self.voice = voice if voice in AVAILABLE_VOICES else DEFAULT_VOICE
        self.speed = max(0.5, min(2.0, float(speed)))  # clamp to safe range
        self._pipeline = None

    def load(self):
        """Lazy-load the Kokoro pipeline (deferred so import errors surface clearly)."""
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline
            lang = _lang_code_for_voice(self.voice)
            self._pipeline = KPipeline(lang_code=lang)
            logger.info("Kokoro pipeline loaded (voice=%s, lang=%s, speed=%.2f)",
                        self.voice, lang, self.speed)
        except Exception as e:
            logger.error("Failed to load Kokoro pipeline: %s", e)
            raise

    def synthesise(self, text: str) -> np.ndarray:
        """
        Synthesise text → float32 numpy array at SAMPLE_RATE Hz.

        Long paragraphs are split at sentence boundaries before synthesis
        but concatenated into a single array (single index entry).
        """
        if self._pipeline is None:
            self.load()

        chunks = _chunk_text(text)
        logger.debug(
            "Paragraph → %d TTS chunk(s), sizes: %s chars",
            len(chunks), [len(c) for c in chunks],
        )

        all_audio: list[np.ndarray] = []

        for chunk in chunks:
            chunk_audio = self._synthesise_chunk(chunk)
            if chunk_audio is not None and len(chunk_audio) > 0:
                all_audio.append(chunk_audio)

        if not all_audio:
            logger.warning("TTS produced no audio for text: %.60s...", text)
            return np.zeros(SAMPLE_RATE // 10, dtype=np.float32)

        return np.concatenate(all_audio)

    def _synthesise_chunk(self, text: str) -> np.ndarray:
        audio_parts = []
        try:
            for _, _, audio in self._pipeline(text, voice=self.voice, speed=self.speed):
                if audio is not None and len(audio) > 0:
                    audio_parts.append(audio)
        except Exception as e:
            logger.error("Kokoro error on chunk '%.60s...': %s", text, e)
            raise

        if not audio_parts:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(audio_parts)

    def close(self):
        self._pipeline = None
