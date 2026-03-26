import logging
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
REPO_ID = 'hexgrad/Kokoro-82M'

# ---------------------------------------------------------------------------
# Voice catalogue — all 52 voices across 9 languages
# ---------------------------------------------------------------------------

# lang_code → {voice_id: display_name}
# Groups within a language are Female then Male, matching the voice prefix.
VOICES_BY_LANGUAGE = {
    'a': {  # American English
        'af_alloy':   'Alloy',   'af_aoede':   'Aoede',   'af_bella':  'Bella',
        'af_heart':   'Heart',   'af_jessica': 'Jessica', 'af_kore':   'Kore',
        'af_nicole':  'Nicole',  'af_nova':    'Nova',    'af_river':  'River',
        'af_sarah':   'Sarah',   'af_sky':     'Sky',
        'am_adam':    'Adam',    'am_echo':    'Echo',    'am_eric':   'Eric',
        'am_fenrir':  'Fenrir',  'am_liam':    'Liam',    'am_michael':'Michael',
        'am_onyx':    'Onyx',    'am_puck':    'Puck',    'am_santa':  'Santa',
    },
    'b': {  # British English
        'bf_alice':   'Alice',   'bf_emma':    'Emma',
        'bf_isabella':'Isabella','bf_lily':    'Lily',
        'bm_daniel':  'Daniel',  'bm_fable':   'Fable',
        'bm_george':  'George',  'bm_lewis':   'Lewis',
    },
    'e': {  # Spanish
        'ef_dora':  'Dora',
        'em_alex':  'Alex',  'em_santa': 'Santa',
    },
    'f': {  # French
        'ff_siwis': 'Siwis',
    },
    'h': {  # Hindi
        'hf_alpha': 'Alpha', 'hf_beta':  'Beta',
        'hm_omega': 'Omega', 'hm_psi':   'Psi',
    },
    'i': {  # Italian
        'if_sara':   'Sara',
        'im_nicola': 'Nicola',
    },
    'j': {  # Japanese
        'jf_alpha':     'Alpha',    'jf_gongitsune': 'Gongitsune',
        'jf_nezumi':    'Nezumi',   'jf_tebukuro':   'Tebukuro',
        'jm_kumo':      'Kumo',
    },
    'p': {  # Portuguese (Brazil)
        'pf_dora':  'Dora',
        'pm_alex':  'Alex',  'pm_santa': 'Santa',
    },
    'z': {  # Mandarin Chinese
        'zf_xiaobei':  'Xiaobei',  'zf_xiaoni':   'Xiaoni',
        'zf_xiaoxiao': 'Xiaoxiao', 'zf_xiaoyi':   'Xiaoyi',
        'zm_yunjian':  'Yunjian',  'zm_yunxi':     'Yunxi',
        'zm_yunxia':   'Yunxia',   'zm_yunyang':   'Yunyang',
    },
}

LANGUAGES = {
    'a': 'American English',
    'b': 'British English',
    'e': 'Spanish',
    'f': 'French',
    'h': 'Hindi',
    'i': 'Italian',
    'j': 'Japanese',
    'p': 'Portuguese (Brazil)',
    'z': 'Mandarin Chinese',
}

# Flat lookups derived from the above
VOICE_LABELS: dict[str, str] = {}
AVAILABLE_VOICES: list[str] = []
_VOICE_TO_LANG: dict[str, str] = {}

for _lang, _voices in VOICES_BY_LANGUAGE.items():
    for _vid, _label in _voices.items():
        VOICE_LABELS[_vid] = _label
        AVAILABLE_VOICES.append(_vid)
        _VOICE_TO_LANG[_vid] = _lang

DEFAULT_VOICE = 'af_heart'

# Voice prefix → gender label (for UI grouping within a language)
_PREFIX_GENDER = {
    'af': 'Female', 'am': 'Male',
    'bf': 'Female', 'bm': 'Male',
    'ef': 'Female', 'em': 'Male',
    'ff': 'Female',
    'hf': 'Female', 'hm': 'Male',
    'if': 'Female', 'im': 'Male',
    'jf': 'Female', 'jm': 'Male',
    'pf': 'Female', 'pm': 'Male',
    'zf': 'Female', 'zm': 'Male',
}


def lang_code_for_voice(voice: str) -> str:
    """Return the Kokoro lang_code for a given voice id."""
    return _VOICE_TO_LANG.get(voice, 'a')


def voices_for_language(lang_code: str) -> dict[str, str]:
    """Return {voice_id: label} for the given language, split into Female/Male."""
    return VOICES_BY_LANGUAGE.get(lang_code, VOICES_BY_LANGUAGE['a'])


def voice_gender_groups(lang_code: str) -> list[tuple[str, dict]]:
    """
    Return [(gender_label, {voice_id: label}), ...] for a language,
    useful for rendering grouped voice pickers.
    """
    groups: dict[str, dict] = {}
    for vid, label in VOICES_BY_LANGUAGE.get(lang_code, {}).items():
        prefix = vid[:2]
        gender = _PREFIX_GENDER.get(prefix, 'Voice')
        groups.setdefault(gender, {})[vid] = label
    # Canonical order: Female before Male
    return [(g, groups[g]) for g in ('Female', 'Male') if g in groups]


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Wraps a Kokoro KPipeline.
    Supports voice blending (weighted average of two voice tensors) and
    word-level timestamp extraction.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        voice_blend: str = None,
        blend_ratio: float = 0.5,
        lang_code: str = None,
    ):
        self.voice = voice if voice in VOICE_LABELS else DEFAULT_VOICE
        self.speed = max(0.5, min(2.0, float(speed)))
        self.voice_blend = voice_blend if voice_blend in VOICE_LABELS else None
        self.blend_ratio = max(0.0, min(1.0, float(blend_ratio)))
        # Derive lang_code from voice if not explicitly provided
        self.lang_code = lang_code or lang_code_for_voice(self.voice)
        self._pipeline = None

    def load(self):
        if self._pipeline is not None:
            return
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code=self.lang_code, repo_id=REPO_ID)
        logger.info("Kokoro pipeline loaded (lang=%s voice=%s speed=%.2f blend=%s@%.0f%%)",
                    self.lang_code, self.voice, self.speed,
                    self.voice_blend or 'none', self.blend_ratio * 100)

    def _voice_tensor(self):
        """Return the voice style tensor, blended if a second voice is configured."""
        import torch
        pack_a = self._pipeline.load_voice(self.voice)
        if self.voice_blend and 0.0 < self.blend_ratio <= 1.0:
            pack_b = self._pipeline.load_voice(self.voice_blend)
            r = self.blend_ratio
            return (1.0 - r) * pack_a + r * pack_b
        return pack_a

    @staticmethod
    def _to_numpy(audio) -> np.ndarray:
        """Convert torch tensor or array-like to float32 numpy array."""
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        if hasattr(audio, 'detach'):        # torch.Tensor
            return audio.detach().cpu().numpy().astype(np.float32)
        return np.asarray(audio, dtype=np.float32)

    def synthesise_with_words(self, text: str) -> tuple[np.ndarray, list]:
        """
        Synthesise text and return (audio_array, word_entries).

        word_entries is a list of dicts:
            {char_start, char_end, start_time, end_time}
        relative to `text` and to the start of the returned audio.

        Kokoro handles its own chunking internally; we just iterate results
        and accumulate audio + word timestamps.  If token timestamps are not
        available (some non-English languages) the entry spans the whole chunk.
        """
        if self._pipeline is None:
            self.load()

        voice_tensor = self._voice_tensor()
        all_audio: list[np.ndarray] = []
        word_entries: list[dict] = []
        current_time = 0.0
        para_pos = 0  # cursor into `text`

        # split_pattern=None — let Kokoro's phoneme-based chunker handle splits;
        # we don't want newline-based splits for pre-segmented paragraphs.
        for result in self._pipeline(text, voice=voice_tensor,
                                     speed=self.speed, split_pattern=None):
            chunk_audio = self._to_numpy(result.audio)
            if chunk_audio is not None and len(chunk_audio) > 0:
                all_audio.append(chunk_audio)
            chunk_dur = len(chunk_audio) / SAMPLE_RATE if chunk_audio is not None else 0.0

            graphemes = result.graphemes or ''

            # Locate this chunk in the source text
            gstart = text.find(graphemes, para_pos)
            if gstart == -1:
                gstart = para_pos

            tokens = getattr(result, 'tokens', None)
            if tokens:
                local_pos = 0
                for token in tokens:
                    if token.start_ts is None or not token.phonemes or not token.text:
                        continue
                    tidx = graphemes.find(token.text, local_pos)
                    if tidx != -1:
                        char_start = gstart + tidx
                        local_pos = tidx + len(token.text)
                    else:
                        char_start = gstart + local_pos
                    char_end = min(char_start + len(token.text), len(text))
                    word_entries.append({
                        'char_start': char_start,
                        'char_end':   char_end,
                        'start_time': current_time + float(token.start_ts),
                        'end_time':   current_time + float(token.end_ts),
                    })
            elif chunk_dur > 0:
                # No token timestamps — one entry for the whole graphemes chunk
                word_entries.append({
                    'char_start': gstart,
                    'char_end':   min(gstart + len(graphemes), len(text)),
                    'start_time': current_time,
                    'end_time':   current_time + chunk_dur,
                })

            current_time += chunk_dur
            para_pos = gstart + len(graphemes)

        if not all_audio:
            logger.warning("TTS produced no audio for: %.60s…", text)
            return np.zeros(SAMPLE_RATE // 10, dtype=np.float32), []

        return np.concatenate(all_audio), word_entries

    def synthesise(self, text: str) -> np.ndarray:
        """Convenience wrapper — returns audio only, discards word entries."""
        audio, _ = self.synthesise_with_words(text)
        return audio

    def close(self):
        self._pipeline = None
