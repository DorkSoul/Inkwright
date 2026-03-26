FROM python:3.11-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
        calibre \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Pre-download Kokoro model weights and all 52 voice files at build time.
# Creates one pipeline per language to trigger language-specific model downloads.
# Voices are language-agnostic style tensors so they can be loaded via any pipeline.
# ---------------------------------------------------------------------------
RUN python - <<'EOF'
from kokoro import KPipeline

REPO = 'hexgrad/Kokoro-82M'

ALL_VOICES = [
    # American English
    'af_alloy','af_aoede','af_bella','af_heart','af_jessica','af_kore',
    'af_nicole','af_nova','af_river','af_sarah','af_sky',
    'am_adam','am_echo','am_eric','am_fenrir','am_liam','am_michael',
    'am_onyx','am_puck','am_santa',
    # British English
    'bf_alice','bf_emma','bf_isabella','bf_lily',
    'bm_daniel','bm_fable','bm_george','bm_lewis',
    # Spanish
    'ef_dora','em_alex','em_santa',
    # French
    'ff_siwis',
    # Hindi
    'hf_alpha','hf_beta','hm_omega','hm_psi',
    # Italian
    'if_sara','im_nicola',
    # Japanese
    'jf_alpha','jf_gongitsune','jf_nezumi','jf_tebukuro','jm_kumo',
    # Portuguese (Brazil)
    'pf_dora','pm_alex','pm_santa',
    # Mandarin Chinese
    'zf_xiaobei','zf_xiaoni','zf_xiaoxiao','zf_xiaoyi',
    'zm_yunjian','zm_yunxi','zm_yunxia','zm_yunyang',
]

print("=== Caching Kokoro model and all voice files ===")

# Use an English pipeline to download the base model and all voice .pt files
p_en = KPipeline(lang_code='a', repo_id=REPO)
for voice in ALL_VOICES:
    try:
        p_en.load_voice(voice)
        print(f"  voice cached: {voice}")
    except Exception as e:
        print(f"  WARNING: {voice}: {e}")

# Warm up each language pipeline so language-specific G2P models download
LANG_TESTS = [
    ('b', 'Hello.', 'bf_emma'),
    ('e', 'Hola.', 'ef_dora'),
    ('f', 'Bonjour.', 'ff_siwis'),
    ('h', 'नमस्ते.', 'hf_alpha'),
    ('i', 'Ciao.', 'if_sara'),
    ('j', 'こんにちは。', 'jf_alpha'),
    ('p', 'Olá.', 'pf_dora'),
    ('z', '你好。', 'zf_xiaoxiao'),
]

for lang, text, voice in LANG_TESTS:
    try:
        p = KPipeline(lang_code=lang, repo_id=REPO)
        for _ in p(text, voice=voice): break
        print(f"  lang '{lang}' pipeline warmed up")
    except Exception as e:
        print(f"  WARNING: lang '{lang}': {e}")

print("=== All Kokoro assets cached ===")
EOF

# Block all HuggingFace Hub network calls at runtime.
# The model and voices are already in the pip package cache from above.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy application source
COPY . .

# Create volume mount points
RUN mkdir -p /books /audiobooks /config

EXPOSE 8084

CMD ["python", "app.py"]
