FROM python:3.11-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Pre-download Kokoro model weights and all voice files at build time.
# This bakes everything into the image so the container never calls out to
# HuggingFace Hub at runtime — important for a NAS with no guaranteed uplink.
# ---------------------------------------------------------------------------
RUN python - <<'EOF'
import numpy as np
from kokoro import KPipeline

print("Loading Kokoro pipeline (downloads model weights)...")
pipeline = KPipeline(lang_code='a')

voices = [
    'af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
    'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella',
    'bm_george', 'bm_lewis',
]

for voice in voices:
    try:
        # Run a short synthesis to force the voice .pt file to download and cache
        for _ in pipeline("Hello.", voice=voice):
            break
        print(f"  cached: {voice}")
    except Exception as e:
        print(f"  WARNING: could not cache {voice}: {e}")

print("All Kokoro assets cached.")
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
