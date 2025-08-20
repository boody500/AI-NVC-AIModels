#!/bin/bash
set -e

echo "🚀 Running startup script for Azure..."

# Preload models (to avoid slow first request)
python - <<EOF
from transformers import T5Tokenizer, T5Model
from faster_whisper import WhisperModel

print("Preloading models...")

T5Tokenizer.from_pretrained("t5-base")
T5Model.from_pretrained("t5-base")

WhisperModel("base", device="cpu")

print("✅ Models cached successfully!")
EOF

# Start API with Gunicorn
exec gunicorn app:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
