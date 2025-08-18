# ===============================
# 1. Base build stage
# ===============================
FROM python:3.11-slim AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip & install dependencies in a clean venv
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir --timeout=1000 \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy app code (needed for model preload)
COPY . .

# Pre-download models into /app/cache
RUN mkdir -p /app/cache && \
    TRANSFORMERS_CACHE=/app/cache HF_HOME=/app/cache /opt/venv/bin/python -c "\
from transformers import T5Tokenizer, T5Model; \
print('Downloading T5...'); \
T5Tokenizer.from_pretrained('t5-base'); \
T5Model.from_pretrained('t5-base'); \
print('T5 ready!')" && \
    TRANSFORMERS_CACHE=/app/cache HF_HOME=/app/cache /opt/venv/bin/python -c "\
from faster_whisper import WhisperModel; \
print('Downloading Whisper...'); \
WhisperModel('base', device='cpu', compute_type='int8'); \
print('Whisper ready!')"

# ===============================
# 2. Final runtime stage
# ===============================
FROM python:3.11-slim

# Install only runtime system deps (ffmpeg is required at runtime)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy cached models
COPY --from=builder /app/cache /app/cache

# Use venv Python by default
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy app code and startup script
COPY . .
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Create runtime directories for downloaded audio files
RUN mkdir -p /app/videos && chmod 755 /app/videos

# Expose port
EXPOSE 8080

# Start application
CMD ["/app/startup.sh"]
