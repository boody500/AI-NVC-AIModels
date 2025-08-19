#!/bin/bash

# Azure App Service startup script for YouTube Transcript Analysis API

set -eu  # Exit on error or undefined variable

echo "=== Starting YouTube Transcript Analysis API ==="
echo "Current working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Available memory: $(free -h)"
echo "Available disk space: $(df -h /)"

# Set up environment
export PYTHONPATH="/app:$PYTHONPATH"
export TRANSFORMERS_CACHE="/app/cache"
export HF_HOME="/app/cache"

# Create required directories if they don't exist
echo "Creating required directories..."
mkdir -p /app/videos /app/cache

# Pre-download models to avoid first-request delays
echo "Pre-loading models..."
python -c "
import sys
sys.path.append('/app')
from app import load_models
print('Loading T5 and Whisper models...')
load_models()
print('Models loaded successfully!')
" || echo "⚠️ Warning: Model preloading failed, will load on first request"

echo "=== Starting Gunicorn server ==="
# Start the application with Gunicorn on the dynamic Azure PORT
exec gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 4 \
    --timeout 3800 \
    --worker-class sync \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --worker-tmp-dir /dev/shm \
    app:app
