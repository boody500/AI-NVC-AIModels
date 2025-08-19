#!/bin/bash

# Azure App Service startup script for YouTube Transcript Analysis API

set -e  # Exit on any error

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
mkdir -p /app/videos
mkdir -p /app/cache
chmod 755 /app/videos
chmod 755 /app/cache

# Skip model preloading (already done at Docker build stage)
echo "✅ Models already cached in Docker image, skipping preload."

echo "=== Starting Gunicorn server ==="
echo "Binding to 0.0.0.0:${PORT:-8080}"
echo "Workers: 1, Threads: 4, Timeout: 600s"

# Start the application with Gunicorn
exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --worker-class sync \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --worker-tmp-dir /dev/shm \
    app:app
