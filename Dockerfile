# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (ffmpeg for Whisper + others you may need)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set HuggingFace/Transformers cache directories inside container
ENV TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache \
    PYTHONPATH=/app

# Create cache + video directories
RUN mkdir -p /app/cache /app/videos

# Copy application code
COPY . /app

# Pre-download models at build time so they’re baked into the image
RUN python -c "import sys; sys.path.append('/app'); from app import load_models; load_models(); print('✅ Models downloaded and cached inside Docker image')"

# Ensure startup script is executable
RUN chmod +x /app/startup.sh

# Expose default port (Azure sets $PORT dynamically, but expose 8080 for local dev)
EXPOSE 8080

# Use startup script as entrypoint
CMD ["/bin/bash", "/app/startup.sh"]
