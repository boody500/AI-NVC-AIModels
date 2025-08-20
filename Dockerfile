# Base image
FROM python:3.10-slim

# Set environment vars
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache \
    PATH="/root/.local/bin:$PATH"

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create workdir
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

# Expose port
EXPOSE 8000

# Start using startup.sh
CMD ["./startup.sh"]
