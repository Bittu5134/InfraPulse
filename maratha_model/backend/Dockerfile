# Hugging Face Spaces — FastAPI backend
FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install minimal system deps needed by OpenCV / PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
# Install CPU-only torch/torchvision first, then the rest
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create writable dirs for SQLite and uploads
RUN mkdir -p /app/app/data/uploads && \
    chown -R appuser:appuser /app

USER appuser

# Cloud Run injects the PORT environment variable (defaults to 8080)
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
