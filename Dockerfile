FROM python:3.11-slim

# Set environment variables for Python, PyTorch thread capping, and single worker stability
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TORCH_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

# Install lightweight system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install CPU-only PyTorch to minimize image footprint and RAM overhead
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure upload directory exists
RUN mkdir -p /app/app/static/uploads

# Expose port 8003 for Hack Club container reverse proxy
EXPOSE 8003

# Hetzner container port binding (Port 8003) with single worker for zero OOM crash guarantee
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8003} --workers 1"]
