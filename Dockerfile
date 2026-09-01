FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ensure storage directories exist
RUN mkdir -p /app/app/static/uploads

# Expose port
EXPOSE 8000

# Run database setup and launch uvicorn
CMD ["sh", "-c", "python reset_db.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
