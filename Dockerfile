# Lightweight OCR Service - Optimized for Railway
# Size: ~250MB (vs ~4GB+ with PyTorch/PaddleOCR)

FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/uploads /app/temp

# Environment - Railway sets PORT dynamically
ENV PYTHONUNBUFFERED=1
ENV DISABLE_MODEL_SOURCE_CHECK=True

# Use shell form to allow $PORT variable expansion
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
