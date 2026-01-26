FROM python:3.10-slim

# Install system dependencies for OCR and ML
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    poppler-utils \
    wget \
    curl \
    pkg-config \
    python3-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install NumPy 1.x first to prevent NumPy 2.x installation
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# Install PyTorch with CPU support
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp /app/models

# Copy application code
COPY . .

# Set environment variable to skip model connectivity checks
ENV DISABLE_MODEL_SOURCE_CHECK=True

# Expose port
EXPOSE 8000

# Run the application with single worker initially for stability
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]