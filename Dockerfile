FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    redis-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for audio input
RUN mkdir -p /app/audio_input

# Expose ports for Streamlit and Redis
EXPOSE 8501 6379

# Create a script to start both Redis and the application
RUN echo '#!/bin/bash\n\
redis-server --daemonize yes\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set the entry point
ENTRYPOINT ["/app/start.sh"] 