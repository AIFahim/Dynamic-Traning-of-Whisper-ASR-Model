# Docker Setup for Whisper ASR Model

This document provides instructions for running the Whisper ASR model using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Building and Running with Docker Compose

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

3. Access the Streamlit application:
   Open your browser and navigate to `http://localhost:8501`

4. Stop the containers:
   ```bash
   docker-compose down
   ```

## Building and Running with Docker

If you prefer to use Docker directly:

1. Build the Docker image:
   ```bash
   docker build -t whisper-asr .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 -p 6379:6379 -v $(pwd)/audio_input:/app/audio_input --gpus all whisper-asr
   ```

3. Access the Streamlit application:
   Open your browser and navigate to `http://localhost:8501`

## Configuration

The Docker setup includes:

- Redis server running on port 6379
- Streamlit application running on port 8501
- GPU support through NVIDIA Container Toolkit
- Volume mounting for the audio_input directory

## Environment Variables

You can modify the following environment variables in the docker-compose.yml file:

- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)

## Troubleshooting

### GPU Issues

If you encounter GPU-related issues, ensure that:

1. NVIDIA Container Toolkit is properly installed
2. Your GPU drivers are up to date
3. The `--gpus all` flag is included when running the container

### Redis Connection Issues

If the application cannot connect to Redis:

1. Check if Redis is running inside the container:
   ```bash
   docker exec -it <container-id> redis-cli ping
   ```
   It should respond with "PONG"

2. Verify the Redis host and port settings in the application configuration

### Audio Processing Issues

If you encounter issues with audio processing:

1. Ensure that the audio_input directory is properly mounted
2. Check that the audio files are in a supported format (WAV or MP3)
3. Verify that the audio files have the correct permissions 