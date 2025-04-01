# Dynamic Training of Whisper ASR Model

This project implements a dynamic training approach for the Whisper ASR model, specifically designed to handle large-scale audio datasets (900hr of audio data, approximately 693k+ files) with efficient memory management and data augmentation.

## Overview

The project addresses the challenge of training on large audio datasets by implementing a dynamic training approach that:
- Processes and augments data in real-time
- Stores processed data in a Redis virtual container
- Feeds data to the model in batches during training
- Manages memory efficiently to prevent out-of-memory errors

## Project Structure

```
.
├── config.py                 # Configuration management using dataclasses
├── redis_manager.py          # Redis operations and memory management
├── data_processor.py         # Data preprocessing and feature extraction
├── training_manager.py       # Model training and evaluation
├── main.py                   # Main entry point
├── audioaugmentations.py     # Audio augmentation implementations
├── requirements.txt          # Pip dependencies
└── environment.yml           # Conda environment specification
```

### Key Components

1. **Configuration Management** (`config.py`)
   - Centralized configuration using dataclasses
   - Separate configs for Redis, Model, and Audio Augmentation
   - Type-safe configuration parameters

2. **Redis Manager** (`redis_manager.py`)
   - Handles Redis operations
   - Memory management and monitoring
   - Data storage and retrieval
   - Error handling and logging

3. **Data Processor** (`data_processor.py`)
   - Audio data preprocessing
   - Feature extraction
   - Batch preparation
   - Error handling for data processing

4. **Training Manager** (`training_manager.py`)
   - Model initialization and configuration
   - Training setup and execution
   - Evaluation metrics computation
   - Checkpoint management

5. **Audio Augmentations** (`audioaugmentations.py`)
   - Speed augmentation
   - Pitch shifting
   - Far-field effect simulation
   - Background noise addition
   - Color noise addition
   - Time and frequency masking
   - Down/upsampling effects
   - Speech enhancement

## Features

- **Dynamic Training**: Virtual container-based training procedure
- **Memory Efficient**: Processes data in batches to prevent memory overflow
- **Real-time Augmentation**: Multiple audio augmentation techniques
- **Configurable**: Easy to modify parameters through config files
- **Error Handling**: Robust error handling throughout the pipeline
- **Type Safety**: Type hints and dataclasses for better code reliability

## Setup and Installation

### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone [your-repo-url]
cd [your-repo-name]

# Create and activate conda environment
conda env create -f environment.yml
conda activate whisper-dynamic-training

# Configure Redis (if needed)
# Update config.py with your Redis settings
```

### Option 2: Using Pip
```bash
# Clone the repository
git clone [your-repo-url]
cd [your-repo-name]

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Redis (if needed)
# Update config.py with your Redis settings
```

### Redis Setup
1. Install Redis Server:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server

   # macOS
   brew install redis

   # Windows
   # Download from https://github.com/microsoftarchive/redis/releases
   ```

2. Start Redis Server:
   ```bash
   # Linux/macOS
   redis-server

   # Windows
   redis-server.exe
   ```

3. Verify Redis Connection:
   ```bash
   redis-cli ping
   # Should return "PONG"
   ```

## Usage

1. **Configuration**
   - Modify `config.py` to adjust:
     - Model parameters
     - Training settings
     - Audio augmentation parameters
     - Redis configuration

2. **Training**
   ```bash
   python main.py
   ```

## Configuration

### Model Configuration
- Model name and language settings
- Training parameters (batch size, learning rate, etc.)
- Evaluation settings
- Checkpoint management

### Audio Augmentation Configuration
- Sample rate
- Augmentation probabilities
- Augmentation parameters
- Background noise settings

### Redis Configuration
- Host and port settings
- Memory limits
- Connection parameters

## Memory Management

The system implements several memory management strategies:
- Redis virtual container for data storage
- Batch-wise processing
- Memory monitoring and limits
- Automatic cleanup of processed data

## Error Handling

The system includes comprehensive error handling:
- Data processing errors
- Redis connection issues
- Memory overflow protection
- Training interruption handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]

## Acknowledgments

- OpenAI for the Whisper model
- [Other acknowledgments]
