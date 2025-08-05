# Enhanced SAC LoRA Training for ATC Models

This directory contains an enhanced LoRA training setup specifically designed for fine-tuning Llama models on Air Traffic Control (ATC) scenarios using SAC (Soft Actor-Critic) reinforcement learning data.

## 🚀 Features

### Enhanced Training Pipeline
- **Advanced Progress Tracking**: Comprehensive tqdm progress bars for all training phases
- **Pre-Training Deduplication**: Removes duplicate data before tensor conversion for optimal memory usage
- **Enhanced Data Validation**: Robust validation with detailed filtering and quality metrics
- **Memory Optimization**: GPU memory management and garbage collection
- **Comprehensive Logging**: Detailed logs with training metrics and performance statistics

### Improved Monitoring
- **Real-time Progress Bars**: Live updates for data loading, deduplication, tokenization, and training
- **Enhanced Visualizations**: Detailed training curves with loss, learning rate, gradient norms, and overfitting indicators
- **Training Analytics**: ETA calculations, step timing, and performance metrics
- **Quality Reports**: Comprehensive data quality analysis and deduplication statistics

### Robust Data Processing
- **Smart Deduplication**: SHA-256 content hashing with configurable similarity thresholds
- **Enhanced Validation**: Improved content filtering and quality checks
- **Environment Tracking**: Detailed distribution analysis across different ATC environments
- **Format Optimization**: Llama 3.1 chat template formatting for optimal performance

## 📁 Directory Structure

```
Bsky_SAC_Finetuned/
├── config/
│   └── training_config.yaml          # Training configuration
├── data/
│   └── combined_atc_training.jsonl   # Training dataset
├── training/
│   └── train_enhanced_sac_lora.py    # Enhanced training script
├── models/                           # Output models directory
├── logs/                            # Training logs
├── requirements.txt                 # Python dependencies
├── run_training.py                  # Easy runner script
└── README_ENHANCED.md               # Enhanced training documentation
```

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **GPU Setup** (recommended):
   - CUDA-compatible GPU with at least 8GB VRAM
   - CUDA 11.8+ and corresponding PyTorch version

## 🎯 Quick Start

### Using the Runner Script (Recommended)

```bash
# Basic training with default settings
python run_training.py

# Install dependencies and train
python run_training.py --install-deps

# Custom configuration
python run_training.py --config config/custom_config.yaml --output-dir models/my_model

# Resume from checkpoint
python run_training.py --resume models/checkpoint-1000
```

### Direct Script Usage

```bash
python training/train_enhanced_sac_lora.py \
    --config config/training_config.yaml \
    --data data/combined_atc_training.jsonl \
    --output-dir models/llama3.1-bsky-sac-lora
```

## ⚙️ Configuration

The `config/training_config.yaml` file contains all training parameters with enhanced deduplication and progress tracking settings.

## 📊 Enhanced Features

### 1. Pre-Training Deduplication
The enhanced script performs deduplication **before** converting data to tensors for optimal memory usage.

### 2. Comprehensive Progress Tracking
Multiple progress bars for different phases with real-time updates and ETA calculations.

### 3. Enhanced Monitoring
Detailed callbacks for training monitoring with loss, learning rate, and gradient norm tracking.

### 4. Memory Optimization
Efficient memory management throughout training with periodic GPU cleanup.

## 🚨 Key Improvements

1. **Deduplication Before Tensor Conversion**: Saves memory and improves training efficiency
2. **Enhanced Progress Bars**: Real-time feedback on training progress with detailed metrics
3. **Comprehensive Data Validation**: Robust filtering and quality checks
4. **Detailed Logging**: Enhanced logging with training statistics and performance metrics
5. **Memory Management**: Automatic GPU memory cleanup and garbage collection

For detailed documentation, see the training script and configuration files.
