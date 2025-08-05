# RTX 5070 Ti Performance Optimization Guide

## Quick Fixes for Slow Training

### 1. Check Your Configuration
The training speed issues were likely caused by:

- **Too small batch size (4)** → Increased to 8-12
- **Too long sequences (2048)** → Reduced to 1024 
- **Conservative LoRA rank (16)** → Increased to 32
- **Excessive memory cleanup** → Reduced frequency
- **Missing Flash Attention** → Added fallback
- **Suboptimal training arguments** → Simplified and optimized

### 2. Hardware Optimization Checklist

#### GPU Settings (RTX 5070 Ti specific):
```yaml
hardware:
  device: "cuda"
  fp16: false
  bf16: true                    # Better for RTX 5070 Ti
  gradient_checkpointing: false # Disable for more speed
  dataloader_num_workers: 4-8   # Tune based on your CPU
  pin_memory: true
  persistent_workers: true
```

#### Training Settings:
```yaml
training:
  learning_rate: 0.0001        # Increased from 0.00001
  batch_size: 8               # Increased from 4
  gradient_accumulation_steps: 2 # Reduced from 4
  logging_steps: 100          # Less frequent logging
  eval_steps: 500             # Less frequent evaluation
```

#### Data Settings:
```yaml
data:
  max_sequence_length: 1024   # Reduced from 2048
```

### 3. Installation Commands

```bash
# Install Flash Attention for maximum speed (optional but recommended)
pip install flash-attn --no-build-isolation

# Or install everything with optimizations:
pip install -r requirements.txt
```

### 4. Performance Testing

Run these scripts to test your setup:

```bash
# Test GPU capabilities
python check_flash_attention.py

# Test model loading and training speed
python test_gpu_performance.py

# Test different configurations
python test_performance.py

# Run optimized training
python run_training.py --install-deps
```

### 5. Expected Performance

With RTX 5070 Ti (16GB VRAM), you should expect:

- **Batch size**: 8-12 (depending on sequence length)
- **Sequence length**: 1024-2048 tokens
- **Training speed**: 1-3 seconds per step
- **Memory usage**: 12-15GB VRAM
- **Total training time**: 2-4 hours for 3 epochs

### 6. Troubleshooting Common Issues

#### Out of Memory (OOM)
```yaml
# Reduce these settings:
training:
  batch_size: 4              # Reduce batch size
  gradient_accumulation_steps: 4 # Increase to maintain effective batch
data:
  max_sequence_length: 512   # Reduce sequence length
hardware:
  gradient_checkpointing: true # Enable to save memory
```

#### Very Slow Training
```bash
# Check if using CPU instead of GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU utilization
nvidia-smi

# Run performance test
python test_gpu_performance.py
```

#### Flash Attention Issues
```bash
# If flash-attn fails to install:
pip install flash-attn --no-build-isolation

# If still fails, the code will fallback to eager attention (slower but works)
```

### 7. Monitoring Training

Watch these indicators:
- **GPU utilization**: Should be >90%
- **Memory usage**: Should be 12-15GB (close to limit but not OOM)
- **Step time**: Should be 1-3 seconds per step
- **Loss**: Should decrease steadily

### 8. Expected Training Output

```
[LOADING] Loading model (optimized for RTX 5070 Ti)...
[SPEED] Using Flash Attention 2 for optimal performance
[DATA] Model Parameters:
   Trainable: 41,943,040 (0.92%)
   Total: 4,582,543,360
[TOKENIZE] Tokenizing training data: 100%|█| 9680/9680 [00:02<00:00, 4000it/s]
[TARGET] Starting training process...
Training Progress: 2%|█| 45/2420 [00:45<39:20, 1.00s/it, loss=2.1234]
```

### 9. Quick Configuration Files

**For Maximum Speed** (if you have enough VRAM):
```yaml
training:
  batch_size: 12
  gradient_accumulation_steps: 1
data:
  max_sequence_length: 1024
lora:
  rank: 32
hardware:
  gradient_checkpointing: false
  bf16: true
```

**For Stability** (if you get OOM errors):
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4
data:
  max_sequence_length: 512
lora:
  rank: 16
hardware:
  gradient_checkpointing: true
  bf16: true
```

Run the performance test script to find your optimal settings!
