# Training Script Improvements

This document outlines all the improvements made to the BEP text generation model training script.

## New Features

### 1. Padding for Variable-Length Sequences

**What it does:** Handles sequences of different lengths more efficiently using padding.

**Benefits:**
- Better batch processing
- More flexible data handling
- Improved memory efficiency

**Implementation:**
- Added `<PAD>` token (index 0) to vocabulary
- Custom `collate_fn_with_padding()` function for DataLoader
- Automatic padding to max sequence length in each batch

**Usage:**
```python
# Automatically handled in training - no changes needed
```

---

### 2. Regular Checkpoint Saving

**What it does:** Saves model checkpoints at regular intervals during training.

**Benefits:**
- Resume training if interrupted
- Track model evolution over time
- Experiment with different checkpoint states

**Implementation:**
- Checkpoints saved every N epochs (default: 10)
- Includes full training state: model, optimizer, scheduler
- Saved in `ml-service/models/checkpoint_epoch_X.pth`

**Usage:**
```bash
# Set checkpoint frequency
python train_model.py --checkpoint-every 5

# Resume from checkpoint
python train_model.py --resume-from ml-service/models/checkpoint_epoch_50.pth
```

**Checkpoint Contents:**
- Model weights
- Optimizer state
- Scheduler state
- Training/validation metrics
- Character mappings
- Epoch number

---

### 3. Weights & Biases Integration

**What it does:** Advanced experiment tracking and visualization with WandB.

**Benefits:**
- Cloud-based experiment tracking
- Better visualization than TensorBoard
- Easy comparison across runs
- Team collaboration features

**Installation:**
```bash
pip install wandb
wandb login
```

**Usage:**
```bash
# Enable WandB logging
python train_model.py --use-wandb

# Custom project and run names
python train_model.py --use-wandb \
    --wandb-project my-bep-project \
    --wandb-run-name experiment-v2
```

**Logged Metrics:**
- Training & validation loss
- Top-1 and Top-5 accuracy
- Learning rate
- Gradient norms
- Batch loss variance
- Early stopping metrics

---

### 4. Unit Tests with pytest

**What it does:** Comprehensive test suite for key functions.

**Benefits:**
- Ensure code correctness
- Catch regressions early
- Document expected behavior
- Easier refactoring

**Running Tests:**
```bash
# Install pytest
pip install pytest

# Run all tests
pytest ml-service/tests/test_train_model.py -v

# Run specific test class
pytest ml-service/tests/test_train_model.py::TestCharRNN -v

# Run with coverage
pytest ml-service/tests/test_train_model.py --cov=ml-service/scripts
```

**Test Coverage:**
- CharDataset creation and types
- Padding collate function
- CharRNN model (LSTM & GRU)
- Data loading and preprocessing
- Dataset preparation
- Text generation
- Model save/load

---

### 5. Multi-GPU Support

**What it does:** Distribute training across multiple GPUs using DataParallel.

**Benefits:**
- Faster training with multiple GPUs
- Larger effective batch sizes
- Better GPU utilization

**Requirements:**
- Multiple CUDA-capable GPUs
- PyTorch with CUDA support

**Usage:**
```bash
# Use all available GPUs
python train_model.py --multi-gpu

# Use specific GPUs (e.g., GPU 0 and 1)
python train_model.py --multi-gpu --gpu-ids "0,1"

# Multi-GPU with custom batch size
python train_model.py --multi-gpu --batch-size 2048
```

**How it Works:**
- Automatically scales batch size by number of GPUs
- Uses `nn.DataParallel` for data parallelism
- Properly saves/loads models (unwraps DataParallel)

**Example Output:**
```
🎮 Multi-GPU enabled: Using all 2 available GPUs
📦 Batch size scaled for 2 GPUs: 2048
🖥️  Using device: cuda
🎮 GPU: NVIDIA RTX 3090
💾 VRAM: 24.0 GB
🎮 GPU 1: NVIDIA RTX 3090
💾 VRAM 1: 24.0 GB
```

---

### 6. Pre-trained Model Loading & Fine-tuning

**What it does:** Load pre-trained models for transfer learning or resume training.

**Benefits:**
- Faster convergence with pre-trained weights
- Resume interrupted training
- Fine-tune on domain-specific data
- Experiment with different training strategies

**Two Modes:**

#### Resume Training
Loads everything: model, optimizer, scheduler states
```bash
python train_model.py --resume-from ml-service/models/checkpoint_epoch_50.pth
```

#### Fine-tuning
Loads only model weights, fresh optimizer
```bash
python train_model.py --finetune-from ml-service/models/best_model_checkpoint.pth
```

**Use Cases:**

1. **Resume interrupted training:**
   ```bash
   # Training crashed at epoch 47
   python train_model.py --resume-from ml-service/models/checkpoint_epoch_40.pth
   ```

2. **Fine-tune on new data:**
   ```bash
   # Train on general BEPs first
   python train_model.py --epochs 100

   # Then fine-tune on specific project BEPs
   python train_model.py --finetune-from ml-service/models/best_model_checkpoint.pth \
       --learning-rate 0.0001 --epochs 50
   ```

3. **Experiment with hyperparameters:**
   ```bash
   # Start from good checkpoint, try different learning rate
   python train_model.py --resume-from ml-service/models/checkpoint_epoch_30.pth \
       --learning-rate 0.0005
   ```

---

## Complete Training Example

Here's a comprehensive example using multiple features:

```bash
python ml-service/scripts/train_model.py \
    --epochs 200 \
    --hidden-size 512 \
    --embed-dim 128 \
    --num-layers 3 \
    --rnn-type lstm \
    --batch-size 512 \
    --learning-rate 0.001 \
    --dropout 0.3 \
    --seq-length 100 \
    --early-stopping-patience 20 \
    --lr-scheduler-patience 7 \
    --checkpoint-every 10 \
    --use-wandb \
    --wandb-project bep-generation \
    --wandb-run-name lstm-512h-3l-v1 \
    --multi-gpu \
    --temperature 0.8
```

## Advanced Usage Examples

### 1. Multi-GPU Training with WandB
```bash
python train_model.py \
    --multi-gpu \
    --use-wandb \
    --wandb-project production-bep \
    --batch-size 2048 \
    --epochs 150
```

### 2. Resume Training with Different Learning Rate
```bash
python train_model.py \
    --resume-from models/checkpoint_epoch_60.pth \
    --learning-rate 0.0005 \
    --epochs 100
```

### 3. Fine-tune Pre-trained Model
```bash
# First: Train base model
python train_model.py --epochs 100 --wandb-run-name base-model

# Then: Fine-tune on specific data
python train_model.py \
    --finetune-from models/best_model_checkpoint.pth \
    --epochs 50 \
    --learning-rate 0.0001 \
    --wandb-run-name finetuned-model
```

### 4. Distributed Training with Regular Checkpoints
```bash
python train_model.py \
    --multi-gpu \
    --gpu-ids "0,1,2,3" \
    --batch-size 4096 \
    --checkpoint-every 5 \
    --epochs 200
```

## Testing Your Installation

Run the test suite to verify everything works:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest ml-service/tests/test_train_model.py -v

# Run with coverage report
pytest ml-service/tests/test_train_model.py --cov=ml-service/scripts --cov-report=html
```

## Performance Tips

### 1. Batch Size Optimization
- **Single GPU (8GB+):** 1024
- **Single GPU (4-8GB):** 512
- **CPU:** 128
- **Multi-GPU:** Scale by number of GPUs

### 2. Multi-GPU Efficiency
- Use `--multi-gpu` with 2+ GPUs
- Increase batch size proportionally
- Monitor GPU utilization with `nvidia-smi`

### 3. WandB Best Practices
- Use descriptive run names
- Group related experiments in same project
- Tag experiments with hyperparameters
- Compare runs to find optimal settings

### 4. Checkpoint Strategy
- Save every 10 epochs for long training
- Save every 5 epochs for experimentation
- Keep best model + recent checkpoints
- Delete old checkpoints to save space

## Troubleshooting

### WandB Issues
```bash
# Not installed
pip install wandb

# Not logged in
wandb login

# Offline mode (no internet)
export WANDB_MODE=offline
```

### Multi-GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.device_count())"

# Specify GPUs explicitly
CUDA_VISIBLE_DEVICES=0,1 python train_model.py --multi-gpu
```

### Out of Memory
```bash
# Reduce batch size
python train_model.py --batch-size 256

# Reduce model size
python train_model.py --hidden-size 256 --num-layers 2
```

### Resume Training Errors
- Ensure model architecture matches checkpoint
- Check PyTorch version compatibility
- Verify checkpoint file isn't corrupted

## Migration from Old Training Script

Old command:
```bash
python train_model.py --epochs 100 --hidden-size 512
```

New equivalent with improvements:
```bash
python train_model.py \
    --epochs 100 \
    --hidden-size 512 \
    --checkpoint-every 10 \
    --use-wandb
```

All old arguments still work - new features are optional!

## Summary

| Feature | Command | Benefit |
|---------|---------|---------|
| Padding | (automatic) | Better batch handling |
| Checkpoints | `--checkpoint-every N` | Resume training |
| WandB | `--use-wandb` | Advanced tracking |
| Tests | `pytest tests/` | Code quality |
| Multi-GPU | `--multi-gpu` | Faster training |
| Resume | `--resume-from PATH` | Continue training |
| Fine-tune | `--finetune-from PATH` | Transfer learning |

---

**Last Updated:** 2025-12-27
**Python Version:** 3.8+
**PyTorch Version:** 1.9+
