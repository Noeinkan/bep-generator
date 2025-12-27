"""
BEP Text Generation Model Training Script

This script trains an LSTM-based character-level language model on BEP documents
conforming to ISO 19650 standards. The trained model can generate contextually
appropriate text for various BEP sections.

Usage:
    python train_model.py [--epochs 100] [--hidden-size 512]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import argparse
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tensorboard_logger import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Optional WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CharDataset(Dataset):
    """PyTorch Dataset for character sequences"""

    def __init__(self, X, y):
        # Keep X as integer indices for embedding layer (not normalized floats)
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_fn_with_padding(batch, pad_idx=0):
    """Custom collate function for variable-length sequences with padding

    Args:
        batch: List of (X, y) tuples where X can be variable length
        pad_idx: Index of the PAD token (default: 0)

    Returns:
        Padded batch_X and batch_y tensors
    """
    # Separate sequences and targets
    sequences, targets = zip(*batch)

    # Pad sequences to max length in batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
    targets = torch.stack(targets)

    return padded_sequences, targets


class CharRNN(nn.Module):
    """Character-level RNN language model for text generation with embedding layer

    Supports both LSTM and GRU architectures.
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, output_size,
                 num_layers=2, rnn_type='lstm', dropout=0.3):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.embed_dim = embed_dim

        # Embedding layer: learns vector representations for each character
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN layer (LSTM or GRU)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # Default to LSTM
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length) - integer indices
        batch_size = x.size(0)

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Pass through embedding layer
        # embedded shape: (batch_size, seq_length, embed_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # Pass through RNN
        out, hidden = self.rnn(embedded, hidden)

        # Take output from last time step
        out = out[:, -1, :]
        out = self.dropout(out)

        # Pass through output layer
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        if self.rnn_type == 'gru':
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        else:  # LSTM
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


# Keep backward compatibility
CharLSTM = CharRNN


def load_training_data(data_path):
    """Load and return training text with preprocessing and error handling"""
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data file not found: {data_path}")

        # Check if file is readable
        if not os.access(data_path, os.R_OK):
            raise PermissionError(f"Cannot read training data file: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Validate text is not empty
        if not text or len(text.strip()) == 0:
            raise ValueError(f"Training data file is empty: {data_path}")

        print(f"[OK] Loaded {len(text):,} characters from {data_path}")

        # Preprocessing steps
        # 1. Convert to lowercase to reduce vocabulary size
        text = text.lower()

        # 2. Remove multiple consecutive spaces/newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' {2,}', ' ', text)  # Remove multiple spaces

        # 3. Remove special characters that don't add meaning (keep basic punctuation)
        # Keep: letters, numbers, basic punctuation, newlines
        text = re.sub(r'[^\w\s.,;:!?()\-\n\'\"]+', '', text)

        print(f"[OK] Text preprocessing completed")
        print(f"[NOTE] Total characters after preprocessing: {len(text):,}")

        # Final validation
        if len(text) < 100:
            raise ValueError(f"Training data too short (< 100 chars). Need more data for training.")

        return text

    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: {e}")
        print(f"[TIP] Please ensure the training data file exists at the specified path.")
        raise
    except PermissionError as e:
        print(f"\n[ERROR] Error: {e}")
        print(f"[TIP] Please check file permissions.")
        raise
    except UnicodeDecodeError as e:
        print(f"\n[ERROR] Error: Could not decode file as UTF-8: {e}")
        print(f"[TIP] Please ensure the file is encoded as UTF-8.")
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected error loading training data: {e}")
        raise


def prepare_dataset(text, seq_length=100, validation_split=0.2):
    """Prepare character mappings and training sequences with train/val split"""
    # Add special tokens
    PAD_TOKEN = '<PAD>'
    EOS_TOKEN = '<EOS>'
    text = text + EOS_TOKEN

    # Create character mappings (PAD must be index 0 for easier masking)
    chars = sorted(list(set(text)))
    # Ensure PAD is first (index 0) and EOS is included
    if PAD_TOKEN in chars:
        chars.remove(PAD_TOKEN)
    if EOS_TOKEN in chars:
        chars.remove(EOS_TOKEN)
    chars = [PAD_TOKEN] + chars + [EOS_TOKEN]

    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    n_vocab = len(chars)

    print(f"Total characters: {len(text)}")
    print(f"Unique characters: {n_vocab}")
    print(f"Sample characters: {chars[:20]}")
    print(f"PAD token: '{PAD_TOKEN}' (index: {char_to_int[PAD_TOKEN]})")
    print(f"EOS token: '{EOS_TOKEN}' (index: {char_to_int[EOS_TOKEN]})")

    # Create training sequences
    dataX = []
    dataY = []
    for i in range(0, len(text) - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)

    # Split into training and validation sets
    split_idx = int(n_patterns * (1 - validation_split))

    train_X = dataX[:split_idx]
    train_Y = dataY[:split_idx]
    val_X = dataX[split_idx:]
    val_Y = dataY[split_idx:]

    print(f"Total sequences: {n_patterns}")
    print(f"Training sequences: {len(train_X)} ({(1-validation_split)*100:.0f}%)")
    print(f"Validation sequences: {len(val_X)} ({validation_split*100:.0f}%)")

    return train_X, train_Y, val_X, val_Y, char_to_int, int_to_char, n_vocab


def train_model(model, train_X, train_y, val_X, val_y, n_vocab, epochs=100, learning_rate=0.001,
                device='cpu', batch_size=128, char_to_int=None, int_to_char=None, models_dir=None,
                early_stopping_patience=15, lr_scheduler_patience=5, checkpoint_every=10, use_wandb=False,
                wandb_project='bep-text-generation', wandb_run_name=None, resume_checkpoint=None):
    """Train the RNN model with early stopping, LR scheduling, and comprehensive monitoring

    Args:
        checkpoint_every: Save checkpoint every N epochs (default: 10)
        use_wandb: Enable Weights & Biases logging (default: False)
        wandb_project: WandB project name (default: 'bep-text-generation')
        wandb_run_name: WandB run name (default: auto-generated)
        resume_checkpoint: Path to checkpoint to resume training from (default: None)
    """

    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(__file__).parent.parent / 'runs' / f'bep_training_{timestamp}'
    logger = TensorBoardLogger(log_dir=str(log_dir), auto_start=True, port=6006)

    # Setup WandB if enabled
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or f'bep_training_{timestamp}',
                config={
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'rnn_type': model.rnn_type,
                    'embed_dim': model.embed_dim,
                    'hidden_size': model.hidden_size,
                    'num_layers': model.num_layers,
                    'vocab_size': n_vocab,
                    'early_stopping_patience': early_stopping_patience,
                    'lr_scheduler_patience': lr_scheduler_patience,
                    'device': str(device),
                }
            )
            print(f"[OK] Weights & Biases initialized: {wandb.run.url}")
        except Exception as e:
            print(f"[WARNING]  Could not initialize WandB: {e}")
            wandb_run = None
    elif use_wandb and not WANDB_AVAILABLE:
        print(f"[WARNING]  WandB requested but not installed. Install with: pip install wandb")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=lr_scheduler_patience
    )

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = None
    start_epoch = 0

    # Load checkpoint if resuming
    if resume_checkpoint:
        start_epoch, best_val_loss = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, device
        )
        print(f"📍 Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")

    # Enable mixed precision training for GPU speedup
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    # Determine optimal number of workers based on CPU cores
    num_workers = min(8, os.cpu_count() or 4) if device.type == 'cuda' else 0

    print(f"[START] Training on device: {device}")
    print(f"[CHART] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[BATCH] Batch size: {batch_size}")
    print(f"[FAST] Mixed precision (AMP): {use_amp}")
    print(f"[WORKERS] DataLoader workers: {num_workers}")

    # Log hyperparameters to TensorBoard
    logger.log_hyperparameters({
        'epochs': epochs,
        'learning_rate': learning_rate,
        'rnn_type': model.rnn_type.upper(),
        'embed_dim': model.embed_dim,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'batch_size': batch_size,
        'device': str(device),
        'mixed_precision': use_amp,
        'dataloader_workers': num_workers,
        'vocab_size': n_vocab,
        'total_parameters': f"{sum(p.numel() for p in model.parameters()):,}",
        'early_stopping_patience': early_stopping_patience,
        'lr_scheduler_patience': lr_scheduler_patience,
    })

    # Convert to numpy arrays (integers for embedding layer)
    train_X = np.array(train_X, dtype=np.int64)
    train_y = np.array(train_y, dtype=np.int64)
    val_X = np.array(val_X, dtype=np.int64)
    val_y = np.array(val_y, dtype=np.int64)

    # Create Dataset and DataLoader for training
    train_dataset = CharDataset(train_X, train_y)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )

    # Create Dataset and DataLoader for validation
    val_dataset = CharDataset(val_X, val_y)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )

    n_batches = len(train_dataloader)

    print(f"\n[RUNNING] Starting training with {n_batches} batches per epoch...")
    print(f"[PATIENCE] Early stopping patience: {early_stopping_patience} epochs")
    print(f"[CHART] LR scheduler patience: {lr_scheduler_patience} epochs")
    if start_epoch > 0:
        print(f"[RESUME] Resuming from epoch {start_epoch}")
    print()

    # Training loop with progress bar
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress", unit="epoch", initial=start_epoch, total=epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        batch_losses = []

        # Batch progress bar
        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}",
                         leave=False, unit="batch")

        for batch_X, batch_y in batch_pbar:
            # Move to device
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Forward pass with mixed precision
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    output, _ = model(batch_X)
                    loss = criterion(output, batch_y)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output, _ = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss
            batch_losses.append(batch_loss)

            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

        avg_train_loss = total_train_loss / n_batches

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if use_amp:
                    with autocast():
                        output, _ = model(batch_X)
                        loss = criterion(output, batch_y)
                else:
                    output, _ = model(batch_X)
                    loss = criterion(output, batch_y)

                total_val_loss += loss.item()

                # Calculate top-1 and top-5 accuracy
                _, pred_top1 = output.max(1)
                _, pred_top5 = output.topk(5, 1, True, True)

                correct_top1 += pred_top1.eq(batch_y).sum().item()
                correct_top5 += pred_top5.eq(batch_y.view(-1, 1).expand_as(pred_top5)).sum().item()
                total_samples += batch_y.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_acc_top1 = 100.0 * correct_top1 / total_samples
        val_acc_top5 = 100.0 * correct_top5 / total_samples

        # Track best validation loss and save best model (Early Stopping)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            # Save best model checkpoint
            if models_dir is not None:
                best_model_path = models_dir / 'best_model_checkpoint.pth'
                # Unwrap DataParallel if necessary
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'hidden_size': model_to_save.hidden_size,
                    'num_layers': model_to_save.num_layers,
                    'embed_dim': model_to_save.embed_dim,
                    'rnn_type': model_to_save.rnn_type,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_acc_top1': val_acc_top1,
                    'val_acc_top5': val_acc_top5,
                }, best_model_path)
        else:
            epochs_without_improvement += 1

        # Learning rate scheduler step (based on validation loss)
        scheduler.step(avg_val_loss)

        # Save regular checkpoint every N epochs
        if (epoch + 1) % checkpoint_every == 0 and models_dir is not None:
            checkpoint_path = models_dir / f'checkpoint_epoch_{epoch+1}.pth'
            # Unwrap DataParallel if necessary
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'hidden_size': model_to_save.hidden_size,
                'num_layers': model_to_save.num_layers,
                'embed_dim': model_to_save.embed_dim,
                'rnn_type': model_to_save.rnn_type,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_acc_top1': val_acc_top1,
                'val_acc_top5': val_acc_top5,
                'char_to_int': char_to_int,
                'int_to_char': {int(k): v for k, v in int_to_char.items()},
                'n_vocab': n_vocab,
            }, checkpoint_path)
            tqdm.write(f'[SAVE] Checkpoint saved: {checkpoint_path}')

        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log comprehensive metrics to TensorBoard
        logger.log_epoch_metrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_acc_top1=val_acc_top1,
            val_acc_top5=val_acc_top5,
            learning_rate=optimizer.param_groups[0]['lr'],
            gradient_norm=total_norm,
            batch_losses=batch_losses,
            epochs_without_improvement=epochs_without_improvement,
            best_val_loss=best_val_loss
        )

        # Log to WandB if enabled
        if wandb_run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'val_acc_top1': val_acc_top1,
                'val_acc_top5': val_acc_top5,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'gradient_norm': total_norm,
                'epochs_without_improvement': epochs_without_improvement,
                'batch_loss_std': np.std(batch_losses),
            })

        # Generate sample text every 5 epochs for qualitative assessment
        if (epoch + 1) % 5 == 0 and char_to_int is not None and int_to_char is not None:
            sample_text = generate_sample_during_training(
                model, char_to_int, int_to_char, n_vocab,
                device=device, max_length=150
            )
            logger.log_text_sample(sample_text, epoch, tag='Generated_Samples')

            tqdm.write(f'\n{"="*60}')
            tqdm.write(f'Epoch [{epoch+1}/{epochs}]')
            tqdm.write(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Best Val: {best_val_loss:.4f}')
            tqdm.write(f'Val Acc (Top-1): {val_acc_top1:.2f}% | Val Acc (Top-5): {val_acc_top5:.2f}%')
            tqdm.write(f'LR: {optimizer.param_groups[0]["lr"]:.2e} | No Improvement: {epochs_without_improvement} epochs')
            tqdm.write(f'Sample: {sample_text[:100]}...')
            tqdm.write(f'{"="*60}\n')

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            tqdm.write(f'\n{"="*60}')
            tqdm.write(f'[STOP]  Early stopping triggered after {epoch+1} epochs')
            tqdm.write(f'No improvement for {early_stopping_patience} consecutive epochs')
            tqdm.write(f'Best validation loss: {best_val_loss:.4f}')
            tqdm.write(f'{"="*60}\n')
            break

    logger.close()

    # Finalize WandB if used
    if wandb_run is not None:
        wandb.finish()
        print("[OK] Weights & Biases run completed")

    # Training completion notification
    print(f"\n\n{'='*60}")
    print(f"[SUCCESS] TRAINING COMPLETATO! [SUCCESS]")
    print(f"{'='*60}")
    print(f"[OK] Epochs completate: {epochs}")
    print(f"[OK] Train Loss finale: {avg_train_loss:.4f}")
    print(f"[OK] Val Loss finale: {avg_val_loss:.4f}")
    print(f"[OK] Best Val Loss: {best_val_loss:.4f}")
    print(f"[OK] TensorBoard logs: {log_dir}")
    print(f"[CHART] Dashboard disponibile su: http://localhost:6006")
    if best_model_path:
        print(f"[SAVE] Best model salvato in: {best_model_path}")
    print(f"{'='*60}\n")

    # Try to show a system notification (Windows)
    try:
        if sys.platform == 'win32':
            subprocess.run([
                'powershell', '-Command',
                f'New-BurntToastNotification -Text "Training Completato!", "Il modello BEP ha finito il training con val loss: {best_val_loss:.4f}"'
            ], check=False, capture_output=True)
    except:
        pass  # Silently fail if notification doesn't work

    return model


def generate_sample_during_training(model, char_to_int, int_to_char, n_vocab,
                                   device='cpu', max_length=150, temperature=0.8, use_greedy=False):
    """Generate sample text during training for monitoring

    Args:
        model: The trained model
        char_to_int: Character to integer mapping
        int_to_char: Integer to character mapping
        n_vocab: Vocabulary size
        device: Device to run on
        max_length: Maximum generation length
        temperature: Sampling temperature (lower = more deterministic, higher = more creative)
                    Only used when use_greedy=False
        use_greedy: If True, use greedy decoding (argmax). If False, use temperature sampling
    """
    model.eval()
    prompts = [
        "the bep establishes",
        "information manager is responsible for",
        "the primary objectives include"
    ]

    # Pick a random prompt
    start_text = np.random.choice(prompts)

    with torch.no_grad():
        # Use integer indices for embedding layer
        inputs = [char_to_int.get(ch, 0) for ch in start_text]
        inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_length)
        inputs = inputs.to(device)

        result = start_text
        hidden = None

        for _ in range(max_length):
            output, hidden = model(inputs, hidden)

            if use_greedy:
                # Greedy decoding: select most probable character
                char_idx = output.argmax(dim=1).item()
            else:
                # Temperature sampling: scale logits by temperature before softmax
                scaled_output = output / temperature
                prob = torch.softmax(scaled_output, dim=1).cpu().detach().numpy()
                char_idx = np.random.choice(range(n_vocab), p=prob[0])

            char = int_to_char[char_idx]
            result += char

            # Stop at sentence end
            if char == '.' and len(result) > 50:
                break

            # Update input with new character (integer index)
            new_input = torch.tensor([[char_idx]], dtype=torch.long).to(device)  # Shape: (1, 1)
            inputs = new_input

    model.train()
    return result


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load a checkpoint for resuming training or fine-tuning

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: Optional optimizer to load state into (for resuming)
        scheduler: Optional scheduler to load state into (for resuming)
        device: Device to load the model on

    Returns:
        Tuple of (start_epoch, best_val_loss) for resuming training,
        or (0, inf) for fine-tuning
    """
    try:
        print(f"\n[DOWNLOAD] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OK] Model weights loaded")

        start_epoch = 0
        best_val_loss = float('inf')

        # If optimizer and scheduler are provided, load their states (resuming training)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"[OK] Optimizer state loaded")

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"[OK] Scheduler state loaded")

        # Get epoch and validation loss if available
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"[OK] Resuming from epoch {start_epoch}")

        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            print(f"[OK] Previous best validation loss: {best_val_loss:.4f}")

        return start_epoch, best_val_loss

    except FileNotFoundError:
        print(f"\n[ERROR] Checkpoint file not found: {checkpoint_path}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Error loading checkpoint: {e}")
        raise


def save_model(model, char_to_int, int_to_char, save_dir):
    """Save model and character mappings with error handling

    Handles both regular models and DataParallel wrapped models.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Unwrap DataParallel if necessary
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model

        # Save model weights
        model_path = os.path.join(save_dir, 'bep_model.pth')
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'hidden_size': model_to_save.hidden_size,
            'num_layers': model_to_save.num_layers,
            'embed_dim': model_to_save.embed_dim,
            'rnn_type': model_to_save.rnn_type,
        }, model_path)

        # Verify model was saved
        if not os.path.exists(model_path):
            raise IOError(f"Model file was not created: {model_path}")

        # Save character mappings
        mappings_path = os.path.join(save_dir, 'char_mappings.json')
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_int': char_to_int,
                'int_to_char': {int(k): v for k, v in int_to_char.items()},
                'n_vocab': len(char_to_int)
            }, f, ensure_ascii=False, indent=2)

        # Verify mappings were saved
        if not os.path.exists(mappings_path):
            raise IOError(f"Mappings file was not created: {mappings_path}")

        # Get file sizes
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        mappings_size = os.path.getsize(mappings_path) / 1024  # KB

        print(f"\n[OK] Model saved to: {model_path} ({model_size:.2f} MB)")
        print(f"[OK] Mappings saved to: {mappings_path} ({mappings_size:.2f} KB)")

    except PermissionError as e:
        print(f"\n[ERROR] Error: Cannot write to directory {save_dir}: {e}")
        print(f"[TIP] Please check write permissions for the models directory.")
        raise
    except IOError as e:
        print(f"\n[ERROR] Error saving model: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during model save: {e}")
        raise


def generate_sample_text(model, char_to_int, int_to_char, n_vocab,
                         start_text="the bep establishes", length=200, device='cpu',
                         temperature=1.0, use_greedy=False, top_k=0, top_p=0.0):
    """Generate sample text with advanced sampling strategies

    Args:
        model: The trained model
        char_to_int: Character to integer mapping
        int_to_char: Integer to character mapping
        n_vocab: Vocabulary size
        start_text: Initial prompt text
        length: Maximum generation length
        device: Device to run on
        temperature: Sampling temperature (0.1-2.0). Lower = more deterministic, higher = more creative
        use_greedy: If True, always pick most probable character (ignores temperature)
        top_k: If > 0, only sample from top k most probable characters (nucleus sampling)
        top_p: If > 0.0, sample from smallest set of chars whose cumulative prob >= top_p
    """
    model.eval()
    with torch.no_grad():
        # Use integer indices for embedding layer
        inputs = [char_to_int.get(ch, 0) for ch in start_text]
        inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_length)
        inputs = inputs.to(device)

        result = start_text
        hidden = None

        for _ in range(length):
            output, hidden = model(inputs, hidden)

            if use_greedy:
                # Greedy decoding: always select most probable character
                char_idx = output.argmax(dim=1).item()
            else:
                # Temperature sampling with optional top-k or top-p filtering
                logits = output / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, n_vocab))
                    # Set all non-top-k values to -inf
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_values)
                    logits = logits_filtered

                # Top-p (nucleus) filtering
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=1), dim=1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift right to keep first token above threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    # Create mask and apply
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample from filtered distribution
                prob = torch.softmax(logits, dim=1).cpu().detach().numpy()
                char_idx = np.random.choice(range(n_vocab), p=prob[0])

            char = int_to_char[char_idx]
            result += char

            # Update input with new character (integer index)
            new_input = torch.tensor([[char_idx]], dtype=torch.long).to(device)  # Shape: (1, 1)
            inputs = new_input

        return result


def main():
    parser = argparse.ArgumentParser(description='Train BEP text generation model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden-size', type=int, default=512, help='RNN hidden size')
    parser.add_argument('--embed-dim', type=int, default=128, help='Character embedding dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--rnn-type', type=str, default='lstm', choices=['lstm', 'gru'],
                       help='Type of RNN to use (lstm or gru)')
    parser.add_argument('--seq-length', type=int, default=100, help='Sequence length')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto-detected based on device)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--lr-scheduler-patience', type=int, default=5,
                       help='Number of epochs to wait before reducing learning rate')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')

    # Logging parameters
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='bep-text-generation',
                       help='WandB project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='WandB run name (auto-generated if not specified)')

    # Multi-GPU parameters
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use all available GPUs with DataParallel')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='Comma-separated GPU IDs to use (e.g., "0,1,2"). If not specified, uses all available GPUs')

    # Pre-trained model parameters
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--finetune-from', type=str, default=None,
                       help='Path to pre-trained model to finetune (loads only weights, not optimizer state)')

    # Text generation parameters
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature for text generation (0.1-2.0)')
    parser.add_argument('--use-greedy', action='store_true',
                       help='Use greedy decoding (argmax) instead of sampling')
    parser.add_argument('--top-k', type=int, default=0,
                       help='Top-k sampling: only sample from k most probable chars (0=disabled)')
    parser.add_argument('--top-p', type=float, default=0.0,
                       help='Nucleus sampling: sample from smallest set with cumulative prob >= p (0=disabled)')

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'data' / 'training_data.txt'
    models_dir = project_root / 'models'

    print("="*60)
    print("BEP Text Generation Model Training")
    print("="*60)

    # Check for GPU and enable optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enable cuDNN benchmark for faster LSTM/GRU training on GPU
    # This finds the best algorithm for your hardware, speeding up training
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(f"[OK] cuDNN benchmark enabled for faster GPU training")
    else:
        # CPU optimizations: set number of threads
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        print(f"[OK] CPU threads set to {num_threads} for faster training")

    # Auto-detect optimal batch size based on device
    if args.batch_size is None:
        if device.type == 'cuda':
            # Check available VRAM
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_mem_gb >= 8:
                batch_size = 1024  # Large batch for 8GB+ VRAM
            elif gpu_mem_gb >= 4:
                batch_size = 512
            else:
                batch_size = 256
        else:
            batch_size = 128  # Conservative for CPU
    else:
        batch_size = args.batch_size

    # Multi-GPU setup
    gpu_ids = None
    num_gpus = 1
    if args.multi_gpu and torch.cuda.is_available():
        if args.gpu_ids:
            # Parse GPU IDs from comma-separated string
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            num_gpus = len(gpu_ids)
            print(f"[GPU] Multi-GPU enabled: Using GPUs {gpu_ids}")
        else:
            # Use all available GPUs
            num_gpus = torch.cuda.device_count()
            gpu_ids = list(range(num_gpus))
            print(f"[GPU] Multi-GPU enabled: Using all {num_gpus} available GPUs")

        # Adjust batch size for multi-GPU
        if args.batch_size is None:
            batch_size = batch_size * num_gpus
            print(f"[BATCH] Batch size scaled for {num_gpus} GPUs: {batch_size}")
    elif args.multi_gpu and not torch.cuda.is_available():
        print("[WARNING]  Multi-GPU requested but CUDA not available. Using CPU.")

    print(f"[DEVICE]  Using device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] GPU: {gpu_name}")
        print(f"[SAVE] VRAM: {gpu_mem:.1f} GB")
        if num_gpus > 1:
            for i in range(1, num_gpus):
                gpu_name_i = torch.cuda.get_device_name(i)
                gpu_mem_i = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"[GPU] GPU {i}: {gpu_name_i}")
                print(f"[SAVE] VRAM {i}: {gpu_mem_i:.1f} GB")
    print(f"[BATCH] Batch size: {batch_size}")

    # Load and prepare data with error handling
    try:
        print("\n[LOAD] Loading training data...")
        text = load_training_data(data_path)

        print("\n[CONFIG] Preparing dataset...")
        train_X, train_Y, val_X, val_Y, char_to_int, int_to_char, n_vocab = prepare_dataset(text, args.seq_length)

        # Validate we have enough data
        if len(train_X) < 100:
            raise ValueError(f"Training set too small ({len(train_X)} sequences). Need at least 100 sequences.")
        if len(val_X) < 10:
            raise ValueError(f"Validation set too small ({len(val_X)} sequences). Need at least 10 sequences.")

    except (FileNotFoundError, PermissionError, ValueError, UnicodeDecodeError) as e:
        print(f"\n[ERROR] Fatal error during data loading: {e}")
        print(f"\n[TIP] Training cannot continue without valid data. Exiting...")
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create model with error handling
    try:
        print("\n[MODEL] Initializing model...")
        print(f"Model architecture: {args.rnn_type.upper()}")
        print(f"Embedding dimension: {args.embed_dim}")
        print(f"Hidden size: {args.hidden_size}")
        print(f"Number of layers: {args.num_layers}")
        print(f"Dropout: {args.dropout}")

        model = CharRNN(
            vocab_size=n_vocab,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            output_size=n_vocab,
            num_layers=args.num_layers,
            rnn_type=args.rnn_type,
            dropout=args.dropout
        ).to(device)

        # Wrap model with DataParallel for multi-GPU if requested
        if args.multi_gpu and num_gpus > 1 and torch.cuda.is_available():
            if gpu_ids:
                model = nn.DataParallel(model, device_ids=gpu_ids)
            else:
                model = nn.DataParallel(model)
            print(f"[OK] Model wrapped with DataParallel for {num_gpus} GPUs")

        print(f"[OK] Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Load pre-trained weights if specified
        if args.resume_from:
            # Resume training: load model, optimizer, and scheduler states
            print("\n[RESUME] Resuming training from checkpoint...")
            # Note: Optimizer and scheduler will be created in train_model, so we'll handle this there
            load_checkpoint(args.resume_from, model, device=device)
        elif args.finetune_from:
            # Fine-tuning: load only model weights
            print("\n[FINETUNE] Fine-tuning from pre-trained model...")
            load_checkpoint(args.finetune_from, model, device=device)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[ERROR] GPU out of memory error: {e}")
            print(f"[TIP] Try reducing --batch-size, --hidden-size, or --embed-dim")
            print(f"[TIP] Current settings: batch_size={batch_size}, hidden_size={args.hidden_size}, embed_dim={args.embed_dim}")
        else:
            print(f"\n[ERROR] Runtime error creating model: {e}")
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error creating model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Train model with error handling
    try:
        print("\n[TRAIN] Starting training...")
        model = train_model(
            model, train_X, train_Y, val_X, val_Y, n_vocab,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            batch_size=batch_size,
            char_to_int=char_to_int,
            int_to_char=int_to_char,
            models_dir=models_dir,
            early_stopping_patience=args.early_stopping_patience,
            lr_scheduler_patience=args.lr_scheduler_patience,
            checkpoint_every=args.checkpoint_every,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            resume_checkpoint=args.resume_from
        )

    except KeyboardInterrupt:
        print(f"\n\n[WARNING]  Training interrupted by user (Ctrl+C)")
        print(f"[TIP] Attempting to save current model state...")
        # Model will be saved below if it exists
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[ERROR] GPU out of memory during training: {e}")
            print(f"[TIP] Try reducing --batch-size from {batch_size}")
            print(f"[TIP] Or reduce model size (--hidden-size, --embed-dim, --num-layers)")
        else:
            print(f"\n[ERROR] Runtime error during training: {e}")
            import traceback
            traceback.print_exc()
        return
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save model with error handling
    try:
        print("\n[SAVE] Saving model...")
        save_model(model, char_to_int, int_to_char, models_dir)
    except Exception as e:
        print(f"\n[WARNING]  Warning: Could not save model: {e}")
        print(f"[TIP] Model training completed but save failed. Check disk space and permissions.")

    # Generate sample text with different strategies
    print("\n" + "="*60)
    print("Sample Generated Text")
    print("="*60)

    # Show generation strategy being used
    if args.use_greedy:
        print("Generation mode: Greedy decoding (argmax)")
    else:
        print(f"Generation mode: Temperature sampling (T={args.temperature})")
        if args.top_k > 0:
            print(f"  + Top-k filtering (k={args.top_k})")
        if args.top_p > 0.0:
            print(f"  + Nucleus sampling (p={args.top_p})")
    print("-" * 60)

    sample = generate_sample_text(
        model, char_to_int, int_to_char, n_vocab,
        start_text="the bep establishes",
        length=300,
        device=device,
        temperature=args.temperature,
        use_greedy=args.use_greedy,
        top_k=args.top_k,
        top_p=args.top_p
    )
    print(sample)
    print("="*60)

    # Final completion message
    print("\n" + "***"*30)
    print("[SUCCESS]" + " "*26 + "TRAINING COMPLETATO!" + " "*26 + "[SUCCESS]")
    print("***"*30)
    print("\n[OK] Il modello è stato salvato con successo!")
    print("[OK] TensorBoard rimane aperto per visualizzare i risultati")
    print("[CHART] Dashboard: http://localhost:6006")
    print("\n[TIP] Premi CTRL+C per chiudere tutto\n")


if __name__ == '__main__':
    main()
