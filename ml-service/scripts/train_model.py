"""
BEP Text Generation Model Training Script

This script trains an LSTM-based character-level language model on BEP documents
conforming to ISO 19650 standards. The trained model can generate contextually
appropriate text for various BEP sections.

Usage:
    python train_model.py [--epochs 50] [--hidden-size 512] [--step 3]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time


class DashboardUpdater:
    """Helper class to update training dashboard state"""

    def __init__(self, state_file):
        self.state_file = Path(state_file)
        self.state = {
            'status': 'idle',
            'current_epoch': 0,
            'total_epochs': 0,
            'current_batch': 0,
            'total_batches': 0,
            'current_loss': 0.0,
            'avg_loss': 0.0,
            'best_loss': float('inf'),
            'losses': [],
            'batch_losses': [],
            'epoch_times': [],
            'device': 'unknown',
            'start_time': None,
            'last_update': None,
            'training_params': {}
        }

    def update(self, **kwargs):
        """Update state and save to file"""
        self.state.update(kwargs)
        self.state['last_update'] = time.time()
        self._save()

    def _save(self):
        """Save state to JSON file"""
        try:
            self.state['last_update'] = time.time()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            # Don't crash training if dashboard update fails
            pass

    def start_training(self, total_epochs, device, **params):
        """Mark training as started"""
        self.state['status'] = 'running'
        self.state['total_epochs'] = total_epochs
        self.state['device'] = str(device)
        self.state['start_time'] = time.time()
        self.state['training_params'] = params
        self.state['losses'] = []
        self.state['batch_losses'] = []
        self.state['epoch_times'] = []
        self._save()

    def update_batch(self, epoch, batch, total_batches, loss):
        """Update batch progress"""
        self.state['current_epoch'] = epoch
        self.state['current_batch'] = batch
        self.state['total_batches'] = total_batches
        self.state['current_loss'] = loss

        # Keep only recent batch losses (last 100)
        self.state['batch_losses'].append(loss)
        if len(self.state['batch_losses']) > 100:
            self.state['batch_losses'] = self.state['batch_losses'][-100:]

        self._save()

    def update_epoch(self, epoch, avg_loss, best_loss, epoch_time):
        """Update epoch completion"""
        self.state['current_epoch'] = epoch
        self.state['avg_loss'] = avg_loss
        self.state['best_loss'] = best_loss
        self.state['losses'].append([epoch, avg_loss])
        self.state['epoch_times'].append(epoch_time)
        self.state['batch_losses'] = []  # Reset for next epoch
        self._save()

    def complete_training(self):
        """Mark training as completed"""
        self.state['status'] = 'completed'
        self._save()

    def error(self, message):
        """Mark training as errored"""
        self.state['status'] = 'error'
        self.state['error_message'] = message
        self._save()


class CharLSTM(nn.Module):
    """Character-level LSTM language model for text generation"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


class CharDataset(Dataset):
    """Custom Dataset for character sequences"""

    def __init__(self, X, y, seq_length, n_vocab):
        """
        Args:
            X: List of character sequences (as integer indices)
            y: List of target characters (as integer indices)
            seq_length: Length of each sequence
            n_vocab: Vocabulary size for normalization
        """
        self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, seq_length, 1) / float(n_vocab)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_training_data(data_path):
    """Load and return training text"""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def prepare_dataset(text, seq_length=100, step=3):
    """Prepare character mappings and training sequences

    Args:
        text: Input text to process
        seq_length: Length of each training sequence
        step: Step size for creating sequences (larger = fewer sequences, faster training)
    """
    # Create character mappings
    chars = sorted(list(set(text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    n_vocab = len(chars)

    print(f"Total characters: {len(text)}")
    print(f"Unique characters: {n_vocab}")
    print(f"Sample characters: {chars[:20]}")

    # Create training sequences with configurable step
    dataX = []
    dataY = []
    print(f"Creating sequences with step={step}...")
    for i in range(0, len(text) - seq_length, step):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)
    print(f"Training sequences: {n_patterns}")

    return dataX, dataY, char_to_int, int_to_char, n_vocab


def train_model(model, X, y, n_vocab, epochs=100, learning_rate=0.001, device='cpu',
                batch_size=128, checkpoint_dir=None, resume_from=None, use_dataloader=False,
                seq_length=100, dashboard_state_file=None):
    """Train the LSTM model using mini-batches with checkpointing and progress bars

    Args:
        model: The LSTM model to train
        X: Input sequences
        y: Target characters
        n_vocab: Vocabulary size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        batch_size: Batch size for training
        checkpoint_dir: Directory to save checkpoints (None = no checkpointing)
        resume_from: Path to checkpoint to resume from (None = start fresh)
        use_dataloader: Use PyTorch DataLoader (better for CPU training)
        seq_length: Sequence length (needed for DataLoader)
        dashboard_state_file: Path to dashboard state file for real-time updates
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize dashboard updater
    dashboard = None
    if dashboard_state_file:
        dashboard = DashboardUpdater(dashboard_state_file)

    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20  # Stop if no improvement for 20 epochs

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")

    print(f"\nTraining on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    # Warn about CPU training time
    if device.type == 'cpu':
        print("\n" + "="*60)
        print("WARNING: Training on CPU - this will be SLOW!")
        print(f"Estimated time: {epochs * 0.3:.1f}-{epochs * 0.6:.1f} minutes")
        print("Consider using GPU for 5-10x speedup if available.")
        print("="*60 + "\n")

    # Optimize CPU threading
    if device.type == 'cpu':
        num_threads = os.cpu_count()
        torch.set_num_threads(num_threads)
        print(f"Using {num_threads} CPU threads")

    # Convert to numpy arrays first
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    n_samples = len(X)
    n_batches = n_samples // batch_size

    print(f"Training samples: {n_samples:,}")
    print(f"Batches per epoch: {n_batches}")

    # Choose between DataLoader and manual batching
    if use_dataloader:
        # Use DataLoader for efficient CPU-GPU data transfer
        print(f"\nPreparing DataLoader...")
        dataset = CharDataset(X, y, seq_length, n_vocab)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == 'cuda'),  # Pin memory for faster GPU transfer
            num_workers=2 if device.type == 'cpu' else 0  # Parallel loading for CPU
        )
        print(f"DataLoader ready with pin_memory={device.type == 'cuda'}")
        X_tensor = None
        y_tensor = None
    else:
        # Pre-load entire dataset to GPU for better performance (recommended for GPU)
        print(f"\nPreparing data tensors...")
        if device.type == 'cuda':
            # Pre-load to GPU and keep there to avoid repeated CPU-GPU transfers
            print("Loading entire dataset to GPU memory...")
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device).reshape(-1, len(X[0]), 1) / float(n_vocab)
            y_tensor = torch.tensor(y, dtype=torch.long, device=device)
            print(f"Dataset loaded to GPU: {X_tensor.shape}")

            torch.cuda.empty_cache()
            print(f"GPU allocated memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            print(f"GPU cached memory: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        else:
            # Keep on CPU for CPU training
            X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, len(X[0]), 1) / float(n_vocab)
            y_tensor = torch.tensor(y, dtype=torch.long)
            print(f"Dataset prepared on CPU: {X_tensor.shape}")
        dataloader = None

    print(f"Starting training from epoch {start_epoch + 1}...\n")

    # Start dashboard tracking
    if dashboard:
        dashboard.start_training(
            total_epochs=epochs,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=model.hidden_size
        )

    # Training loop with progress bar
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        batch_count = 0

        if use_dataloader:
            # Use DataLoader for batch iteration
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100)

            for batch_X, batch_y in pbar:
                # Move to device if not already there
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                # Forward pass
                optimizer.zero_grad()
                output, _ = model(batch_X)
                loss = criterion(output, batch_y)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Update dashboard
                if dashboard and batch_count % 5 == 0:  # Update every 5 batches
                    dashboard.update_batch(epoch + 1, batch_count, len(dataloader), loss.item())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        else:
            # Manual batching with pre-loaded GPU tensors (faster for small datasets on GPU)
            # Shuffle data each epoch - now done on GPU for better performance
            if device.type == 'cuda':
                # GPU-based shuffling - no CPU-GPU transfer needed
                indices = torch.randperm(n_samples, device=device)
                X_shuffled = X_tensor[indices]
                y_shuffled = y_tensor[indices]
            else:
                # CPU-based shuffling
                indices = np.random.permutation(n_samples)
                X_shuffled = X_tensor[indices]
                y_shuffled = y_tensor[indices]

            # Progress bar for batches
            pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}', ncols=100)

            for i in pbar:
                # Get batch - data already on correct device (GPU or CPU)
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                # No conversion needed - data is already on device as tensors!

                # Forward pass
                optimizer.zero_grad()
                output, _ = model(batch_X)
                loss = criterion(output, batch_y)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Update dashboard
                if dashboard and i % 5 == 0:  # Update every 5 batches
                    dashboard.update_batch(epoch + 1, i + 1, n_batches, loss.item())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / batch_count
        epoch_time = time.time() - epoch_start_time

        print(f'Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Time: {epoch_time:.1f}s')

        # Update dashboard with epoch completion
        if dashboard:
            dashboard.update_epoch(epoch + 1, avg_loss, best_loss if avg_loss < best_loss else best_loss, epoch_time)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered! No improvement for {early_stop_patience} epochs.")
            print(f"Best loss: {best_loss:.4f}")
            break

        # Save checkpoint every 10 epochs
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")

    # Mark training as complete in dashboard
    if dashboard:
        dashboard.complete_training()

    return model


def save_model(model, char_to_int, int_to_char, save_dir):
    """Save model and character mappings"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(save_dir, 'bep_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
    }, model_path)

    # Save character mappings
    mappings_path = os.path.join(save_dir, 'char_mappings.json')
    with open(mappings_path, 'w', encoding='utf-8') as f:
        json.dump({
            'char_to_int': char_to_int,
            'int_to_char': {int(k): v for k, v in int_to_char.items()},
            'n_vocab': len(char_to_int)
        }, f, ensure_ascii=False, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Mappings saved to: {mappings_path}")


def generate_sample_text(model, char_to_int, int_to_char, n_vocab,
                         start_text="BIM Execution Plan", length=200, device='cpu'):
    """Generate sample text to verify model"""
    model.eval()
    with torch.no_grad():
        inputs = [char_to_int.get(ch, 0) for ch in start_text]
        inputs = torch.tensor(inputs, dtype=torch.float32).reshape(1, len(inputs), 1) / float(n_vocab)
        inputs = inputs.to(device)

        result = start_text
        hidden = None

        for _ in range(length):
            output, hidden = model(inputs, hidden)
            prob = torch.softmax(output, dim=1).cpu().detach().numpy()
            char_idx = np.random.choice(range(n_vocab), p=prob[0])
            char = int_to_char[char_idx]
            result += char

            new_input = torch.tensor([[char_idx]], dtype=torch.float32).reshape(1, 1, 1) / float(n_vocab)
            new_input = new_input.to(device)
            inputs = new_input

        return result


def main():
    parser = argparse.ArgumentParser(description='Train BEP text generation model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50 for CPU, use 100-150 for GPU)')
    parser.add_argument('--hidden-size', type=int, default=512, help='LSTM hidden size')
    parser.add_argument('--seq-length', type=int, default=100, help='Sequence length')
    parser.add_argument('--step', type=int, default=3,
                       help='Step size for sequence creation (larger = faster training, less data)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to training data file (default: data/training_data.txt or data/consolidated_training_data.txt)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Training batch size (default: 512 for GPU, use 128 for CPU)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    parser.add_argument('--no-checkpoint', action='store_true',
                       help='Disable checkpoint saving during training')
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Determine data file path
    if args.data_file:
        data_path = Path(args.data_file)
        if not data_path.is_absolute():
            data_path = project_root / args.data_file
    else:
        # Try consolidated file first, fallback to training_data.txt
        consolidated_path = project_root / 'data' / 'consolidated_training_data.txt'
        default_path = project_root / 'data' / 'training_data.txt'

        if consolidated_path.exists():
            data_path = consolidated_path
            print(f"Using consolidated training data: {consolidated_path}")
        elif default_path.exists():
            data_path = default_path
            print(f"Using default training data: {default_path}")
        else:
            print("ERROR: No training data found!")
            print(f"Please create one of these files:")
            print(f"  - {consolidated_path}")
            print(f"  - {default_path}")
            print(f"\nOr process documents first:")
            print(f"  python scripts/process_training_documents.py")
            return

    if not data_path.exists():
        print(f"ERROR: Training data file not found: {data_path}")
        return

    models_dir = project_root / 'models'

    print("="*60)
    print("BEP Text Generation Model Training")
    print("="*60)

    # Check for GPU - Force CUDA:0 (NVIDIA GTX 1050 Ti)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    # Setup checkpoint directory
    checkpoint_dir = None if args.no_checkpoint else project_root / 'checkpoints'

    # Load and prepare data
    print("\nLoading training data...")
    text = load_training_data(data_path)

    print("\nPreparing dataset...")
    dataX, dataY, char_to_int, int_to_char, n_vocab = prepare_dataset(
        text, args.seq_length, args.step
    )

    # Create model
    print("\nInitializing model...")
    model = CharLSTM(
        input_size=1,
        hidden_size=args.hidden_size,
        output_size=n_vocab,
        num_layers=2
    ).to(device)

    # Train model
    print("\nStarting training...")

    # Setup dashboard state file
    dashboard_state_file = project_root / 'training_state.json'

    model = train_model(
        model, dataX, dataY, n_vocab,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume,
        use_dataloader=False,  # Use manual GPU pre-loading for better performance
        seq_length=args.seq_length,
        dashboard_state_file=dashboard_state_file
    )

    # Save model
    print("\nSaving model...")
    save_model(model, char_to_int, int_to_char, models_dir)

    # Generate sample text
    print("\n" + "="*60)
    print("Sample Generated Text:")
    print("="*60)
    sample = generate_sample_text(
        model, char_to_int, int_to_char, n_vocab,
        start_text="The BEP establishes",
        length=300,
        device=device
    )
    print(sample)
    print("="*60)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
