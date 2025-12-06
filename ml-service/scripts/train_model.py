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
from pathlib import Path


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


def load_training_data(data_path):
    """Load and return training text"""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def prepare_dataset(text, seq_length=100):
    """Prepare character mappings and training sequences"""
    # Create character mappings
    chars = sorted(list(set(text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    n_vocab = len(chars)

    print(f"Total characters: {len(text)}")
    print(f"Unique characters: {n_vocab}")
    print(f"Sample characters: {chars[:20]}")

    # Create training sequences
    dataX = []
    dataY = []
    for i in range(0, len(text) - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)
    print(f"Training sequences: {n_patterns}")

    return dataX, dataY, char_to_int, int_to_char, n_vocab


def train_model(model, X, y, n_vocab, epochs=100, learning_rate=0.001, device='cpu', batch_size=128):
    """Train the LSTM model using mini-batches"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Batch size: {batch_size}")

    # Convert to numpy arrays first for efficient batching
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    n_samples = len(X)
    n_batches = n_samples // batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            # Convert to tensors
            batch_X = torch.tensor(batch_X, dtype=torch.float32).reshape(batch_size, -1, 1) / float(n_vocab)
            batch_y = torch.tensor(batch_y, dtype=torch.long)

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

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
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden-size', type=int, default=512, help='LSTM hidden size')
    parser.add_argument('--seq-length', type=int, default=100, help='Sequence length')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'data' / 'training_data.txt'
    models_dir = project_root / 'models'

    print("="*60)
    print("BEP Text Generation Model Training")
    print("="*60)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare data
    print("\nLoading training data...")
    text = load_training_data(data_path)

    print("\nPreparing dataset...")
    dataX, dataY, char_to_int, int_to_char, n_vocab = prepare_dataset(text, args.seq_length)

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
    model = train_model(
        model, dataX, dataY, n_vocab,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
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
