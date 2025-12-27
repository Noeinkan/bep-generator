"""
Unit tests for train_model.py

Run tests with: pytest ml-service/tests/test_train_model.py -v
"""

import pytest
import torch
import numpy as np
import os
import tempfile
from pathlib import Path
import sys

# Add parent directory to path to import train_model
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from train_model import (
    CharDataset,
    CharRNN,
    collate_fn_with_padding,
    load_training_data,
    prepare_dataset,
    generate_sample_text
)


class TestCharDataset:
    """Test CharDataset class"""

    def test_dataset_creation(self):
        """Test that dataset can be created with valid data"""
        X = [[1, 2, 3], [4, 5, 6]]
        y = [7, 8]
        dataset = CharDataset(X, y)

        assert len(dataset) == 2
        assert dataset[0][0].shape == torch.Size([3])
        assert dataset[0][1] == 7

    def test_dataset_types(self):
        """Test that dataset returns correct tensor types"""
        X = [[1, 2, 3]]
        y = [4]
        dataset = CharDataset(X, y)

        X_tensor, y_tensor = dataset[0]
        assert X_tensor.dtype == torch.long
        assert y_tensor.dtype == torch.long


class TestCollateFn:
    """Test custom collate function"""

    def test_padding_basic(self):
        """Test basic padding functionality"""
        # Create sequences of different lengths
        seq1 = torch.tensor([1, 2, 3], dtype=torch.long)
        seq2 = torch.tensor([4, 5], dtype=torch.long)
        target1 = torch.tensor(6, dtype=torch.long)
        target2 = torch.tensor(7, dtype=torch.long)

        batch = [(seq1, target1), (seq2, target2)]
        padded_seqs, targets = collate_fn_with_padding(batch, pad_idx=0)

        # Check shapes
        assert padded_seqs.shape == torch.Size([2, 3])  # Batch size 2, max length 3
        assert targets.shape == torch.Size([2])

        # Check padding was applied to second sequence
        assert padded_seqs[1, 2].item() == 0  # Should be padded with 0

    def test_padding_values(self):
        """Test that padding uses correct value"""
        seq1 = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        seq2 = torch.tensor([5, 6], dtype=torch.long)
        target1 = torch.tensor(7, dtype=torch.long)
        target2 = torch.tensor(8, dtype=torch.long)

        batch = [(seq1, target1), (seq2, target2)]
        padded_seqs, _ = collate_fn_with_padding(batch, pad_idx=99)

        # Second sequence should be padded with 99
        assert padded_seqs[1, 2].item() == 99
        assert padded_seqs[1, 3].item() == 99


class TestCharRNN:
    """Test CharRNN model"""

    def test_model_creation_lstm(self):
        """Test LSTM model creation"""
        model = CharRNN(
            vocab_size=50,
            embed_dim=32,
            hidden_size=64,
            output_size=50,
            num_layers=2,
            rnn_type='lstm'
        )

        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.rnn_type == 'lstm'

    def test_model_creation_gru(self):
        """Test GRU model creation"""
        model = CharRNN(
            vocab_size=50,
            embed_dim=32,
            hidden_size=64,
            output_size=50,
            num_layers=1,
            rnn_type='gru'
        )

        assert model.rnn_type == 'gru'

    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = CharRNN(
            vocab_size=50,
            embed_dim=32,
            hidden_size=64,
            output_size=50,
            num_layers=2,
            rnn_type='lstm'
        )

        # Create dummy input (batch_size=4, seq_length=10)
        x = torch.randint(0, 50, (4, 10), dtype=torch.long)

        output, hidden = model(x)

        # Check output shape
        assert output.shape == torch.Size([4, 50])  # Batch size 4, vocab size 50

        # Check hidden state structure for LSTM
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2

    def test_model_init_hidden(self):
        """Test hidden state initialization"""
        model = CharRNN(
            vocab_size=50,
            embed_dim=32,
            hidden_size=64,
            output_size=50,
            num_layers=2,
            rnn_type='lstm'
        )

        batch_size = 8
        hidden = model.init_hidden(batch_size, torch.device('cpu'))

        # For LSTM, hidden is a tuple of (h, c)
        assert isinstance(hidden, tuple)
        assert hidden[0].shape == torch.Size([2, 8, 64])  # num_layers, batch_size, hidden_size


class TestLoadTrainingData:
    """Test load_training_data function"""

    def test_load_valid_file(self):
        """Test loading a valid text file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("This is test data for the BEP model.\n" * 10)
            temp_path = f.name

        try:
            text = load_training_data(temp_path)
            assert len(text) > 0
            assert isinstance(text, str)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_training_data("nonexistent_file_12345.txt")

    def test_preprocessing(self):
        """Test that preprocessing works correctly"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("HELLO    World\n\n\n\nTest")
            temp_path = f.name

        try:
            text = load_training_data(temp_path)
            # Should be lowercase
            assert text.islower()
            # Should not have excessive newlines
            assert '\n\n\n' not in text
        finally:
            os.unlink(temp_path)

    def test_empty_file(self):
        """Test that empty file raises ValueError"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Training data file is empty"):
                load_training_data(temp_path)
        finally:
            os.unlink(temp_path)


class TestPrepareDataset:
    """Test prepare_dataset function"""

    def test_prepare_basic(self):
        """Test basic dataset preparation"""
        text = "hello world this is a test"
        train_X, train_Y, val_X, val_Y, char_to_int, int_to_char, n_vocab = prepare_dataset(
            text, seq_length=5, validation_split=0.2
        )

        # Check that we got data
        assert len(train_X) > 0
        assert len(val_X) > 0
        assert len(train_X) > len(val_X)  # Training set should be larger

        # Check character mappings
        assert '<PAD>' in char_to_int
        assert char_to_int['<PAD>'] == 0  # PAD should be at index 0
        assert '<EOS>' in char_to_int
        assert n_vocab > 0

    def test_validation_split(self):
        """Test that validation split works correctly"""
        text = "a" * 1000
        train_X, train_Y, val_X, val_Y, _, _, _ = prepare_dataset(
            text, seq_length=10, validation_split=0.2
        )

        total = len(train_X) + len(val_X)
        val_ratio = len(val_X) / total

        # Should be approximately 0.2
        assert 0.15 < val_ratio < 0.25

    def test_sequence_length(self):
        """Test that sequences have correct length"""
        text = "abcdefghijklmnop"
        train_X, _, _, _, _, _, _ = prepare_dataset(text, seq_length=5, validation_split=0.1)

        # Each sequence should have length 5
        assert len(train_X[0]) == 5


class TestGenerateSampleText:
    """Test text generation function"""

    def test_generation_basic(self):
        """Test basic text generation"""
        # Create a small model
        model = CharRNN(
            vocab_size=30,
            embed_dim=16,
            hidden_size=32,
            output_size=30,
            num_layers=1,
            rnn_type='lstm'
        )
        model.eval()

        # Create dummy mappings
        char_to_int = {chr(i): i for i in range(30)}
        int_to_char = {i: chr(i) for i in range(30)}

        # Generate text
        text = generate_sample_text(
            model, char_to_int, int_to_char, n_vocab=30,
            start_text="test", length=20, device='cpu', use_greedy=True
        )

        assert isinstance(text, str)
        assert len(text) >= len("test")  # Should be at least as long as start text

    def test_generation_temperature(self):
        """Test temperature parameter"""
        model = CharRNN(
            vocab_size=30,
            embed_dim=16,
            hidden_size=32,
            output_size=30,
            num_layers=1,
            rnn_type='lstm'
        )
        model.eval()

        char_to_int = {chr(i): i for i in range(30)}
        int_to_char = {i: chr(i) for i in range(30)}

        # Should work with different temperatures
        text_low = generate_sample_text(
            model, char_to_int, int_to_char, n_vocab=30,
            start_text="a", length=10, temperature=0.5
        )
        text_high = generate_sample_text(
            model, char_to_int, int_to_char, n_vocab=30,
            start_text="a", length=10, temperature=2.0
        )

        assert isinstance(text_low, str)
        assert isinstance(text_high, str)


class TestModelSaveLoad:
    """Test model saving and loading (integration test)"""

    def test_model_state_dict(self):
        """Test that model state can be saved and loaded"""
        model1 = CharRNN(
            vocab_size=50,
            embed_dim=32,
            hidden_size=64,
            output_size=50,
            num_layers=2,
            rnn_type='lstm'
        )

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model with same architecture
        model2 = CharRNN(
            vocab_size=50,
            embed_dim=32,
            hidden_size=64,
            output_size=50,
            num_layers=2,
            rnn_type='lstm'
        )

        # Load state dict
        model2.load_state_dict(state_dict)

        # Models should produce same output
        x = torch.randint(0, 50, (2, 10), dtype=torch.long)
        with torch.no_grad():
            output1, _ = model1(x)
            output2, _ = model2(x)

        assert torch.allclose(output1, output2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
