"""
Find Maximum Batch Size for GPU Training

This script tests different batch sizes to find the maximum that fits in GPU memory.
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_model import CharLSTM

def test_batch_size(batch_size, hidden_size=512, seq_length=100, n_vocab=91, device='cuda'):
    """Test if a batch size fits in GPU memory"""
    try:
        # Create model
        model = CharLSTM(
            input_size=1,
            hidden_size=hidden_size,
            output_size=n_vocab,
            num_layers=2
        ).to(device)

        # Create dummy batch
        dummy_input = torch.randn(batch_size, seq_length, 1, device=device)
        dummy_target = torch.randint(0, n_vocab, (batch_size,), device=device)

        # Forward pass
        output, _ = model(dummy_input)

        # Backward pass
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, dummy_target)
        loss.backward()

        # Check memory usage
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9

        # Clear memory
        del model, dummy_input, dummy_target, output, loss
        torch.cuda.empty_cache()

        return True, allocated, reserved

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return False, 0, 0
        raise

def find_max_batch_size(hidden_size=512, seq_length=100):
    """Binary search to find maximum batch size"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return None

    device = torch.device('cuda:0')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"\nTesting with hidden_size={hidden_size}, seq_length={seq_length}")
    print("=" * 60)

    # Binary search
    min_batch = 64
    max_batch = 8192
    best_batch = min_batch

    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2

        print(f"\nTesting batch_size={mid_batch}...", end=" ")
        success, allocated, reserved = test_batch_size(
            mid_batch, hidden_size, seq_length, device=device
        )

        if success:
            print(f"OK (Mem: {allocated:.2f} GB / {reserved:.2f} GB)")
            best_batch = mid_batch
            min_batch = mid_batch + 1
        else:
            print("OOM")
            max_batch = mid_batch - 1

    return best_batch

def main():
    """Test different configurations"""
    print("=" * 60)
    print("GPU Batch Size Optimizer")
    print("=" * 60)

    configurations = [
        (512, 100, "Default configuration"),
        (768, 100, "Larger model (768 hidden)"),
        (1024, 100, "Very large model (1024 hidden)"),
        (512, 150, "Longer sequences (150 chars)"),
        (768, 150, "Large model + long sequences"),
    ]

    results = []

    for hidden_size, seq_length, description in configurations:
        print(f"\n{'=' * 60}")
        print(f"{description}")
        print(f"{'=' * 60}")

        max_batch = find_max_batch_size(hidden_size, seq_length)

        if max_batch:
            results.append((description, hidden_size, seq_length, max_batch))
            print(f"\n✓ Maximum batch_size: {max_batch}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Recommended Batch Sizes")
    print("=" * 60)

    for desc, hidden, seq_len, batch in results:
        safe_batch = int(batch * 0.9)  # 90% of max for safety
        print(f"\n{desc}:")
        print(f"  --hidden-size {hidden} --seq-length {seq_len} --batch-size {safe_batch}")
        print(f"  (Max tested: {batch}, recommended: {safe_batch} for safety)")

    print("\n" + "=" * 60)
    print("Use these values with train_model.py:")
    print("=" * 60)

    if results:
        desc, hidden, seq_len, batch = results[0]
        safe_batch = int(batch * 0.9)
        print(f"\npython ml-service/scripts/train_model.py \\")
        print(f"    --epochs 100 \\")
        print(f"    --batch-size {safe_batch} \\")
        print(f"    --hidden-size {hidden} \\")
        print(f"    --seq-length {seq_len}")

if __name__ == '__main__':
    main()
