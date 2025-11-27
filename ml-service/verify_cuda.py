"""
CUDA Verification Script
Checks if PyTorch can access the NVIDIA GPU
"""
import torch

print("=" * 60)
print("PyTorch CUDA Verification")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"\nGPU Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU Properties:")
    print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Multi-processors: {props.multi_processor_count}")

    # Test tensor creation on GPU
    print(f"\n" + "=" * 60)
    print("Testing GPU tensor operations...")
    print("=" * 60)

    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("✓ GPU tensor operations working correctly!")
        print(f"  Test tensor shape: {z.shape}")
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
        print(f"  GPU memory cached: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")
    except Exception as e:
        print(f"✗ Error during GPU operations: {e}")
else:
    print("\n⚠ WARNING: CUDA is not available!")
    print("This could mean:")
    print("  1. PyTorch CPU-only version is installed")
    print("  2. NVIDIA drivers are not installed")
    print("  3. GPU is not detected by the system")
    print("\nTo install CUDA-enabled PyTorch, run:")
    print("  .\\ml-service\\install_cuda_pytorch.ps1")

print("\n" + "=" * 60)
