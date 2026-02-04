"""
GPU Verification and Configuration Test
Tests CUDA availability and configures GPU for training
"""

import sys

print("=" * 80)
print("GPU CONFIGURATION TEST")
print("=" * 80)

# Test 1: Import PyTorch
print("\n[1/5] Testing PyTorch import...")
try:
    import torch
    print("✅ PyTorch imported successfully")
    print(f"   Version: {torch.__version__}")
except Exception as e:
    print(f"❌ Failed to import PyTorch: {e}")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n[2/5] Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("✅ CUDA is available")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("⚠️  CUDA is not available - will use CPU")
    print("   This might be a DLL issue or driver compatibility")

# Test 3: Get GPU information
print("\n[3/5] GPU Information...")
if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"✅ Number of GPUs available: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n   GPU {i}:")
        print(f"   - Name: {torch.cuda.get_device_name(i)}")
        print(f"   - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"   - Compute Capability: {torch.cuda.get_device_capability(i)}")
else:
    print("⚠️  No GPU detected")

# Test 4: Test tensor operations on GPU
print("\n[4/5] Testing GPU tensor operations...")
if cuda_available:
    try:
        # Create tensor on GPU
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("✅ GPU tensor operations successful")
        print(f"   Device: {z.device}")
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
        cuda_available = False
else:
    print("⚠️  Skipping GPU test (CUDA not available)")

# Test 5: Configure optimal settings
print("\n[5/5] Configuring optimal GPU settings...")
if cuda_available:
    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True
    print("✅ cuDNN benchmark enabled")
    
    # Enable TF32 for faster training on Ampere+ GPUs (RTX 30/40 series)
    if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled for faster training (Ampere+ GPU)")
    
    # Set device
    device = torch.device("cuda:0")
    print(f"✅ Default device set to: {device}")
else:
    device = torch.device("cpu")
    print(f"⚠️  Default device set to: {device}")

# Final Summary
print("\n" + "=" * 80)
print("CONFIGURATION SUMMARY")
print("=" * 80)

if cuda_available:
    print("✅ GPU ENABLED AND READY FOR TRAINING!")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   Recommended batch size: 32-64 (adjust based on model)")
    print(f"   Number of workers: 4-8 (for DataLoader)")
    print("\n🚀 Ready to proceed with GPU-accelerated training!")
else:
    print("❌ GPU NOT AVAILABLE - Training will use CPU")
    print("   This will be significantly slower")
    print("   Possible issues:")
    print("   - NVIDIA drivers not installed or outdated")
    print("   - CUDA toolkit compatibility issues")
    print("   - DLL loading problems")
    print("\n   Run 'nvidia-smi' to verify GPU and drivers")

print("=" * 80)

# Save device configuration for future use
print("\n📝 Saving GPU configuration...")
config = {
    'device': 'cuda' if cuda_available else 'cpu',
    'gpu_name': torch.cuda.get_device_name(0) if cuda_available else 'N/A',
    'cuda_available': cuda_available,
    'num_gpus': torch.cuda.device_count() if cuda_available else 0,
}

import json
with open(r"d:\Disease Final\gpu_config.json", 'w') as f:
    json.dump(config, f, indent=2)
print("✅ Configuration saved to gpu_config.json")
