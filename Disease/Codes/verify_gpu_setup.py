"""
Final GPU Setup Verification
Confirms everything is ready for GPU-accelerated training
"""

import torch
import sys
import json

print("=" * 80)
print("FINAL GPU SETUP VERIFICATION")
print("=" * 80)

# 1. Python version
print(f"\n✅ Python Version: {sys.version.split()[0]}")
if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
    print("   ⚠️  Recommended: Python 3.11 or 3.12")
else:
    print("   ✅ Compatible with PyTorch CUDA builds")

# 2. PyTorch installation
print(f"\n✅ PyTorch Version: {torch.__version__}")
if "+cu" in torch.__version__:
    print("   ✅ CUDA-enabled build")
else:
    print("   ❌ CPU-only build (no GPU support)")
    sys.exit(1)

# 3. CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n✅ CUDA Available: {cuda_available}")
if cuda_available:
    print(f"   ✅ CUDA Version: {torch.version.cuda}")
    print(f"   ✅ cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("   ❌ CUDA not available")
    sys.exit(1)

# 4. GPU information
gpu_count = torch.cuda.device_count()
print(f"\n✅ GPUs Detected: {gpu_count}")
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"   - Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"   - Compute Capability: {props.major}.{props.minor}")
    print(f"   - Multi Processors: {props.multi_processor_count}")

# 5. Test GPU operation
print("\n✅ Testing GPU Operations:")
try:
    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print("   ✅ Matrix multiplication successful")
    print(f"   ✅ Result tensor on: {z.device}")
except Exception as e:
    print(f"   ❌ GPU operation failed: {e}")
    sys.exit(1)

# 6. Performance settings
print("\n✅ Performance Settings:")
print(f"   - cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
print(f"   - cuDNN Enabled: {torch.backends.cudnn.enabled}")
if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
    print(f"   - TF32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   - TF32 (cuDNN): {torch.backends.cudnn.allow_tf32}")

# 7. Memory info
print("\n✅ GPU Memory Status:")
memory_allocated = torch.cuda.memory_allocated(0) / 1e9
memory_reserved = torch.cuda.memory_reserved(0) / 1e9
memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"   - Allocated: {memory_allocated:.3f} GB")
print(f"   - Reserved: {memory_reserved:.3f} GB")
print(f"   - Total: {memory_total:.2f} GB")
print(f"   - Available: {memory_total - memory_reserved:.2f} GB")

# 8. Recommended settings
print("\n✅ Recommended Training Settings:")
print(f"   - Batch Size: 32-64 (EfficientNet-B1 @ 240x240)")
print(f"   - Num Workers: 4-8 (DataLoader)")
print(f"   - Pin Memory: True")
print(f"   - Mixed Precision: Recommended (torch.cuda.amp)")

# 9. Save configuration
config = {
    'python_version': sys.version.split()[0],
    'pytorch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),
    'gpu_name': torch.cuda.get_device_name(0),
    'gpu_memory_gb': round(memory_total, 2),
    'compute_capability': f"{props.major}.{props.minor}",
    'environment_path': sys.executable,
    'ready_for_training': True
}

with open(r"d:\Disease Final\gpu_config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Final summary
print("\n" + "=" * 80)
print("🚀 GPU SETUP COMPLETE AND VERIFIED!")
print("=" * 80)
print("\n✅ All checks passed!")
print("✅ GPU is ready for model training")
print("✅ Configuration saved to gpu_config.json")
print("\n📌 Use this Python executable for all future training:")
print(f'   & "{sys.executable}" <script.py>')
print("\n🎯 Ready to proceed with training steps!")
print("=" * 80)
