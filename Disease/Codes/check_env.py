"""
Helper script to ensure GPU environment is used
Run this before any training scripts
"""

import sys
import os

# GPU environment path
GPU_ENV = r"d:\Disease Final\.venv-gpu\Scripts\python.exe"

print("=" * 80)
print("ENVIRONMENT CHECK")
print("=" * 80)

current_python = sys.executable
print(f"Current Python: {current_python}")

if ".venv-gpu" in current_python:
    print("✅ Using GPU environment (Python 3.12 + CUDA)")
    
    # Verify GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"✅ PyTorch Version: {torch.__version__}")
        else:
            print("⚠️  GPU not detected in this environment")
    except ImportError:
        print("⚠️  PyTorch not installed in this environment")
else:
    print("⚠️  NOT using GPU environment!")
    print(f"\n🔄 Please use the GPU environment for training:")
    print(f'   & "{GPU_ENV}" <your_script.py>')
    print("\n   Or run: configure_python_environment with GPU path")

print("=" * 80)
