import torch

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Version: {torch.version.cuda}')
    print(f'Devices: {torch.cuda.device_count()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU available')
