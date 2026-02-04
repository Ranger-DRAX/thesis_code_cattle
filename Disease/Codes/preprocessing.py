"""
Step 3: Preprocessing Pipeline
Implements data augmentation and preprocessing for training and validation/test.
Input size: 240×240 (EfficientNet-B1 standard)
"""

import torch
from torchvision import transforms
from PIL import Image

# ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INPUT_SIZE = 240


class ConvertRGB:
    """Convert image to RGB"""
    def __call__(self, img):
        return img.convert('RGB')

def get_train_transforms():
    """
    Step 3.1: Train preprocessing with augmentation
    
    Returns:
        transforms.Compose: Training transform pipeline
    """
    return transforms.Compose([
        ConvertRGB(),  # Convert to RGB
        transforms.RandomResizedCrop(
            INPUT_SIZE, 
            scale=(0.7, 1.0),  # More aggressive scale range
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomRotation(
            degrees=25,  # Increased from ±10° to ±25°
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Random translation up to 10%
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ColorJitter(
            brightness=0.3,  # Increased brightness adjustment
            contrast=0.3,    # Increased contrast adjustment
            saturation=0.3,  # Increased saturation adjustment
            hue=0.1          # Added hue adjustment
        ),
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # Random erasing for regularization
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def get_val_test_transforms():
    """
    Step 3.2: Validation/Test preprocessing (no augmentation)
    
    Returns:
        transforms.Compose: Validation/test transform pipeline
    """
    return transforms.Compose([
        ConvertRGB(),  # Convert to RGB
        transforms.Resize(
            int(INPUT_SIZE * 256 / 224),  # Resize to slightly larger
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(INPUT_SIZE),  # Center crop to target size
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def load_and_preprocess_image(image_path, transform):
    """
    Load an image and apply preprocessing transform
    
    Args:
        image_path: Path to image file
        transform: Transform to apply
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image = Image.open(image_path)
    return transform(image)


if __name__ == "__main__":
    # Test the transforms
    print("Step 3: Preprocessing Pipeline")
    print("=" * 60)
    
    train_transform = get_train_transforms()
    val_transform = get_val_test_transforms()
    
    print("✅ Train transforms created:")
    print(f"   - Input size: {INPUT_SIZE}×{INPUT_SIZE}")
    print(f"   - RandomResizedCrop: scale=(0.85, 1.0)")
    print(f"   - RandomHorizontalFlip: p=0.5")
    print(f"   - RandomRotation: ±10°")
    print(f"   - ColorJitter: brightness=0.2, contrast=0.2, saturation=0.1")
    print(f"   - Normalize: ImageNet mean/std")
    
    print("\n✅ Val/Test transforms created:")
    print(f"   - Input size: {INPUT_SIZE}×{INPUT_SIZE}")
    print(f"   - Resize + CenterCrop")
    print(f"   - Normalize: ImageNet mean/std")
    
    print("\n✅ Step 3 preprocessing pipeline ready!")
