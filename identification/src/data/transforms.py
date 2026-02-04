import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1), 
                           interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.25, 0.25, 0.15, 0.02)], p=0.8),
        T.RandomGrayscale(p=0.05),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ])


def get_eval_transforms():
    return T.Compose([
        T.Resize(224, interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = torch.clamp(img, 0, 1)
    return T.ToPILImage()(img)


def generate_augmentation_examples(num_examples=8, output_path=None):
    base_dir = Path(r"d:\identification")
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    if output_path is None:
        output_path = figures_dir / "aug_examples.png"
    
    sample_img_path = base_dir / "main" / "images" / "1" / "1_front.JPG"
    if not sample_img_path.exists():
        print(f"Sample image not found: {sample_img_path}")
        return
    
    original_img = Image.open(sample_img_path).convert('RGB')
    train_transforms = get_train_transforms()
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    for idx in range(num_examples):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        
        augmented_tensor = train_transforms(original_img)
        augmented_img = denormalize_image(augmented_tensor)
        
        axes[row, col].imshow(augmented_img)
        axes[row, col].set_title(f'Aug #{idx+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    return output_path


def save_augmentation_config():
    base_dir = Path(r"d:\identification")
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    config = [
        {'augmentation': 'RandomResizedCrop', 'parameter': 'size', 'value': '224'},
        {'augmentation': 'RandomResizedCrop', 'parameter': 'scale', 'value': '(0.85, 1.0)'},
        {'augmentation': 'RandomResizedCrop', 'parameter': 'ratio', 'value': '(0.9, 1.1)'},
        {'augmentation': 'RandomHorizontalFlip', 'parameter': 'p', 'value': '0.5'},
        {'augmentation': 'ColorJitter', 'parameter': 'brightness', 'value': '0.25'},
        {'augmentation': 'ColorJitter', 'parameter': 'contrast', 'value': '0.25'},
        {'augmentation': 'ColorJitter', 'parameter': 'saturation', 'value': '0.15'},
        {'augmentation': 'ColorJitter', 'parameter': 'hue', 'value': '0.02'},
        {'augmentation': 'ColorJitter', 'parameter': 'apply_prob', 'value': '0.8'},
        {'augmentation': 'RandomGrayscale', 'parameter': 'p', 'value': '0.05'},
        {'augmentation': 'GaussianBlur', 'parameter': 'kernel_size', 'value': '3'},
        {'augmentation': 'GaussianBlur', 'parameter': 'sigma', 'value': '(0.1, 1.0)'},
        {'augmentation': 'GaussianBlur', 'parameter': 'apply_prob', 'value': '0.1'},
        {'augmentation': 'RandomErasing', 'parameter': 'p', 'value': '0.25'},
        {'augmentation': 'RandomErasing', 'parameter': 'scale', 'value': '(0.02, 0.15)'},
        {'augmentation': 'RandomErasing', 'parameter': 'ratio', 'value': '(0.3, 3.3)'},
        {'augmentation': 'Normalize', 'parameter': 'mean', 'value': str(IMAGENET_MEAN)},
        {'augmentation': 'Normalize', 'parameter': 'std', 'value': str(IMAGENET_STD)}
    ]
    
    df = pd.DataFrame(config)
    output_path = results_dir / "augmentation.csv"
    df.to_csv(output_path, index=False)
    print(f"Config saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Creating transform pipelines...")
    train_transforms = get_train_transforms()
    eval_transforms = get_eval_transforms()
    print("Transforms ready")
    
    generate_augmentation_examples(num_examples=8)
    save_augmentation_config()
