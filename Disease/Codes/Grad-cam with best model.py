"""
Grad-CAM++ Visualization for Option-E EfficientNet (Best Model)
================================================================
Generates heatmaps for successful and unsuccessful predictions
- 6 successful cases per class
- 4 unsuccessful cases per class
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2

# Set paths
PROJECT_ROOT = Path(r"E:\Disease Classification\Project")
DATASET_ROOT = Path(r"E:\Disease Classification\Dataset")
EXPLAINABLE_AI_DIR = Path(r"E:\Disease Classification\Explainable Ai")
sys.path.append(str(PROJECT_ROOT))

# Model path
MODEL_PATH = PROJECT_ROOT / "Results/efficientnet/option_e/fold_0/best_model_fold0.pth"
SPLITS_DIR = PROJECT_ROOT / "splits"

# Create output directories
OUTPUT_DIR = EXPLAINABLE_AI_DIR / "Option-E_EfficientNet_GradCAM"
SUCCESSFUL_DIR = OUTPUT_DIR / "successful"
UNSUCCESSFUL_DIR = OUTPUT_DIR / "unsuccessful"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUCCESSFUL_DIR.mkdir(parents=True, exist_ok=True)
UNSUCCESSFUL_DIR.mkdir(parents=True, exist_ok=True)

# Class mappings
CLASS_10_NAMES = ['healthy', 'lsd_s1', 'lsd_s2', 'lsd_s3', 'fmd_s1', 'fmd_s2', 'fmd_s3', 'ibk_s1', 'ibk_s2', 'ibk_s3']


class OptionE_EfficientNet(nn.Module):
    """Option E model with EfficientNet backbone"""
    def __init__(self, dropout=0.25):
        super().__init__()
        efficientnet = models.efficientnet_b1(weights='IMAGENET1K_V1')
        self.features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        in_features = 1280  # EfficientNet-B1
        
        self.disease_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 4)
        )
        
        self.severity_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        disease_out = self.disease_head(x)
        severity_out = self.severity_head(x)
        return disease_out, severity_out


class GradCAMPlusPlus:
    """Grad-CAM++ implementation"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class, task='disease'):
        # Forward pass
        self.model.eval()
        disease_out, severity_out = self.model(input_tensor)
        
        # Select output based on task
        output = disease_out if task == 'disease' else severity_out
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Grad-CAM++ calculation
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculate alpha weights
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2) + (activations * gradients.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha = numerator / (denominator + 1e-8)
        
        # Weighted combination
        weights = (alpha * torch.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        
        # Normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


class HierarchicalDataset(Dataset):
    """Dataset for hierarchical classification"""
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        
        self.disease_to_idx = {'healthy': 0, 'lsd': 1, 'fmd': 2, 'ibk': 3}
        self.severity_to_idx = {1: 0, 2: 1, 3: 2}
        
        self.class_10_to_idx = {
            'healthy': 0, 'lsd_s1': 1, 'lsd_s2': 2, 'lsd_s3': 3,
            'fmd_s1': 4, 'fmd_s2': 5, 'fmd_s3': 6,
            'ibk_s1': 7, 'ibk_s2': 8, 'ibk_s3': 9
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_path = row['filepath']
        if not os.path.isabs(img_path):
            img_path = DATASET_ROOT / img_path
        
        image = Image.open(img_path).convert('RGB')
        original_image = np.array(image)
        
        if self.transform:
            image = self.transform(image)
        
        disease_label = self.disease_to_idx[row['disease']]
        
        if row['disease'] == 'healthy':
            severity_label = -1
            class_10_label = 0
        else:
            severity_label = self.severity_to_idx[row['severity']]
            class_10_name = f"{row['disease']}_s{row['severity']}"
            class_10_label = self.class_10_to_idx[class_10_name]
        
        return {
            'image': image,
            'original': original_image,
            'disease_label': disease_label,
            'severity_label': severity_label,
            'class_10_label': class_10_label,
            'filepath': str(img_path),
            'idx': idx
        }


def get_transform():
    """Get image transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on image"""
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed


def save_gradcam_visualization(original_img, heatmap, actual_class, pred_class, 
                                save_path, is_successful=True):
    """Save Grad-CAM++ visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM++ Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    overlayed = overlay_heatmap(original_img, heatmap)
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add title with prediction info
    status = "✓ CORRECT" if is_successful else "✗ INCORRECT"
    color = 'green' if is_successful else 'red'
    
    title = f"{status}\nActual: {actual_class} | Predicted: {pred_class}"
    fig.suptitle(title, fontsize=14, fontweight='bold', color=color, y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def hierarchical_predict(disease_pred, severity_pred, device):
    """Convert disease and severity predictions to 10-class"""
    if disease_pred == 0:  # healthy
        return 0
    else:
        # Map: lsd=1→[1,2,3], fmd=2→[4,5,6], ibk=3→[7,8,9]
        base_idx = (disease_pred - 1) * 3 + 1
        return base_idx + severity_pred


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading Option-E EfficientNet model...")
    model = OptionE_EfficientNet(dropout=0.25)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from: {MODEL_PATH}\n")
    
    # Initialize Grad-CAM++ (target last convolutional layer)
    target_layer = model.features[-1]  # Last layer of EfficientNet features
    gradcam = GradCAMPlusPlus(model, target_layer)
    
    # Load validation data
    print("Loading validation data...")
    folds = pd.read_csv(SPLITS_DIR / "folds.csv")
    val_data = folds[folds['fold'] == 0].copy()
    
    transform = get_transform()
    val_dataset = HierarchicalDataset(val_data, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"✓ Loaded {len(val_data)} validation samples\n")
    
    # Collect predictions
    print("Running inference to find successful/unsuccessful cases...")
    successful_cases = {i: [] for i in range(10)}
    unsuccessful_cases = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            original = batch['original'][0].numpy()
            class_10_label = batch['class_10_label'].item()
            idx = batch['idx'].item()
            
            disease_out, severity_out = model(images)
            disease_pred = torch.argmax(disease_out, dim=1).item()
            severity_pred = torch.argmax(severity_out, dim=1).item()
            
            class_10_pred = hierarchical_predict(disease_pred, severity_pred, device)
            
            is_correct = (class_10_pred == class_10_label)
            
            case_info = {
                'idx': idx,
                'image_tensor': images,
                'original': original,
                'actual': class_10_label,
                'predicted': class_10_pred,
                'disease_pred': disease_pred,
                'filepath': batch['filepath'][0]
            }
            
            if is_correct and len(successful_cases[class_10_label]) < 6:
                successful_cases[class_10_label].append(case_info)
            elif not is_correct and len(unsuccessful_cases[class_10_label]) < 4:
                unsuccessful_cases[class_10_label].append(case_info)
    
    print("✓ Inference complete\n")
    
    # Generate Grad-CAM++ visualizations
    print("="*70)
    print("GENERATING GRAD-CAM++ VISUALIZATIONS")
    print("="*70)
    
    total_generated = 0
    
    # Process successful cases
    print("\n--- Successful Predictions (6 per class) ---")
    for class_idx in range(10):
        class_name = CLASS_10_NAMES[class_idx]
        cases = successful_cases[class_idx]
        
        print(f"\nClass {class_idx} ({class_name}): {len(cases)} cases")
        
        for i, case in enumerate(cases):
            # Generate Grad-CAM++
            cam = gradcam.generate_cam(case['image_tensor'], case['disease_pred'], task='disease')
            
            # Save visualization
            filename = f"successful_{class_name}_case{i+1}_idx{case['idx']}.png"
            save_path = SUCCESSFUL_DIR / filename
            
            save_gradcam_visualization(
                case['original'], cam,
                actual_class=CLASS_10_NAMES[case['actual']],
                pred_class=CLASS_10_NAMES[case['predicted']],
                save_path=save_path,
                is_successful=True
            )
            
            total_generated += 1
            print(f"  ✓ Saved: {filename}")
    
    # Process unsuccessful cases
    print("\n--- Unsuccessful Predictions (4 per class) ---")
    for class_idx in range(10):
        class_name = CLASS_10_NAMES[class_idx]
        cases = unsuccessful_cases[class_idx]
        
        print(f"\nClass {class_idx} ({class_name}): {len(cases)} cases")
        
        for i, case in enumerate(cases):
            # Generate Grad-CAM++
            cam = gradcam.generate_cam(case['image_tensor'], case['disease_pred'], task='disease')
            
            # Save visualization with actual and predicted in filename
            filename = f"unsuccessful_{class_name}_ACTUAL_{CLASS_10_NAMES[case['actual']]}_PRED_{CLASS_10_NAMES[case['predicted']]}_case{i+1}_idx{case['idx']}.png"
            save_path = UNSUCCESSFUL_DIR / filename
            
            save_gradcam_visualization(
                case['original'], cam,
                actual_class=CLASS_10_NAMES[case['actual']],
                pred_class=CLASS_10_NAMES[case['predicted']],
                save_path=save_path,
                is_successful=False
            )
            
            total_generated += 1
            print(f"  ✓ Saved: {filename}")
    
    print("\n" + "="*70)
    print(f"GRAD-CAM++ GENERATION COMPLETE!")
    print(f"Total visualizations generated: {total_generated}")
    print(f"Successful cases: {SUCCESSFUL_DIR}")
    print(f"Unsuccessful cases: {UNSUCCESSFUL_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
