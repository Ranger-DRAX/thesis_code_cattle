"""
Dataset and DataLoader utilities for disease classification
Steps 3 & 4: Preprocessing and Class imbalance handling
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 3: PREPROCESSING - Image Transforms
# ============================================================================

class ImageTransforms:
    """
    Preprocessing transforms for training, validation, and testing
    Input size: 240×240 (EfficientNet-B1 standard)
    """
    
    # ImageNet normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    INPUT_SIZE = 240
    
    @staticmethod
    def get_train_transforms():
        """
        Step 3.1: Train preprocessing with augmentation
        - RandomResizedCrop to 240×240 (scale 0.85–1.0)
        - Random horizontal flip
        - Small rotation (±10°)
        - Mild brightness/contrast (and slight saturation)
        - Convert to tensor
        - Normalize with ImageNet mean/std
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(
                ImageTransforms.INPUT_SIZE,
                scale=(0.85, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,      # Mild brightness
                contrast=0.2,        # Mild contrast
                saturation=0.15,     # Slight saturation
                hue=0.0
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ImageTransforms.IMAGENET_MEAN,
                std=ImageTransforms.IMAGENET_STD
            )
        ])
    
    @staticmethod
    def get_val_test_transforms():
        """
        Step 3.2: Validation/Test preprocessing (no augmentation)
        - Resize + CenterCrop to 240×240
        - Convert to tensor
        - Normalize with ImageNet mean/std
        """
        return transforms.Compose([
            transforms.Resize(256),  # Resize to slightly larger
            transforms.CenterCrop(ImageTransforms.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ImageTransforms.IMAGENET_MEAN,
                std=ImageTransforms.IMAGENET_STD
            )
        ])


# ============================================================================
# Dataset Classes
# ============================================================================

class DiseaseDataset(Dataset):
    """
    Dataset for disease classification with hierarchical labels
    Supports both disease and severity prediction
    """
    
    def __init__(self, dataframe, dataset_root, transform=None, mode='train'):
        """
        Args:
            dataframe: DataFrame with columns [filepath, disease, severity, label_10]
            dataset_root: Root directory containing the Dataset folder
            transform: torchvision transforms to apply
            mode: 'train', 'val', or 'test'
        """
        self.df = dataframe.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.mode = mode
        
        # Label encodings
        self.disease_to_idx = {'healthy': 0, 'fmd': 1, 'ibk': 2, 'lsd': 3}
        self.idx_to_disease = {v: k for k, v in self.disease_to_idx.items()}
        
        # Severity: 1->0, 2->1, 3->2 (for diseased images only)
        self.severity_to_idx = {1: 0, 2: 1, 3: 2}
        self.idx_to_severity = {v: k for k, v in self.severity_to_idx.items()}
        
        # Label_10 encoding
        self.label10_classes = sorted(self.df['label_10'].unique())
        self.label10_to_idx = {label: idx for idx, label in enumerate(self.label10_classes)}
        self.idx_to_label10 = {v: k for k, v in self.label10_to_idx.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor [3, 240, 240]
            labels: Dict containing:
                - disease: Disease class index (0-3)
                - severity: Severity index (0-2) or -1 for healthy
                - label_10: Combined label index (0-9)
                - is_diseased: 1 if diseased, 0 if healthy
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.dataset_root / row['filepath']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (240, 240), color=(128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Prepare labels
        disease = self.disease_to_idx[row['disease']]
        
        # Severity handling
        if pd.isna(row['severity']) or row['disease'] == 'healthy':
            severity = -1  # -1 indicates N/A (for healthy images)
            is_diseased = 0
        else:
            severity = self.severity_to_idx[int(row['severity'])]
            is_diseased = 1
        
        # Label_10
        label_10 = self.label10_to_idx[row['label_10']]
        
        labels = {
            'disease': torch.tensor(disease, dtype=torch.long),
            'severity': torch.tensor(severity, dtype=torch.long),
            'label_10': torch.tensor(label_10, dtype=torch.long),
            'is_diseased': torch.tensor(is_diseased, dtype=torch.float32)
        }
        
        return image, labels


# ============================================================================
# STEP 4: CLASS IMBALANCE HANDLING
# ============================================================================

class ClassWeightCalculator:
    """
    Calculate class weights for handling imbalanced datasets
    Step 4.3: Weighted loss
    """
    
    @staticmethod
    def compute_class_weights(labels, num_classes):
        """
        Compute class weights using inverse frequency
        
        Args:
            labels: Array of class labels
            num_classes: Total number of classes
            
        Returns:
            weights: Tensor of class weights
        """
        # Count samples per class
        class_counts = np.bincount(labels, minlength=num_classes)
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        
        # Compute weights: inverse frequency
        total_samples = len(labels)
        weights = total_samples / (num_classes * class_counts)
        
        return torch.FloatTensor(weights)
    
    @staticmethod
    def get_disease_weights(dataframe):
        """
        Get class weights for disease classification (4 classes)
        """
        disease_to_idx = {'healthy': 0, 'fmd': 1, 'ibk': 2, 'lsd': 3}
        labels = dataframe['disease'].map(disease_to_idx).values
        return ClassWeightCalculator.compute_class_weights(labels, num_classes=4)
    
    @staticmethod
    def get_severity_weights(dataframe):
        """
        Get class weights for severity classification (3 classes)
        Only computed on diseased samples
        """
        # Filter diseased samples only
        diseased_df = dataframe[dataframe['disease'] != 'healthy'].copy()
        
        if len(diseased_df) == 0:
            return torch.ones(3)
        
        # Map severity to 0-indexed labels
        severity_to_idx = {1: 0, 2: 1, 3: 2}
        labels = diseased_df['severity'].map(severity_to_idx).values
        return ClassWeightCalculator.compute_class_weights(labels, num_classes=3)
    
    @staticmethod
    def get_label10_weights(dataframe):
        """
        Get class weights for 10-class flat classification
        """
        label10_classes = sorted(dataframe['label_10'].unique())
        label10_to_idx = {label: idx for idx, label in enumerate(label10_classes)}
        labels = dataframe['label_10'].map(label10_to_idx).values
        return ClassWeightCalculator.compute_class_weights(labels, num_classes=10)


class BalancedSampler:
    """
    Step 4.1: Oversampling / balanced sampling
    Creates a sampler that balances classes during training
    """
    
    @staticmethod
    def create_disease_balanced_sampler(dataframe):
        """
        Create a WeightedRandomSampler for disease-balanced batches
        Balances: healthy vs lsd vs fmd vs ibk
        """
        disease_to_idx = {'healthy': 0, 'fmd': 1, 'ibk': 2, 'lsd': 3}
        labels = dataframe['disease'].map(disease_to_idx).values
        
        # Compute class weights
        class_counts = np.bincount(labels, minlength=4)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        
        # Assign weight to each sample based on its class
        sample_weights = class_weights[labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler
    
    @staticmethod
    def create_label10_balanced_sampler(dataframe):
        """
        Create a WeightedRandomSampler for label_10 balanced batches
        Balances all 10 classes
        """
        label10_classes = sorted(dataframe['label_10'].unique())
        label10_to_idx = {label: idx for idx, label in enumerate(label10_classes)}
        labels = dataframe['label_10'].map(label10_to_idx).values
        
        # Compute class weights
        class_counts = np.bincount(labels, minlength=len(label10_classes))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        
        # Assign weight to each sample
        sample_weights = class_weights[labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler


# ============================================================================
# DataLoader Factory
# ============================================================================

class DataLoaderFactory:
    """
    Factory for creating dataloaders with proper preprocessing and balancing
    """
    
    @staticmethod
    def create_dataloaders(
        train_df,
        val_df,
        dataset_root,
        batch_size=32,
        num_workers=4,
        use_balanced_sampling=True,
        balance_strategy='disease'  # 'disease' or 'label10'
    ):
        """
        Create train and validation dataloaders
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            dataset_root: Root directory containing Dataset folder
            batch_size: Batch size
            num_workers: Number of worker processes
            use_balanced_sampling: Whether to use balanced sampling for training
            balance_strategy: 'disease' or 'label10'
            
        Returns:
            train_loader, val_loader
        """
        # Get transforms
        train_transforms = ImageTransforms.get_train_transforms()
        val_transforms = ImageTransforms.get_val_test_transforms()
        
        # Create datasets
        train_dataset = DiseaseDataset(
            train_df,
            dataset_root,
            transform=train_transforms,
            mode='train'
        )
        
        val_dataset = DiseaseDataset(
            val_df,
            dataset_root,
            transform=val_transforms,
            mode='val'
        )
        
        # Create sampler for training
        if use_balanced_sampling:
            if balance_strategy == 'disease':
                train_sampler = BalancedSampler.create_disease_balanced_sampler(train_df)
            elif balance_strategy == 'label10':
                train_sampler = BalancedSampler.create_label10_balanced_sampler(train_df)
            else:
                raise ValueError(f"Unknown balance_strategy: {balance_strategy}")
            
            shuffle = False  # Don't shuffle when using sampler
        else:
            train_sampler = None
            shuffle = True
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch for consistent batch size
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    @staticmethod
    def create_test_loader(test_df, dataset_root, batch_size=32, num_workers=4):
        """
        Create test dataloader
        """
        test_transforms = ImageTransforms.get_val_test_transforms()
        
        test_dataset = DiseaseDataset(
            test_df,
            dataset_root,
            transform=test_transforms,
            mode='test'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return test_loader


# ============================================================================
# Utility Functions
# ============================================================================

def print_dataset_statistics(dataloader, name="Dataset"):
    """Print statistics about a dataset"""
    print(f"\n{name} Statistics:")
    print(f"  Total batches: {len(dataloader)}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Total samples: ~{len(dataloader) * dataloader.batch_size}")
    

def test_dataloader():
    """Test the dataloader functionality"""
    print("=" * 70)
    print("Testing DataLoader Implementation")
    print("=" * 70)
    
    # Load data
    BASE_DIR = Path(r"e:\Disease Classification")
    PROJECT_DIR = BASE_DIR / "Project"
    DATASET_DIR = BASE_DIR / "Dataset"
    
    # Load folds
    folds_df = pd.read_csv(PROJECT_DIR / "splits" / "folds.csv")
    
    # Use fold 0 as validation, rest as training
    train_df = folds_df[folds_df['fold'] != 0].reset_index(drop=True)
    val_df = folds_df[folds_df['fold'] == 0].reset_index(drop=True)
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Calculate class weights
    print("\n" + "=" * 70)
    print("Class Weights (Step 4.3)")
    print("=" * 70)
    
    disease_weights = ClassWeightCalculator.get_disease_weights(train_df)
    print(f"\nDisease weights (4 classes):")
    disease_names = ['healthy', 'fmd', 'ibk', 'lsd']
    for i, name in enumerate(disease_names):
        print(f"  {name}: {disease_weights[i]:.4f}")
    
    severity_weights = ClassWeightCalculator.get_severity_weights(train_df)
    print(f"\nSeverity weights (3 classes, diseased only):")
    for i in range(3):
        print(f"  Stage {i+1}: {severity_weights[i]:.4f}")
    
    label10_weights = ClassWeightCalculator.get_label10_weights(train_df)
    print(f"\nLabel_10 weights (10 classes):")
    label10_classes = sorted(train_df['label_10'].unique())
    for i, label in enumerate(label10_classes):
        print(f"  {label}: {label10_weights[i]:.4f}")
    
    # Create dataloaders
    print("\n" + "=" * 70)
    print("Creating DataLoaders")
    print("=" * 70)
    
    train_loader, val_loader = DataLoaderFactory.create_dataloaders(
        train_df,
        val_df,
        DATASET_DIR,
        batch_size=32,
        num_workers=0,  # Use 0 for testing
        use_balanced_sampling=True,
        balance_strategy='disease'
    )
    
    print_dataset_statistics(train_loader, "Training")
    print_dataset_statistics(val_loader, "Validation")
    
    # Test loading a batch
    print("\n" + "=" * 70)
    print("Testing Batch Loading")
    print("=" * 70)
    
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Disease labels: {labels['disease'].shape}")
    print(f"  Severity labels: {labels['severity'].shape}")
    print(f"  Label_10: {labels['label_10'].shape}")
    print(f"  Is diseased: {labels['is_diseased'].shape}")
    
    print(f"\nSample disease distribution in batch:")
    disease_counts = torch.bincount(labels['disease'], minlength=4)
    for i, name in enumerate(disease_names):
        print(f"  {name}: {disease_counts[i].item()}")
    
    print(f"\nImage statistics:")
    print(f"  Min value: {images.min():.3f}")
    print(f"  Max value: {images.max():.3f}")
    print(f"  Mean: {images.mean():.3f}")
    print(f"  Std: {images.std():.3f}")
    
    print("\n" + "=" * 70)
    print("✓ DataLoader Test Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_dataloader()
