"""
Quick Reference: Using the Dataset Module
Steps 3 & 4 Implementation Guide
"""

# ============================================================================
# BASIC USAGE EXAMPLE
# ============================================================================

from pathlib import Path
import pandas as pd
from dataset import (
    ImageTransforms,
    DiseaseDataset,
    DataLoaderFactory,
    ClassWeightCalculator,
    BalancedSampler
)

# Setup paths
BASE_DIR = Path(r"e:\Disease Classification")
PROJECT_DIR = BASE_DIR / "Project"
DATASET_DIR = BASE_DIR / "Dataset"

# Load splits
folds_df = pd.read_csv(PROJECT_DIR / "splits" / "folds.csv")
test_df = pd.read_csv(PROJECT_DIR / "splits" / "test_split.csv")

# ============================================================================
# EXAMPLE 1: Create DataLoaders for a Single Fold
# ============================================================================

def example_single_fold_dataloaders():
    """Create dataloaders for fold 0 validation"""
    
    # Split data: fold 0 = validation, rest = training
    train_df = folds_df[folds_df['fold'] != 0].reset_index(drop=True)
    val_df = folds_df[folds_df['fold'] == 0].reset_index(drop=True)
    
    # Create dataloaders with balanced sampling
    train_loader, val_loader = DataLoaderFactory.create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        dataset_root=DATASET_DIR,
        batch_size=32,
        num_workers=4,
        use_balanced_sampling=True,
        balance_strategy='disease'  # or 'label10'
    )
    
    return train_loader, val_loader


# ============================================================================
# EXAMPLE 2: Get Class Weights for Loss Functions
# ============================================================================

def example_get_class_weights():
    """Get class weights for weighted loss functions"""
    
    train_df = folds_df[folds_df['fold'] != 0]
    
    # Disease weights (4 classes)
    disease_weights = ClassWeightCalculator.get_disease_weights(train_df)
    # Returns: tensor([w_healthy, w_fmd, w_ibk, w_lsd])
    
    # Severity weights (3 classes, diseased only)
    severity_weights = ClassWeightCalculator.get_severity_weights(train_df)
    # Returns: tensor([w_stage1, w_stage2, w_stage3])
    
    # Label_10 weights (10 classes)
    label10_weights = ClassWeightCalculator.get_label10_weights(train_df)
    # Returns: tensor of length 10
    
    return disease_weights, severity_weights, label10_weights


# ============================================================================
# EXAMPLE 3: Create Test DataLoader
# ============================================================================

def example_test_dataloader():
    """Create test dataloader for final evaluation"""
    
    test_loader = DataLoaderFactory.create_test_loader(
        test_df=test_df,
        dataset_root=DATASET_DIR,
        batch_size=32,
        num_workers=4
    )
    
    return test_loader


# ============================================================================
# EXAMPLE 4: Iterate Through Batches
# ============================================================================

def example_batch_iteration():
    """How to iterate through batches"""
    
    train_loader, _ = example_single_fold_dataloaders()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: torch.Tensor [batch_size, 3, 240, 240]
        # labels: dict with keys:
        #   - 'disease': torch.Tensor [batch_size] (0-3)
        #   - 'severity': torch.Tensor [batch_size] (0-2 or -1)
        #   - 'label_10': torch.Tensor [batch_size] (0-9)
        #   - 'is_diseased': torch.Tensor [batch_size] (0 or 1)
        
        disease_labels = labels['disease']
        severity_labels = labels['severity']
        is_diseased = labels['is_diseased']
        
        # Your training code here
        break  # Just showing first batch


# ============================================================================
# EXAMPLE 5: 5-Fold Cross-Validation Setup
# ============================================================================

def example_5fold_cv():
    """Setup for 5-fold cross-validation"""
    
    results = []
    
    for fold in range(5):
        print(f"\n{'='*70}")
        print(f"Training Fold {fold}")
        print(f"{'='*70}")
        
        # Split data
        train_df = folds_df[folds_df['fold'] != fold].reset_index(drop=True)
        val_df = folds_df[folds_df['fold'] == fold].reset_index(drop=True)
        
        # Create dataloaders
        train_loader, val_loader = DataLoaderFactory.create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            dataset_root=DATASET_DIR,
            batch_size=32,
            num_workers=4,
            use_balanced_sampling=True,
            balance_strategy='disease'
        )
        
        # Get class weights for this fold
        disease_weights = ClassWeightCalculator.get_disease_weights(train_df)
        severity_weights = ClassWeightCalculator.get_severity_weights(train_df)
        
        # Train model (your code here)
        # model = ...
        # train(model, train_loader, val_loader, disease_weights, severity_weights)
        
        # Evaluate and store results
        # fold_results = evaluate(model, val_loader)
        # results.append(fold_results)
    
    # Aggregate results across folds
    # mean_results = aggregate_results(results)
    # return mean_results


# ============================================================================
# EXAMPLE 6: Custom Dataset Creation (Advanced)
# ============================================================================

def example_custom_dataset():
    """Create dataset with custom transforms"""
    
    # Get transforms
    train_transforms = ImageTransforms.get_train_transforms()
    val_transforms = ImageTransforms.get_val_test_transforms()
    
    # Create custom dataset
    train_df = folds_df[folds_df['fold'] != 0]
    
    train_dataset = DiseaseDataset(
        dataframe=train_df,
        dataset_root=DATASET_DIR,
        transform=train_transforms,
        mode='train'
    )
    
    # Access a single sample
    image, labels = train_dataset[0]
    # image: torch.Tensor [3, 240, 240]
    # labels: dict with disease, severity, label_10, is_diseased
    
    return train_dataset


# ============================================================================
# KEY PARAMETERS AND CONFIGURATIONS
# ============================================================================

"""
PREPROCESSING (Step 3):
----------------------
Input size: 240×240

Train augmentation:
  - RandomResizedCrop: scale=(0.85, 1.0)
  - RandomHorizontalFlip: p=0.5
  - RandomRotation: degrees=10
  - ColorJitter: brightness=0.2, contrast=0.2, saturation=0.15
  - Normalization: ImageNet mean/std

Val/Test (no augmentation):
  - Resize: 256
  - CenterCrop: 240
  - Normalization: ImageNet mean/std


CLASS IMBALANCE HANDLING (Step 4):
----------------------------------
1. Balanced Sampling:
   - Strategy: 'disease' or 'label10'
   - WeightedRandomSampler with replacement
   - Minority classes oversampled, majority undersampled

2. On-the-fly Augmentation:
   - Applied during data loading
   - Each oversampled image looks different
   - Memory efficient

3. Weighted Loss:
   - Disease: Inverse frequency weights (4 classes)
   - Severity: Inverse frequency weights (3 classes, diseased only)
   - Label_10: Inverse frequency weights (10 classes)


LABEL STRUCTURE:
---------------
Disease (4 classes):
  0: healthy
  1: fmd
  2: ibk
  3: lsd

Severity (3 classes):
  0: stage 1
  1: stage 2
  2: stage 3
  -1: N/A (for healthy images)

Label_10 (10 classes):
  healthy, fmd_s1, fmd_s2, fmd_s3, ibk_s1, ibk_s2, ibk_s3, lsd_s1, lsd_s2, lsd_s3
"""

# ============================================================================
# TYPICAL USAGE IN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Typical usage pattern in a training script
    """
    
    # 1. Load data
    folds_df = pd.read_csv(PROJECT_DIR / "splits" / "folds.csv")
    
    # 2. Select fold for validation
    fold = 0
    train_df = folds_df[folds_df['fold'] != fold]
    val_df = folds_df[folds_df['fold'] == fold]
    
    # 3. Create dataloaders
    train_loader, val_loader = DataLoaderFactory.create_dataloaders(
        train_df, val_df, DATASET_DIR,
        batch_size=32,
        use_balanced_sampling=True,
        balance_strategy='disease'
    )
    
    # 4. Get class weights for loss functions
    disease_weights = ClassWeightCalculator.get_disease_weights(train_df)
    severity_weights = ClassWeightCalculator.get_severity_weights(train_df)
    
    # 5. Training loop
    for epoch in range(25):
        # Train
        for images, labels in train_loader:
            # Forward pass
            # Calculate weighted loss
            # Backward pass
            pass
        
        # Validate
        for images, labels in val_loader:
            # Forward pass
            # Calculate metrics
            pass
    
    print("Training complete!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Disease weights: {disease_weights}")
    print(f"Severity weights: {severity_weights}")
