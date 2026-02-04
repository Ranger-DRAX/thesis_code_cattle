"""
Step 4: Class Imbalance Handling
Implements balanced sampling and weighted loss computation for training.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance better than weighted CE.
    
    Focal Loss = -alpha * (1-pt)^gamma * log(pt)
    where pt is the probability of the true class.
    
    Focuses learning on hard examples by down-weighting easy examples.
    
    Args:
        alpha: Class weights tensor (default: None)
        gamma: Focusing parameter, higher gamma = more focus on hard examples (default: 2.0)
        label_smoothing: Label smoothing factor (default: 0.1)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply label smoothing if specified
        num_classes = inputs.shape[1]
        if self.label_smoothing > 0:
            # Create smoothed one-hot targets
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            
            # Compute cross entropy manually with smoothed labels
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(targets_one_hot * log_probs).sum(dim=1)
            
            # Get probabilities for focal weighting
            probs = torch.softmax(inputs, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # Standard cross entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            probs = torch.softmax(inputs, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        
        # Apply alpha (class) weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_disease_weights(labels):
    """
    Compute class weights for disease classification (4 classes)
    
    Args:
        labels: Array or list of disease labels
        
    Returns:
        torch.Tensor: Class weights for weighted cross entropy
    """
    unique_classes = np.array(['fmd', 'healthy', 'ibk', 'lsd'])
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    return torch.FloatTensor(weights)


def compute_severity_weights(labels):
    """
    Compute class weights for severity classification (3 classes)
    Only computed on diseased samples.
    
    Args:
        labels: Array or list of severity labels (1, 2, 3)
        
    Returns:
        torch.Tensor: Class weights for weighted cross entropy
    """
    # Filter out NA/NaN values (healthy samples)
    labels = [l for l in labels if str(l) not in ['NA', 'nan', '']]
    labels = [int(float(l)) for l in labels]
    
    unique_classes = np.array([1, 2, 3])
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    return torch.FloatTensor(weights)


def compute_label10_weights(labels):
    """
    Compute class weights for flat 10-class classification (Option A)
    
    Args:
        labels: Array or list of label_10 values
        
    Returns:
        torch.Tensor: Class weights for weighted cross entropy
    """
    unique_classes = np.array([
        'fmd_s1', 'fmd_s2', 'fmd_s3',
        'healthy',
        'ibk_s1', 'ibk_s2', 'ibk_s3',
        'lsd_s1', 'lsd_s2', 'lsd_s3'
    ])
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    return torch.FloatTensor(weights)


def create_balanced_sampler(df, balance_type='disease'):
    """
    Step 4.1: Create a balanced sampler for handling class imbalance
    
    Args:
        df: DataFrame with columns including 'disease', 'severity', 'label_10'
        balance_type: Type of balancing
            - 'disease': Balance by disease (4 classes)
            - 'label_10': Balance by label_10 (10 classes)
            - 'hybrid': Balance disease first, then severity within diseased
            
    Returns:
        WeightedRandomSampler: Sampler for DataLoader
    """
    if balance_type == 'disease':
        # Balance by disease category
        labels = df['disease'].values
        class_counts = df['disease'].value_counts().to_dict()
        
    elif balance_type == 'label_10':
        # Balance by label_10 (flat 10-class)
        labels = df['label_10'].values
        class_counts = df['label_10'].value_counts().to_dict()
        
    elif balance_type == 'hybrid':
        # Hybrid: disease-balanced, with stage-balancing within diseased
        # Create custom weights
        weights = []
        disease_counts = df['disease'].value_counts().to_dict()
        
        for idx, row in df.iterrows():
            disease = row['disease']
            disease_weight = 1.0 / disease_counts[disease]
            
            if disease == 'healthy':
                weights.append(disease_weight)
            else:
                # For diseased, further balance by severity
                label_10 = row['label_10']
                severity_counts = df[df['disease'] == disease]['severity'].value_counts().to_dict()
                severity = row['severity']
                severity_weight = 1.0 / severity_counts[severity] if severity in severity_counts else 1.0
                weights.append(disease_weight * severity_weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=True
        )
        return sampler
    else:
        raise ValueError(f"Unknown balance_type: {balance_type}")
    
    # For disease and label_10 balancing
    weights = []
    for label in labels:
        weights.append(1.0 / class_counts[label])
    
    weights = np.array(weights)
    weights = weights / weights.sum() * len(weights)  # Normalize
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def get_class_weights_from_metadata(metadata_path):
    """
    Compute all class weights from metadata.csv
    
    Args:
        metadata_path: Path to metadata.csv
        
    Returns:
        dict: Dictionary with disease_weights, severity_weights, label10_weights
    """
    df = pd.read_csv(metadata_path)
    
    # Disease weights (4 classes)
    disease_weights = compute_disease_weights(df['disease'].values)
    
    # Severity weights (3 classes) - only on diseased samples
    diseased_df = df[df['disease'] != 'healthy']
    severity_weights = compute_severity_weights(diseased_df['severity'].values)
    
    # Label_10 weights (10 classes)
    label10_weights = compute_label10_weights(df['label_10'].values)
    
    return {
        'disease_weights': disease_weights,
        'severity_weights': severity_weights,
        'label10_weights': label10_weights
    }


if __name__ == "__main__":
    # Test the class imbalance handling
    print("Step 4: Class Imbalance Handling")
    print("=" * 60)
    
    metadata_path = r"d:\Disease Final\metadata.csv"
    df = pd.read_csv(metadata_path)
    
    print("Computing class weights...")
    weights = get_class_weights_from_metadata(metadata_path)
    
    print("\n✅ Disease class weights (4 classes):")
    diseases = ['fmd', 'healthy', 'ibk', 'lsd']
    for i, disease in enumerate(diseases):
        print(f"   {disease}: {weights['disease_weights'][i]:.4f}")
    
    print("\n✅ Severity class weights (3 classes, diseased-only):")
    severities = [1, 2, 3]
    for i, severity in enumerate(severities):
        print(f"   Stage {severity}: {weights['severity_weights'][i]:.4f}")
    
    print("\n✅ Label_10 class weights (10 classes):")
    labels_10 = ['fmd_s1', 'fmd_s2', 'fmd_s3', 'healthy', 
                 'ibk_s1', 'ibk_s2', 'ibk_s3', 
                 'lsd_s1', 'lsd_s2', 'lsd_s3']
    for i, label in enumerate(labels_10):
        print(f"   {label}: {weights['label10_weights'][i]:.4f}")
    
    print("\n" + "=" * 60)
    print("Creating balanced samplers...")
    
    # Get training data from fold 0
    folds_df = pd.read_csv(r"d:\Disease Final\Splits\folds.csv")
    train_df = folds_df[(folds_df['fold'] == 0) & (folds_df['split'] == 'train')]
    
    print(f"\n✅ Training set (Fold 0): {len(train_df)} samples")
    print(f"\nOriginal disease distribution:")
    print(train_df['disease'].value_counts())
    
    # Create different types of samplers
    sampler_disease = create_balanced_sampler(train_df, balance_type='disease')
    sampler_label10 = create_balanced_sampler(train_df, balance_type='label_10')
    sampler_hybrid = create_balanced_sampler(train_df, balance_type='hybrid')
    
    print("\n✅ Created 3 types of balanced samplers:")
    print("   1. Disease-balanced sampler (4 diseases)")
    print("   2. Label_10-balanced sampler (10 classes)")
    print("   3. Hybrid sampler (disease + severity balancing)")
    
    print("\n✅ Step 4 class imbalance handling ready!")
