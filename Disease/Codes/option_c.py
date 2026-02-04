"""
Option C - Multi-Task Hierarchical Classification
===================================================
One shared backbone (EfficientNet-B1) with two heads:
- Disease head (4-class): trained on all images
- Severity head (3-class): trained only on diseased images (masked loss)

Total loss: L = L_disease + λ * L_severity
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths
PROJECT_ROOT = Path(r"E:\Disease Classification\Project")
METADATA_PATH = PROJECT_ROOT / "metadata.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "Results" / "Option-C Metrics"
MODELS_DIR = PROJECT_ROOT / "Models" / "Option-C"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class HierarchicalDataset(Dataset):
    """Dataset for multi-task hierarchical learning"""
    
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Disease mapping
        self.disease_to_idx = {'healthy': 0, 'lsd': 1, 'fmd': 2, 'ibk': 3}
        # Severity mapping (1, 2, 3 → 0, 1, 2)
        self.severity_to_idx = {1: 0, 2: 1, 3: 2}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = row['filepath']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Disease label (always available)
        disease_label = self.disease_to_idx[row['disease']]
        
        # Severity label (only for diseased, -1 for healthy as ignore_index)
        if row['disease'] == 'healthy':
            severity_label = -1  # Will be ignored in loss
        else:
            severity_label = self.severity_to_idx[row['severity']]
        
        return image, disease_label, severity_label


def get_transforms(augment=True):
    """Get train/val transforms"""
    
    if augment:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(240, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


class MultiTaskHierarchicalModel(nn.Module):
    """Multi-task model with shared backbone and two heads"""
    
    def __init__(self, dropout=0.25):
        super().__init__()
        
        # Shared backbone: EfficientNet-B1
        efficientnet = models.efficientnet_b1(weights='IMAGENET1K_V1')
        self.backbone = efficientnet.features  # Shared features
        
        # Get feature dimension
        in_features = 1280  # EfficientNet-B1 output channels
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Disease head (4-class)
        self.disease_head = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 4)
        )
        
        # Severity head (3-class)
        self.severity_head = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 3)
        )
    
    def forward(self, x):
        # Shared backbone
        features = self.backbone(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        
        # Two heads
        disease_output = self.disease_head(features)
        severity_output = self.severity_head(features)
        
        return disease_output, severity_output
    
    def freeze_backbone(self):
        """Freeze backbone for warmup phase"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def compute_class_weights(labels):
    """Compute class weights for imbalanced data"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, disease_criterion, severity_criterion, 
                    optimizer, device, lambda_severity):
    """Train for one epoch with multi-task loss"""
    model.train()
    running_disease_loss = 0.0
    running_severity_loss = 0.0
    running_total_loss = 0.0
    
    all_disease_preds = []
    all_disease_labels = []
    all_severity_preds = []
    all_severity_labels = []
    
    for images, disease_labels, severity_labels in loader:
        images = images.to(device)
        disease_labels = disease_labels.to(device)
        severity_labels = severity_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        disease_output, severity_output = model(images)
        
        # Disease loss (all images)
        disease_loss = disease_criterion(disease_output, disease_labels)
        
        # Severity loss (only diseased images, masked)
        # severity_labels = -1 for healthy images (ignored)
        severity_loss = severity_criterion(severity_output, severity_labels)
        
        # Combined loss
        total_loss = disease_loss + lambda_severity * severity_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Track losses
        running_disease_loss += disease_loss.item() * images.size(0)
        running_severity_loss += severity_loss.item() * images.size(0)
        running_total_loss += total_loss.item() * images.size(0)
        
        # Predictions
        disease_preds = torch.argmax(disease_output, dim=1)
        severity_preds = torch.argmax(severity_output, dim=1)
        
        all_disease_preds.extend(disease_preds.cpu().numpy())
        all_disease_labels.extend(disease_labels.cpu().numpy())
        
        # Only evaluate severity on diseased images
        diseased_mask = severity_labels != -1
        if diseased_mask.sum() > 0:
            all_severity_preds.extend(severity_preds[diseased_mask].cpu().numpy())
            all_severity_labels.extend(severity_labels[diseased_mask].cpu().numpy())
    
    # Compute metrics
    epoch_disease_loss = running_disease_loss / len(loader.dataset)
    epoch_severity_loss = running_severity_loss / len(loader.dataset)
    epoch_total_loss = running_total_loss / len(loader.dataset)
    
    disease_acc = accuracy_score(all_disease_labels, all_disease_preds)
    _, _, disease_f1, _ = precision_recall_fscore_support(
        all_disease_labels, all_disease_preds, average='macro', zero_division=0
    )
    
    if len(all_severity_labels) > 0:
        severity_acc = accuracy_score(all_severity_labels, all_severity_preds)
        _, _, severity_f1, _ = precision_recall_fscore_support(
            all_severity_labels, all_severity_preds, average='macro', zero_division=0
        )
    else:
        severity_acc = severity_f1 = 0.0
    
    return {
        'total_loss': epoch_total_loss,
        'disease_loss': epoch_disease_loss,
        'severity_loss': epoch_severity_loss,
        'disease_acc': disease_acc,
        'disease_f1': disease_f1,
        'severity_acc': severity_acc,
        'severity_f1': severity_f1
    }


def validate(model, loader, disease_criterion, severity_criterion, 
            device, lambda_severity):
    """Validate model"""
    model.eval()
    running_disease_loss = 0.0
    running_severity_loss = 0.0
    running_total_loss = 0.0
    
    all_disease_preds = []
    all_disease_labels = []
    all_severity_preds = []
    all_severity_labels = []
    
    with torch.no_grad():
        for images, disease_labels, severity_labels in loader:
            images = images.to(device)
            disease_labels = disease_labels.to(device)
            severity_labels = severity_labels.to(device)
            
            # Forward pass
            disease_output, severity_output = model(images)
            
            # Disease loss
            disease_loss = disease_criterion(disease_output, disease_labels)
            
            # Severity loss (masked)
            severity_loss = severity_criterion(severity_output, severity_labels)
            
            # Combined loss
            total_loss = disease_loss + lambda_severity * severity_loss
            
            running_disease_loss += disease_loss.item() * images.size(0)
            running_severity_loss += severity_loss.item() * images.size(0)
            running_total_loss += total_loss.item() * images.size(0)
            
            # Predictions
            disease_preds = torch.argmax(disease_output, dim=1)
            severity_preds = torch.argmax(severity_output, dim=1)
            
            all_disease_preds.extend(disease_preds.cpu().numpy())
            all_disease_labels.extend(disease_labels.cpu().numpy())
            
            # Only evaluate severity on diseased images
            diseased_mask = severity_labels != -1
            if diseased_mask.sum() > 0:
                all_severity_preds.extend(severity_preds[diseased_mask].cpu().numpy())
                all_severity_labels.extend(severity_labels[diseased_mask].cpu().numpy())
    
    # Compute metrics
    epoch_disease_loss = running_disease_loss / len(loader.dataset)
    epoch_severity_loss = running_severity_loss / len(loader.dataset)
    epoch_total_loss = running_total_loss / len(loader.dataset)
    
    disease_acc = accuracy_score(all_disease_labels, all_disease_preds)
    _, _, disease_f1, _ = precision_recall_fscore_support(
        all_disease_labels, all_disease_preds, average='macro', zero_division=0
    )
    
    if len(all_severity_labels) > 0:
        severity_acc = accuracy_score(all_severity_labels, all_severity_preds)
        _, _, severity_f1, _ = precision_recall_fscore_support(
            all_severity_labels, all_severity_preds, average='macro', zero_division=0
        )
    else:
        severity_acc = severity_f1 = 0.0
    
    return {
        'total_loss': epoch_total_loss,
        'disease_loss': epoch_disease_loss,
        'severity_loss': epoch_severity_loss,
        'disease_acc': disease_acc,
        'disease_f1': disease_f1,
        'severity_acc': severity_acc,
        'severity_f1': severity_f1,
        'disease_preds': all_disease_preds,
        'disease_labels': all_disease_labels,
        'severity_preds': all_severity_preds,
        'severity_labels': all_severity_labels
    }


def train_model(model, train_loader, val_loader, config, device):
    """
    Two-phase training: warmup (frozen) + fine-tuning (unfrozen)
    """
    
    # Loss functions
    disease_criterion = nn.CrossEntropyLoss(weight=config['disease_weights'].to(device))
    severity_criterion = nn.CrossEntropyLoss(
        weight=config['severity_weights'].to(device),
        ignore_index=-1  # Ignore healthy images
    )
    
    # Training history
    history = {
        'train_total_loss': [], 'train_disease_loss': [], 'train_severity_loss': [],
        'train_disease_acc': [], 'train_disease_f1': [],
        'train_severity_acc': [], 'train_severity_f1': [],
        'val_total_loss': [], 'val_disease_loss': [], 'val_severity_loss': [],
        'val_disease_acc': [], 'val_disease_f1': [],
        'val_severity_acc': [], 'val_severity_f1': [],
        'lr': []
    }
    
    best_val_f1_combined = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Training Multi-Task Hierarchical Model (Option C)")
    print(f"{'='*60}")
    
    # PHASE 1: Warmup (frozen backbone)
    print(f"\nPhase 1: Warmup (frozen backbone) - {config['warmup_epochs']} epochs")
    model.freeze_backbone()
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['head_lr'],
        weight_decay=config['weight_decay']
    )
    
    for epoch in range(config['warmup_epochs']):
        train_metrics = train_one_epoch(
            model, train_loader, disease_criterion, severity_criterion,
            optimizer, device, config['lambda_severity']
        )
        val_metrics = validate(
            model, val_loader, disease_criterion, severity_criterion,
            device, config['lambda_severity']
        )
        
        # Store history
        for key in ['total_loss', 'disease_loss', 'severity_loss', 
                    'disease_acc', 'disease_f1', 'severity_acc', 'severity_f1']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}/{config['warmup_epochs']} | "
              f"Train: D_F1={train_metrics['disease_f1']:.4f}, S_F1={train_metrics['severity_f1']:.4f} | "
              f"Val: D_F1={val_metrics['disease_f1']:.4f}, S_F1={val_metrics['severity_f1']:.4f}")
    
    # PHASE 2: Fine-tuning (unfrozen backbone)
    print(f"\nPhase 2: Fine-tuning (unfrozen backbone) - up to {config['max_epochs']} epochs")
    model.unfreeze_backbone()
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['backbone_lr']},
        {'params': model.disease_head.parameters(), 'lr': config['head_lr']},
        {'params': model.severity_head.parameters(), 'lr': config['head_lr']}
    ], weight_decay=config['weight_decay'])
    
    # Scheduler monitors combined F1 score
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True, min_lr=1e-6
    )
    
    for epoch in range(config['warmup_epochs'], config['warmup_epochs'] + config['max_epochs']):
        train_metrics = train_one_epoch(
            model, train_loader, disease_criterion, severity_criterion,
            optimizer, device, config['lambda_severity']
        )
        val_metrics = validate(
            model, val_loader, disease_criterion, severity_criterion,
            device, config['lambda_severity']
        )
        
        # Store history
        for key in ['total_loss', 'disease_loss', 'severity_loss',
                    'disease_acc', 'disease_f1', 'severity_acc', 'severity_f1']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Combined F1 for early stopping
        val_f1_combined = val_metrics['disease_f1'] + val_metrics['severity_f1']
        
        scheduler.step(val_f1_combined)
        
        print(f"Epoch {epoch+1}/{config['warmup_epochs']+config['max_epochs']} | "
              f"Train: D_F1={train_metrics['disease_f1']:.4f}, S_F1={train_metrics['severity_f1']:.4f} | "
              f"Val: D_F1={val_metrics['disease_f1']:.4f}, S_F1={val_metrics['severity_f1']:.4f}")
        
        # Early stopping
        if val_f1_combined > best_val_f1_combined:
            best_val_f1_combined = val_f1_combined
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    
    print(f"\nBest combined F1: {best_val_f1_combined:.4f} at epoch {best_epoch}")
    
    return model, history, best_val_f1_combined, best_epoch


def train_option_c(fold_idx, config, save_dir):
    """Train Option C for one fold"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load metadata and splits
    metadata = pd.read_csv(METADATA_PATH)
    folds_df = pd.read_csv(SPLITS_DIR / "folds.csv")
    
    # Get train/val split
    train_indices = folds_df[folds_df[f'fold_{fold_idx}'] == 'train'].index
    val_indices = folds_df[folds_df[f'fold_{fold_idx}'] == 'val'].index
    
    train_df = metadata.iloc[train_indices]
    val_df = metadata.iloc[val_indices]
    
    print(f"\nFold {fold_idx}:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    
    # Datasets
    train_dataset = HierarchicalDataset(train_df, get_transforms(augment=True))
    val_dataset = HierarchicalDataset(val_df, get_transforms(augment=False))
    
    # Compute class weights
    disease_labels = train_df['disease'].map({'healthy': 0, 'lsd': 1, 'fmd': 2, 'ibk': 3}).values
    disease_weights = compute_class_weights(disease_labels)
    
    severity_labels = train_df[train_df['disease'] != 'healthy']['severity'].map({1: 0, 2: 1, 3: 2}).values
    severity_weights = compute_class_weights(severity_labels)
    
    print(f"\nDisease weights: {disease_weights.numpy()}")
    print(f"Severity weights: {severity_weights.numpy()}")
    
    # Balanced sampling
    sample_weights = disease_weights[disease_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = MultiTaskHierarchicalModel(dropout=config['dropout']).to(device)
    
    # Update config with weights
    config['disease_weights'] = disease_weights
    config['severity_weights'] = severity_weights
    
    # Train
    model, history, best_val_f1, best_epoch = train_model(
        model, train_loader, val_loader, config, device
    )
    
    # Save model
    torch.save(model.state_dict(), save_dir / f"model_fold{fold_idx}.pth")
    
    # Final validation
    final_val_metrics = validate(
        model, val_loader, 
        nn.CrossEntropyLoss(weight=disease_weights.to(device)),
        nn.CrossEntropyLoss(weight=severity_weights.to(device), ignore_index=-1),
        device, config['lambda_severity']
    )
    
    # Compute hierarchical accuracy
    hierarchical_correct = 0
    total = 0
    
    for idx, row in val_df.iterrows():
        disease_pred = final_val_metrics['disease_preds'][total]
        true_disease = row['disease']
        disease_map = {0: 'healthy', 1: 'lsd', 2: 'fmd', 3: 'ibk'}
        
        if true_disease == 'healthy':
            if disease_pred == 0:  # Correctly predicted healthy
                hierarchical_correct += 1
        else:
            # Check if disease is correct
            true_disease_idx = {'healthy': 0, 'lsd': 1, 'fmd': 2, 'ibk': 3}[true_disease]
            if disease_pred == true_disease_idx:
                # Check severity
                severity_idx = [i for i, (d_pred, d_label) in enumerate(
                    zip(final_val_metrics['disease_preds'], final_val_metrics['disease_labels'])
                ) if d_label != 0]
                
                if total < len(severity_idx):
                    sev_pred_idx = [i for i, x in enumerate(severity_idx) if x == total]
                    if sev_pred_idx:
                        severity_pred = final_val_metrics['severity_preds'][sev_pred_idx[0]]
                        true_severity = row['severity']
                        if severity_pred == (true_severity - 1):  # Map 1,2,3 to 0,1,2
                            hierarchical_correct += 1
        
        total += 1
    
    hierarchical_acc = hierarchical_correct / total if total > 0 else 0.0
    
    # Save results
    results = {
        'disease': {
            'accuracy': float(final_val_metrics['disease_acc']),
            'f1_macro': float(final_val_metrics['disease_f1'])
        },
        'severity': {
            'accuracy': float(final_val_metrics['severity_acc']),
            'f1_macro': float(final_val_metrics['severity_f1'])
        },
        'hierarchical_accuracy': float(hierarchical_acc),
        'training': {
            'history': history,
            'best_combined_f1': float(best_val_f1),
            'best_epoch': int(best_epoch)
        }
    }
    
    with open(save_dir / f"results_fold{fold_idx}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Default configuration
    config = {
        'batch_size': 32,
        'warmup_epochs': 5,
        'max_epochs': 25,
        'backbone_lr': 5e-5,
        'head_lr': 1e-3,
        'lambda_severity': 1.0,
        'weight_decay': 1e-4,
        'dropout': 0.25,
        'patience': 5
    }
    
    # Train on fold 0
    fold_idx = 0
    save_dir = RESULTS_DIR / f"fold_{fold_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"OPTION C - MULTI-TASK HIERARCHICAL")
    print(f"Training on Fold {fold_idx}")
    print(f"{'='*80}")
    
    start_time = time.time()
    results = train_option_c(fold_idx, config, save_dir)
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Training completed in {elapsed_time/60:.2f} minutes")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}")
