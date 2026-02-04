"""
Option B - Two-Stage Cascade Classification
============================================
Stage 1: Disease classification (4-class: healthy, lsd, fmd, ibk)
Stage 2: Severity classification (3-class: stage1, stage2, stage3) - trained only on diseased images

Inference: Disease → if diseased → Severity
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

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths
PROJECT_ROOT = Path(r"E:\Disease Classification\Project")
METADATA_PATH = PROJECT_ROOT / "metadata.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "Results" / "Option-B Metrics"
MODELS_DIR = PROJECT_ROOT / "Models" / "Option-B"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class DiseaseDataset(Dataset):
    """Dataset for disease classification (4-class)"""
    
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Map disease to index
        self.disease_to_idx = {'healthy': 0, 'lsd': 1, 'fmd': 2, 'ibk': 3}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = row['filepath']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get disease label
        disease_label = self.disease_to_idx[row['disease']]
        
        return image, disease_label


class SeverityDataset(Dataset):
    """Dataset for severity classification (3-class) - diseased images only"""
    
    def __init__(self, dataframe, transform=None):
        # Filter only diseased images
        self.data = dataframe[dataframe['disease'] != 'healthy'].reset_index(drop=True)
        self.transform = transform
        
        # Severity to index (1, 2, 3 → 0, 1, 2)
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
        
        # Get severity label
        severity_label = self.severity_to_idx[row['severity']]
        
        return image, severity_label


def get_transforms(augment=True):
    """Get train/val transforms"""
    
    if augment:
        # Training transforms with augmentation
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
        # Validation/test transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B1 classifier"""
    
    def __init__(self, num_classes, dropout=0.25):
        super().__init__()
        
        # Load pretrained EfficientNet-B1
        self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1')
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except classifier"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def compute_class_weights(labels):
    """Compute class weights for imbalanced data"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    _, _, epoch_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    _, _, epoch_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, config, device, model_name="model"):
    """
    Two-phase training: warmup (frozen) + fine-tuning (unfrozen)
    """
    
    # Class weights for loss
    criterion = nn.CrossEntropyLoss(weight=config['class_weights'].to(device))
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
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
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}/{config['warmup_epochs']} | "
              f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
    
    # PHASE 2: Fine-tuning (unfrozen backbone)
    print(f"\nPhase 2: Fine-tuning (unfrozen backbone) - up to {config['max_epochs']} epochs")
    model.unfreeze_backbone()
    
    optimizer = optim.AdamW([
        {'params': model.backbone.features.parameters(), 'lr': config['backbone_lr']},
        {'params': model.backbone.classifier.parameters(), 'lr': config['head_lr']}
    ], weight_decay=config['weight_decay'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True, min_lr=1e-6
    )
    
    for epoch in range(config['warmup_epochs'], config['warmup_epochs'] + config['max_epochs']):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch+1}/{config['warmup_epochs']+config['max_epochs']} | "
              f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            best_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    
    print(f"\nBest validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    return model, history, best_val_f1, best_epoch


def train_option_b(fold_idx, config, save_dir):
    """
    Train Option B for one fold
    
    Args:
        fold_idx: Fold number (0-4)
        config: Hyperparameter configuration
        save_dir: Directory to save results
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load metadata and splits
    metadata = pd.read_csv(METADATA_PATH)
    folds_df = pd.read_csv(SPLITS_DIR / "folds.csv")
    
    # Get train/val split for this fold
    train_indices = folds_df[folds_df[f'fold_{fold_idx}'] == 'train'].index
    val_indices = folds_df[folds_df[f'fold_{fold_idx}'] == 'val'].index
    
    train_df = metadata.iloc[train_indices]
    val_df = metadata.iloc[val_indices]
    
    print(f"\nFold {fold_idx}:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    
    # ========================================
    # STAGE 1: DISEASE MODEL
    # ========================================
    print(f"\n{'='*80}")
    print("STAGE 1: DISEASE CLASSIFICATION (4-class)")
    print(f"{'='*80}")
    
    # Disease datasets
    train_disease_dataset = DiseaseDataset(train_df, get_transforms(augment=True))
    val_disease_dataset = DiseaseDataset(val_df, get_transforms(augment=False))
    
    # Compute class weights
    disease_labels = train_df['disease'].map({'healthy': 0, 'lsd': 1, 'fmd': 2, 'ibk': 3}).values
    disease_weights = compute_class_weights(disease_labels)
    print(f"\nDisease class weights: {disease_weights.numpy()}")
    
    # Create samplers for balanced batches
    sample_weights = disease_weights[disease_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders
    train_disease_loader = DataLoader(
        train_disease_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=0
    )
    val_disease_loader = DataLoader(
        val_disease_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create disease model
    disease_model = EfficientNetClassifier(num_classes=4, dropout=config['dropout']).to(device)
    
    # Train disease model
    disease_config = {
        'class_weights': disease_weights,
        'warmup_epochs': config['warmup_epochs'],
        'max_epochs': config['max_epochs'],
        'head_lr': config['disease_head_lr'],
        'backbone_lr': config['disease_backbone_lr'],
        'weight_decay': config['weight_decay'],
        'patience': config['patience']
    }
    
    disease_model, disease_history, disease_best_f1, disease_best_epoch = train_model(
        disease_model, train_disease_loader, val_disease_loader,
        disease_config, device, model_name="Disease Model"
    )
    
    # Save disease model
    torch.save(disease_model.state_dict(), save_dir / f"disease_model_fold{fold_idx}.pth")
    
    # ========================================
    # STAGE 2: SEVERITY MODEL
    # ========================================
    print(f"\n{'='*80}")
    print("STAGE 2: SEVERITY CLASSIFICATION (3-class) - Diseased images only")
    print(f"{'='*80}")
    
    # Severity datasets (diseased only)
    train_severity_dataset = SeverityDataset(train_df, get_transforms(augment=True))
    val_severity_dataset = SeverityDataset(val_df, get_transforms(augment=False))
    
    print(f"\nSeverity training samples (diseased only): {len(train_severity_dataset)}")
    print(f"Severity validation samples (diseased only): {len(val_severity_dataset)}")
    
    # Compute class weights for severity
    severity_labels = train_df[train_df['disease'] != 'healthy']['severity'].map({1: 0, 2: 1, 3: 2}).values
    severity_weights = compute_class_weights(severity_labels)
    print(f"\nSeverity class weights: {severity_weights.numpy()}")
    
    # Create samplers for balanced batches
    sample_weights_sev = severity_weights[severity_labels]
    sampler_sev = WeightedRandomSampler(
        weights=sample_weights_sev,
        num_samples=len(sample_weights_sev),
        replacement=True
    )
    
    # Data loaders
    train_severity_loader = DataLoader(
        train_severity_dataset,
        batch_size=config['batch_size'],
        sampler=sampler_sev,
        num_workers=0
    )
    val_severity_loader = DataLoader(
        val_severity_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create severity model
    severity_model = EfficientNetClassifier(num_classes=3, dropout=config['dropout']).to(device)
    
    # Train severity model
    severity_config = {
        'class_weights': severity_weights,
        'warmup_epochs': config['warmup_epochs'],
        'max_epochs': config['max_epochs'],
        'head_lr': config['severity_head_lr'],
        'backbone_lr': config['severity_backbone_lr'],
        'weight_decay': config['weight_decay'],
        'patience': config['patience']
    }
    
    severity_model, severity_history, severity_best_f1, severity_best_epoch = train_model(
        severity_model, train_severity_loader, val_severity_loader,
        severity_config, device, model_name="Severity Model"
    )
    
    # Save severity model
    torch.save(severity_model.state_dict(), save_dir / f"severity_model_fold{fold_idx}.pth")
    
    # ========================================
    # EVALUATE CASCADE
    # ========================================
    print(f"\n{'='*80}")
    print("EVALUATING CASCADE INFERENCE")
    print(f"{'='*80}")
    
    results = evaluate_cascade(
        disease_model, severity_model, val_df, device, fold_idx, save_dir
    )
    
    # Save training history
    training_metrics = {
        'disease_model': {
            'history': disease_history,
            'best_val_f1': float(disease_best_f1),
            'best_epoch': int(disease_best_epoch)
        },
        'severity_model': {
            'history': severity_history,
            'best_val_f1': float(severity_best_f1),
            'best_epoch': int(severity_best_epoch)
        }
    }
    
    with open(save_dir / f"training_metrics_fold{fold_idx}.json", 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Combine results
    results['training'] = training_metrics
    
    return results


def evaluate_cascade(disease_model, severity_model, val_df, device, fold_idx, save_dir):
    """
    Evaluate cascade: Disease → Severity
    """
    
    disease_model.eval()
    severity_model.eval()
    
    # Prepare data
    val_transform = get_transforms(augment=False)
    
    all_disease_preds = []
    all_disease_labels = []
    all_severity_preds = []
    all_severity_labels = []
    all_hierarchical_correct = []
    
    disease_map = {0: 'healthy', 1: 'lsd', 2: 'fmd', 3: 'ibk'}
    severity_map = {0: 1, 1: 2, 2: 3}
    
    with torch.no_grad():
        for idx, row in val_df.iterrows():
            # Load image
            img_path = row['filepath']
            image = Image.open(img_path).convert('RGB')
            image_tensor = val_transform(image).unsqueeze(0).to(device)
            
            # Stage 1: Predict disease
            disease_output = disease_model(image_tensor)
            disease_pred_idx = torch.argmax(disease_output, dim=1).item()
            disease_pred = disease_map[disease_pred_idx]
            
            # True labels
            true_disease = row['disease']
            true_severity = row['severity']
            
            all_disease_preds.append(disease_pred)
            all_disease_labels.append(true_disease)
            
            # Stage 2: Predict severity if diseased
            if disease_pred != 'healthy':
                severity_output = severity_model(image_tensor)
                severity_pred_idx = torch.argmax(severity_output, dim=1).item()
                severity_pred = severity_map[severity_pred_idx]
            else:
                severity_pred = None
            
            # For diseased images, check severity
            if true_disease != 'healthy':
                all_severity_labels.append(true_severity)
                if disease_pred != 'healthy':
                    all_severity_preds.append(severity_pred)
                else:
                    all_severity_preds.append(None)  # Wrong disease prediction
            
            # Hierarchical correctness
            if true_disease == 'healthy':
                hierarchical_correct = (disease_pred == 'healthy')
            else:
                hierarchical_correct = (disease_pred == true_disease) and (severity_pred == true_severity)
            
            all_hierarchical_correct.append(hierarchical_correct)
    
    # Compute metrics
    disease_to_idx = {'healthy': 0, 'lsd': 1, 'fmd': 2, 'ibk': 3}
    disease_preds_idx = [disease_to_idx[d] for d in all_disease_preds]
    disease_labels_idx = [disease_to_idx[d] for d in all_disease_labels]
    
    disease_acc = accuracy_score(disease_labels_idx, disease_preds_idx)
    disease_prec, disease_rec, disease_f1, _ = precision_recall_fscore_support(
        disease_labels_idx, disease_preds_idx, average='macro', zero_division=0
    )
    
    # Severity metrics (only for diseased images)
    # Filter out None predictions
    valid_severity_indices = [i for i, pred in enumerate(all_severity_preds) if pred is not None]
    valid_severity_preds = [all_severity_preds[i] for i in valid_severity_indices]
    valid_severity_labels = [all_severity_labels[i] for i in valid_severity_indices]
    
    if len(valid_severity_preds) > 0:
        severity_acc = accuracy_score(valid_severity_labels, valid_severity_preds)
        severity_prec, severity_rec, severity_f1, _ = precision_recall_fscore_support(
            valid_severity_labels, valid_severity_preds, average='macro', zero_division=0
        )
    else:
        severity_acc = severity_prec = severity_rec = severity_f1 = 0.0
    
    hierarchical_acc = np.mean(all_hierarchical_correct)
    
    print(f"\n{'='*60}")
    print("CASCADE RESULTS")
    print(f"{'='*60}")
    print(f"Disease Accuracy: {disease_acc:.4f}")
    print(f"Disease Macro-F1: {disease_f1:.4f}")
    print(f"\nSeverity Accuracy (diseased only): {severity_acc:.4f}")
    print(f"Severity Macro-F1 (diseased only): {severity_f1:.4f}")
    print(f"\nHierarchical Accuracy: {hierarchical_acc:.4f}")
    
    results = {
        'disease': {
            'accuracy': float(disease_acc),
            'precision': float(disease_prec),
            'recall': float(disease_rec),
            'f1_macro': float(disease_f1)
        },
        'severity': {
            'accuracy': float(severity_acc),
            'precision': float(severity_prec),
            'recall': float(severity_rec),
            'f1_macro': float(severity_f1)
        },
        'hierarchical_accuracy': float(hierarchical_acc)
    }
    
    # Save results
    with open(save_dir / f"results_fold{fold_idx}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Default configuration
    config = {
        'batch_size': 32,
        'warmup_epochs': 5,
        'max_epochs': 25,
        'disease_backbone_lr': 5e-5,
        'disease_head_lr': 1e-3,
        'severity_backbone_lr': 5e-5,
        'severity_head_lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 0.25,
        'patience': 5
    }
    
    # Train on fold 0 for demonstration
    fold_idx = 0
    save_dir = RESULTS_DIR / f"fold_{fold_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"OPTION B - TWO-STAGE CASCADE")
    print(f"Training on Fold {fold_idx}")
    print(f"{'='*80}")
    
    start_time = time.time()
    results = train_option_b(fold_idx, config, save_dir)
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Training completed in {elapsed_time/60:.2f} minutes")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}")
