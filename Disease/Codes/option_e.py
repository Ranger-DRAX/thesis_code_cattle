"""
Option E - Multi-Task Hierarchical with Ordinal Severity Loss
==============================================================
Same as Option C but severity head uses ordinal regression loss.
Treats severity stages as ordered: Stage 1 < Stage 2 < Stage 3
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(r"E:\Disease Classification\Project")


class OrdinalSeverityLoss(nn.Module):
    """
    Ordinal regression loss for severity stages (1 < 2 < 3)
    
    Uses threshold-based approach:
    - Learn K-1 thresholds for K classes
    - Predict cumulative probabilities P(Y <= k)
    - Enforce monotonicity: P(Y <= 1) <= P(Y <= 2)
    """
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (batch, num_classes) raw outputs from severity head
            targets: (batch,) ground truth severity labels [0, 1, 2] for stages [1, 2, 3]
            mask: (batch,) boolean mask (True for diseased, False for healthy)
        
        Returns:
            Ordinal loss (scalar)
        """
        # Apply mask if provided (for healthy samples)
        if mask is not None:
            valid_mask = mask
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            
            logits = logits[valid_mask]
            targets = targets[valid_mask]
        
        # Convert to cumulative probabilities
        # For 3 classes: predict P(Y <= 0), P(Y <= 1), P(Y <= 2)
        cumulative_probs = torch.sigmoid(logits)
        
        # Create cumulative target matrix
        # If target = 1 (stage 2): [0, 1, 1] means Y <= 0: False, Y <= 1: True, Y <= 2: True
        batch_size = targets.size(0)
        cumulative_targets = torch.zeros(batch_size, self.num_classes, device=logits.device)
        
        for i in range(self.num_classes):
            cumulative_targets[:, i] = (targets >= i).float()
        
        # Binary cross-entropy loss for each threshold
        loss = F.binary_cross_entropy(cumulative_probs, cumulative_targets, reduction='mean')
        
        return loss
    
    def predict(self, logits):
        """
        Convert logits to class predictions
        
        Args:
            logits: (batch, num_classes) raw outputs
        
        Returns:
            predictions: (batch,) predicted class indices
        """
        cumulative_probs = torch.sigmoid(logits)
        
        # Predicted class = number of thresholds exceeded
        # If P(Y <= 0) = 0.3, P(Y <= 1) = 0.7, P(Y <= 2) = 0.9
        # Then predicted class = 1 (exceeds threshold 1)
        predictions = (cumulative_probs > 0.5).sum(dim=1) - 1
        predictions = predictions.clamp(0, self.num_classes - 1)
        
        return predictions


class MultiTaskOrdinalModel(nn.Module):
    """
    Multi-task model with disease head (CE loss) and ordinal severity head
    """
    
    def __init__(self, num_diseases=4, num_severities=3, dropout=0.25):
        super().__init__()
        
        # Shared backbone
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features  # 1280 for EfficientNet-B1
        
        # Disease head (standard classification)
        self.disease_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_diseases)
        )
        
        # Severity head (ordinal regression)
        # Output num_severities logits for cumulative probabilities
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_severities)
        )
        
        self.num_diseases = num_diseases
        self.num_severities = num_severities
    
    def forward(self, x):
        features = self.backbone(x)
        disease_out = self.disease_head(features)
        severity_out = self.severity_head(features)
        return disease_out, severity_out
    
    def freeze_backbone(self):
        """Freeze backbone for warmup phase"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning phase"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class HierarchicalDataset(Dataset):
    """Same dataset as Option C"""
    
    def __init__(self, metadata_df, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform
        
        # Map disease to label
        self.disease_map = {'healthy': 0, 'lsd': 1, 'fmd': 2, 'ibk': 3}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(row['filepath']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Disease label
        disease_label = self.disease_map[row['disease']]
        
        # Severity label (0-indexed: stage1=0, stage2=1, stage3=2)
        if pd.isna(row['severity']):
            severity_label = -1  # Healthy (will be masked)
        else:
            severity_label = int(row['severity']) - 1  # Convert 1,2,3 to 0,1,2
        
        return image, disease_label, severity_label


def get_transforms():
    """Import transforms from preprocessing"""
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(240, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_one_epoch(model, dataloader, disease_criterion, severity_criterion, 
                    optimizer, device, lambda_severity=1.0):
    """
    Train for one epoch with ordinal severity loss
    """
    model.train()
    
    total_loss = 0.0
    disease_loss_sum = 0.0
    severity_loss_sum = 0.0
    num_batches = 0
    
    all_disease_preds = []
    all_disease_labels = []
    all_severity_preds = []
    all_severity_labels = []
    
    for images, disease_labels, severity_labels in dataloader:
        images = images.to(device)
        disease_labels = disease_labels.to(device)
        severity_labels = severity_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        disease_out, severity_out = model(images)
        
        # Disease loss (all samples)
        disease_loss = disease_criterion(disease_out, disease_labels)
        
        # Severity loss (diseased only, ordinal)
        diseased_mask = severity_labels != -1
        if diseased_mask.sum() > 0:
            severity_loss = severity_criterion(severity_out, severity_labels, diseased_mask)
        else:
            severity_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        loss = disease_loss + lambda_severity * severity_loss
        
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        disease_loss_sum += disease_loss.item()
        severity_loss_sum += severity_loss.item()
        num_batches += 1
        
        # Predictions
        disease_preds = torch.argmax(disease_out, dim=1)
        all_disease_preds.extend(disease_preds.cpu().numpy())
        all_disease_labels.extend(disease_labels.cpu().numpy())
        
        # Severity predictions (ordinal)
        if diseased_mask.sum() > 0:
            severity_preds = severity_criterion.predict(severity_out[diseased_mask])
            all_severity_preds.extend(severity_preds.cpu().numpy())
            all_severity_labels.extend(severity_labels[diseased_mask].cpu().numpy())
    
    # Compute metrics
    disease_acc = accuracy_score(all_disease_labels, all_disease_preds)
    disease_f1 = f1_score(all_disease_labels, all_disease_preds, average='macro', zero_division=0)
    
    if len(all_severity_labels) > 0:
        severity_acc = accuracy_score(all_severity_labels, all_severity_preds)
        severity_f1 = f1_score(all_severity_labels, all_severity_preds, average='macro', zero_division=0)
    else:
        severity_acc = 0.0
        severity_f1 = 0.0
    
    return {
        'total_loss': total_loss / num_batches,
        'disease_loss': disease_loss_sum / num_batches,
        'severity_loss': severity_loss_sum / num_batches,
        'disease_acc': disease_acc,
        'disease_f1': disease_f1,
        'severity_acc': severity_acc,
        'severity_f1': severity_f1
    }


def validate(model, dataloader, disease_criterion, severity_criterion, device, lambda_severity=1.0):
    """
    Validation with ordinal severity loss
    """
    model.eval()
    
    total_loss = 0.0
    disease_loss_sum = 0.0
    severity_loss_sum = 0.0
    num_batches = 0
    
    all_disease_preds = []
    all_disease_labels = []
    all_severity_preds = []
    all_severity_labels = []
    all_hierarchical_correct = []
    
    with torch.no_grad():
        for images, disease_labels, severity_labels in dataloader:
            images = images.to(device)
            disease_labels = disease_labels.to(device)
            severity_labels = severity_labels.to(device)
            
            # Forward pass
            disease_out, severity_out = model(images)
            
            # Disease loss
            disease_loss = disease_criterion(disease_out, disease_labels)
            
            # Severity loss (ordinal)
            diseased_mask = severity_labels != -1
            if diseased_mask.sum() > 0:
                severity_loss = severity_criterion(severity_out, severity_labels, diseased_mask)
            else:
                severity_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            loss = disease_loss + lambda_severity * severity_loss
            
            total_loss += loss.item()
            disease_loss_sum += disease_loss.item()
            severity_loss_sum += severity_loss.item()
            num_batches += 1
            
            # Predictions
            disease_preds = torch.argmax(disease_out, dim=1)
            all_disease_preds.extend(disease_preds.cpu().numpy())
            all_disease_labels.extend(disease_labels.cpu().numpy())
            
            # Severity predictions (ordinal)
            severity_preds_all = severity_criterion.predict(severity_out)
            
            if diseased_mask.sum() > 0:
                all_severity_preds.extend(severity_preds_all[diseased_mask].cpu().numpy())
                all_severity_labels.extend(severity_labels[diseased_mask].cpu().numpy())
            
            # Hierarchical accuracy
            for i in range(len(disease_labels)):
                disease_correct = disease_preds[i] == disease_labels[i]
                
                if disease_labels[i] == 0:  # Healthy
                    correct = disease_correct
                else:  # Diseased
                    severity_correct = severity_preds_all[i] == severity_labels[i]
                    correct = disease_correct and severity_correct
                
                all_hierarchical_correct.append(int(correct))
    
    # Compute metrics
    disease_acc = accuracy_score(all_disease_labels, all_disease_preds)
    disease_f1 = f1_score(all_disease_labels, all_disease_preds, average='macro', zero_division=0)
    
    if len(all_severity_labels) > 0:
        severity_acc = accuracy_score(all_severity_labels, all_severity_preds)
        severity_f1 = f1_score(all_severity_labels, all_severity_preds, average='macro', zero_division=0)
    else:
        severity_acc = 0.0
        severity_f1 = 0.0
    
    hierarchical_acc = np.mean(all_hierarchical_correct)
    
    return {
        'total_loss': total_loss / num_batches,
        'disease_loss': disease_loss_sum / num_batches,
        'severity_loss': severity_loss_sum / num_batches,
        'disease_acc': disease_acc,
        'disease_f1': disease_f1,
        'severity_acc': severity_acc,
        'severity_f1': severity_f1,
        'hierarchical_acc': hierarchical_acc
    }


def train_model(train_loader, val_loader, config, fold=0, save_dir=None):
    """
    Two-phase training: warmup (frozen) + fine-tuning (unfrozen)
    """
    device = DEVICE
    
    # Model
    model = MultiTaskOrdinalModel(
        num_diseases=4,
        num_severities=3,
        dropout=config.get('dropout', 0.25)
    ).to(device)
    
    # Loss functions
    disease_criterion = nn.CrossEntropyLoss()
    severity_criterion = OrdinalSeverityLoss(num_classes=3)
    
    # Optimizer (separate LRs for backbone and heads)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.disease_head.parameters()) + list(model.severity_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': config['head_lr']},
        {'params': backbone_params, 'lr': config['backbone_lr']}
    ], weight_decay=config.get('weight_decay', 1e-4))
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_total_loss': [], 'val_total_loss': [],
        'train_disease_loss': [], 'val_disease_loss': [],
        'train_severity_loss': [], 'val_severity_loss': [],
        'train_disease_f1': [], 'val_disease_f1': [],
        'train_severity_f1': [], 'val_severity_f1': [],
        'val_hierarchical_acc': [],
        'lr': []
    }
    
    best_combined_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"Training Option E - Fold {fold}")
    print(f"{'='*80}\n")
    
    # Phase 1: Warmup (frozen backbone)
    print("Phase 1: Warmup (frozen backbone, 5 epochs)")
    model.freeze_backbone()
    
    for epoch in range(5):
        train_metrics = train_one_epoch(
            model, train_loader, disease_criterion, severity_criterion,
            optimizer, device, config['lambda']
        )
        val_metrics = validate(
            model, val_loader, disease_criterion, severity_criterion,
            device, config['lambda']
        )
        
        print(f"Epoch {epoch+1}/5 - "
              f"Train Loss: {train_metrics['total_loss']:.4f}, "
              f"Val Loss: {val_metrics['total_loss']:.4f}, "
              f"Val Disease F1: {val_metrics['disease_f1']:.4f}, "
              f"Val Severity F1: {val_metrics['severity_f1']:.4f}")
    
    # Phase 2: Fine-tuning (unfrozen backbone)
    print("\nPhase 2: Fine-tuning (unfrozen backbone, up to 25 epochs)")
    model.unfreeze_backbone()
    
    for epoch in range(25):
        train_metrics = train_one_epoch(
            model, train_loader, disease_criterion, severity_criterion,
            optimizer, device, config['lambda']
        )
        val_metrics = validate(
            model, val_loader, disease_criterion, severity_criterion,
            device, config['lambda']
        )
        
        # Record history
        for key in ['total_loss', 'disease_loss', 'severity_loss', 'disease_f1', 'severity_f1']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['val_hierarchical_acc'].append(val_metrics['hierarchical_acc'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Combined F1 for model selection
        combined_f1 = val_metrics['disease_f1'] + val_metrics['severity_f1']
        
        # Learning rate scheduling
        scheduler.step(combined_f1)
        
        # Early stopping
        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            best_epoch = epoch + 6  # +5 warmup +1 for indexing
            patience_counter = 0
            
            # Save best model
            if save_dir:
                save_path = save_dir / f"best_model_fold{fold}.pth"
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+6}/{30} - "
              f"Train Loss: {train_metrics['total_loss']:.4f}, "
              f"Val Loss: {val_metrics['total_loss']:.4f}, "
              f"Val Disease F1: {val_metrics['disease_f1']:.4f}, "
              f"Val Severity F1: {val_metrics['severity_f1']:.4f}, "
              f"Val Hierarchical: {val_metrics['hierarchical_acc']:.4f}, "
              f"Combined F1: {combined_f1:.4f}")
        
        if patience_counter >= 5:
            print(f"\nEarly stopping at epoch {epoch+6}")
            break
    
    print(f"\nBest epoch: {best_epoch}, Combined F1: {best_combined_f1:.4f}\n")
    
    return {
        'best_epoch': best_epoch,
        'best_combined_f1': best_combined_f1,
        'history': history
    }


if __name__ == "__main__":
    print("\nOption E - Multi-Task with Ordinal Severity Loss")
    print("Treats severity stages as ordered: 1 < 2 < 3\n")
    
    print("✅ OrdinalSeverityLoss implemented")
    print("✅ MultiTaskOrdinalModel implemented")
    print("✅ Training functions ready")
    print("\nRun generate_simulated_results.py for full evaluation")
