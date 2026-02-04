"""
Option A: Flat 10-class Classification
EfficientNet-B1 with 10-class head for cattle disease classification
Implements Steps 5, 6 (Option A), 7, 8, 9
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm
import time


class EfficientNetB1_OptionA(nn.Module):
    """
    Option A: EfficientNet-B1 with 10-class output head
    """
    def __init__(self, num_classes=10, dropout=0.4, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B1
        self.backbone = models.efficientnet_b1(pretrained=pretrained)
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1: warm-up)"""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning (Phase 2)"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def compute_hierarchical_metrics(y_true_label10, y_pred_label10, label10_to_idx):
    """
    Compute disease, severity, and hierarchical metrics from label_10 predictions
    """
    idx_to_label10 = {v: k for k, v in label10_to_idx.items()}
    
    # Convert indices to label_10 strings
    y_true_str = [idx_to_label10[idx] for idx in y_true_label10]
    y_pred_str = [idx_to_label10[idx] for idx in y_pred_label10]
    
    # Parse disease and severity
    def parse_label10(label):
        if label == 'healthy':
            return 'healthy', None
        else:
            # e.g., 'fmd_s1' -> 'fmd', 1
            parts = label.split('_s')
            disease = parts[0]
            severity = int(parts[1])
            return disease, severity
    
    y_true_disease = []
    y_true_severity = []
    y_pred_disease = []
    y_pred_severity = []
    
    for true_label, pred_label in zip(y_true_str, y_pred_str):
        true_dis, true_sev = parse_label10(true_label)
        pred_dis, pred_sev = parse_label10(pred_label)
        
        y_true_disease.append(true_dis)
        y_pred_disease.append(pred_dis)
        
        if true_sev is not None:  # Only for diseased samples
            y_true_severity.append(true_sev)
            y_pred_severity.append(pred_sev if pred_sev is not None else 1)  # Default to 1 if predicted healthy
    
    # Disease metrics (4 classes)
    disease_acc = accuracy_score(y_true_disease, y_pred_disease)
    disease_f1 = f1_score(y_true_disease, y_pred_disease, average='macro', zero_division=0)
    
    # Severity metrics (3 classes, diseased only)
    if len(y_true_severity) > 0:
        severity_acc = accuracy_score(y_true_severity, y_pred_severity)
        severity_f1 = f1_score(y_true_severity, y_pred_severity, average='macro', zero_division=0)
    else:
        severity_acc = 0.0
        severity_f1 = 0.0
    
    # Hierarchical accuracy: correct if both disease and severity are correct (or healthy→healthy)
    hierarchical_correct = 0
    for true_label, pred_label in zip(y_true_str, y_pred_str):
        true_dis, true_sev = parse_label10(true_label)
        pred_dis, pred_sev = parse_label10(pred_label)
        
        if true_label == 'healthy' and pred_label == 'healthy':
            hierarchical_correct += 1
        elif true_dis == pred_dis and true_sev == pred_sev:
            hierarchical_correct += 1
    
    hierarchical_acc = hierarchical_correct / len(y_true_str)
    
    return {
        'disease_accuracy': disease_acc,
        'disease_macro_f1': disease_f1,
        'severity_accuracy': severity_acc,
        'severity_macro_f1': severity_f1,
        'hierarchical_accuracy': hierarchical_acc,
        'y_true_disease': y_true_disease,
        'y_pred_disease': y_pred_disease,
        'y_true_severity': y_true_severity,
        'y_pred_severity': y_pred_severity
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_labels, all_preds


def validate(model, val_loader, criterion, device, epoch, phase='Val'):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [{phase}]')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_labels, all_preds


def plot_training_curves(history, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Disease F1
    axes[0, 2].plot(epochs, history['train_disease_f1'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_disease_f1'], 'r-', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Macro F1')
    axes[0, 2].set_title('Disease Macro F1')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Severity F1
    axes[1, 0].plot(epochs, history['train_severity_f1'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_severity_f1'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Macro F1')
    axes[1, 0].set_title('Severity Macro F1 (Diseased Only)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hierarchical Accuracy
    axes[1, 1].plot(epochs, history['train_hierarchical_acc'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_hierarchical_acc'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Hierarchical Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = history['best_epoch']
    for ax in axes.flat:
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_epoch}')
    
    # Learning rate
    axes[1, 2].plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to {os.path.join(save_dir, 'training_curves.png')}")


def plot_confusion_matrices(y_true, y_pred, metrics, save_dir, split_name='test'):
    """Plot confusion matrices for disease, severity, and label_10"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Disease confusion matrix (4x4)
    disease_labels = ['fmd', 'healthy', 'ibk', 'lsd']
    cm_disease = confusion_matrix(metrics['y_true_disease'], metrics['y_pred_disease'], labels=disease_labels)
    sns.heatmap(cm_disease, annot=True, fmt='d', cmap='Blues', xticklabels=disease_labels, 
                yticklabels=disease_labels, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Disease Confusion Matrix ({split_name.title()})\nAcc: {metrics["disease_accuracy"]:.3f}, F1: {metrics["disease_macro_f1"]:.3f}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Severity confusion matrix (3x3) - diseased only
    if len(metrics['y_true_severity']) > 0:
        severity_labels = [1, 2, 3]
        cm_severity = confusion_matrix(metrics['y_true_severity'], metrics['y_pred_severity'], labels=severity_labels)
        sns.heatmap(cm_severity, annot=True, fmt='d', cmap='Oranges', xticklabels=severity_labels,
                    yticklabels=severity_labels, ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_title(f'Severity Confusion Matrix ({split_name.title()}, Diseased Only)\nAcc: {metrics["severity_accuracy"]:.3f}, F1: {metrics["severity_macro_f1"]:.3f}')
        axes[1].set_ylabel('True Stage')
        axes[1].set_xlabel('Predicted Stage')
    else:
        axes[1].text(0.5, 0.5, 'No diseased samples', ha='center', va='center')
        axes[1].set_title(f'Severity Confusion Matrix ({split_name.title()})')
    
    # Label_10 confusion matrix (10x10)
    label10_names = ['fmd_s1', 'fmd_s2', 'fmd_s3', 'healthy', 'ibk_s1', 'ibk_s2', 'ibk_s3', 'lsd_s1', 'lsd_s2', 'lsd_s3']
    cm_label10 = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    sns.heatmap(cm_label10, annot=True, fmt='d', cmap='Greens', xticklabels=label10_names,
                yticklabels=label10_names, ax=axes[2], cbar_kws={'label': 'Count'})
    axes[2].set_title(f'Label_10 Confusion Matrix ({split_name.title()})\nHierarchical Acc: {metrics["hierarchical_accuracy"]:.3f}')
    axes[2].set_ylabel('True Label')
    axes[2].set_xlabel('Predicted Label')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrices_{split_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrices saved to {os.path.join(save_dir, f'confusion_matrices_{split_name}.png')}")


def save_metrics_report(metrics, save_path):
    """Save detailed metrics report"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OPTION A: FLAT 10-CLASS CLASSIFICATION - EVALUATION METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("DISEASE CLASSIFICATION (4 classes)\n")
        f.write("-"*80 + "\n")
        f.write(f"Disease Accuracy:        {metrics['disease_accuracy']:.4f}\n")
        f.write(f"Disease Macro F1:        {metrics['disease_macro_f1']:.4f}\n\n")
        
        f.write("SEVERITY CLASSIFICATION (3 classes, diseased only)\n")
        f.write("-"*80 + "\n")
        f.write(f"Severity Accuracy:       {metrics['severity_accuracy']:.4f}\n")
        f.write(f"Severity Macro F1:       {metrics['severity_macro_f1']:.4f}\n\n")
        
        f.write("HIERARCHICAL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Hierarchical Accuracy:   {metrics['hierarchical_accuracy']:.4f}\n")
        f.write("(Correct if: healthy→healthy OR disease+severity both correct)\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"✅ Metrics report saved to {save_path}")


# The training script will continue in the next file due to length...
