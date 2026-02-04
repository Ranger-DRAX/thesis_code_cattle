"""
Option D - Multi-Task with Severity Masking
============================================
Uses the same multi-task architecture as Option C with intelligent inference:
- If predicted disease = healthy (or P(healthy) > threshold) → severity = N/A
- Else output severity stage

This option trains the same model as Option C but applies masking during inference.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import time

# Import Option C model and training components
sys.path.append(str(Path(__file__).parent.parent / "Option-C"))
from option_c import (
    MultiTaskHierarchicalModel, 
    HierarchicalDataset, 
    get_transforms,
    compute_class_weights,
    train_one_epoch as c_train_one_epoch,
    validate as c_validate
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(r"E:\Disease Classification\Project")
RESULTS_DIR = PROJECT_ROOT / "Results" / "Option-D Metrics"
OPTION_D_MODEL_DIR = RESULTS_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OPTION_D_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_option_d_model(fold=0):
    """
    Load the trained Option D model (same architecture as Option C)
    """
    model_path = OPTION_D_MODEL_DIR / f"model_fold{fold}.pth"
    
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print("Model needs to be trained first")
        return None
    
    model = MultiTaskHierarchicalModel(dropout=0.25)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model


def evaluate_with_threshold(model, dataloader, healthy_threshold=0.5):
    """
    Evaluate Option D with severity masking based on healthy probability threshold
    
    Args:
        model: Trained Option C model
        dataloader: Test/validation dataloader
        healthy_threshold: If P(healthy) > threshold, set severity = N/A
    
    Returns:
        Dictionary with metrics
    """
    all_disease_preds = []
    all_disease_labels = []
    all_severity_preds = []
    all_severity_labels = []
    all_hierarchical_correct = []
    
    with torch.no_grad():
        for images, disease_labels, severity_labels in dataloader:
            images = images.to(DEVICE)
            disease_labels = disease_labels.to(DEVICE)
            severity_labels = severity_labels.to(DEVICE)
            
            # Forward pass
            disease_out, severity_out = model(images)
            
            # Get disease predictions and probabilities
            disease_probs = torch.softmax(disease_out, dim=1)
            disease_preds = torch.argmax(disease_out, dim=1)
            healthy_probs = disease_probs[:, 0]  # Assuming healthy is class 0
            
            # Get severity predictions
            severity_preds = torch.argmax(severity_out, dim=1)
            
            # Apply Option D inference rule
            # If P(healthy) > threshold OR predicted disease = healthy → mask severity
            mask_severity = (healthy_probs > healthy_threshold) | (disease_preds == 0)
            
            # For masked samples, set severity to -1 (N/A)
            severity_preds_masked = severity_preds.clone()
            severity_preds_masked[mask_severity] = -1
            
            # Store predictions
            all_disease_preds.extend(disease_preds.cpu().numpy())
            all_disease_labels.extend(disease_labels.cpu().numpy())
            
            # Only evaluate severity on diseased samples (ground truth)
            diseased_mask = severity_labels != -1
            if diseased_mask.sum() > 0:
                # For diseased samples, check if we correctly predicted them as diseased
                diseased_indices = diseased_mask.nonzero(as_tuple=True)[0]
                
                for idx in diseased_indices:
                    true_sev = severity_labels[idx].item()
                    pred_sev = severity_preds_masked[idx].item()
                    
                    all_severity_labels.append(true_sev)
                    all_severity_preds.append(pred_sev if pred_sev != -1 else -1)
            
            # Hierarchical accuracy
            for i in range(len(disease_labels)):
                disease_correct = disease_preds[i] == disease_labels[i]
                
                if disease_labels[i] == 0:  # True healthy
                    # Correct if predicted healthy
                    correct = disease_correct
                else:  # True diseased
                    # Correct if disease correct AND severity correct
                    if disease_correct and not mask_severity[i]:
                        severity_correct = severity_preds[i] == severity_labels[i]
                        correct = severity_correct
                    else:
                        correct = False
                
                all_hierarchical_correct.append(int(correct))
    
    # Compute disease metrics (4-class)
    disease_acc = accuracy_score(all_disease_labels, all_disease_preds)
    disease_f1 = f1_score(all_disease_labels, all_disease_preds, average='macro', zero_division=0)
    disease_precision = precision_score(all_disease_labels, all_disease_preds, average='macro', zero_division=0)
    disease_recall = recall_score(all_disease_labels, all_disease_preds, average='macro', zero_division=0)
    
    # Compute severity metrics (3-class, diseased only, excluding masked predictions)
    valid_severity_mask = np.array(all_severity_preds) != -1
    if valid_severity_mask.sum() > 0:
        valid_severity_labels = np.array(all_severity_labels)[valid_severity_mask]
        valid_severity_preds = np.array(all_severity_preds)[valid_severity_mask]
        
        severity_acc = accuracy_score(valid_severity_labels, valid_severity_preds)
        severity_f1 = f1_score(valid_severity_labels, valid_severity_preds, average='macro', zero_division=0)
        severity_precision = precision_score(valid_severity_labels, valid_severity_preds, average='macro', zero_division=0)
        severity_recall = recall_score(valid_severity_labels, valid_severity_preds, average='macro', zero_division=0)
    else:
        severity_acc = 0.0
        severity_f1 = 0.0
        severity_precision = 0.0
        severity_recall = 0.0
    
    # Hierarchical accuracy
    hierarchical_acc = np.mean(all_hierarchical_correct)
    
    # Count masked predictions
    num_total = len(all_disease_preds)
    num_masked = np.sum(np.array(all_severity_preds) == -1)
    num_should_be_masked = np.sum(np.array(all_disease_labels) == 0)  # True healthy
    
    return {
        'disease': {
            'accuracy': disease_acc,
            'f1_macro': disease_f1,
            'precision_macro': disease_precision,
            'recall_macro': disease_recall
        },
        'severity': {
            'accuracy': severity_acc,
            'f1_macro': severity_f1,
            'precision_macro': severity_precision,
            'recall_macro': severity_recall,
            'num_evaluated': int(valid_severity_mask.sum()),
            'num_masked': int(num_masked)
        },
        'hierarchical_accuracy': hierarchical_acc,
        'masking_stats': {
            'total_samples': int(num_total),
            'masked_predictions': int(num_masked),
            'true_healthy_samples': int(num_should_be_masked),
            'masking_accuracy': float(num_masked / num_should_be_masked) if num_should_be_masked > 0 else 0.0
        }
    }


def tune_threshold_on_validation(model, val_loader):
    """
    Tune the healthy threshold on validation set
    Try thresholds from 0.3 to 0.9
    """
    print("\n" + "="*80)
    print("THRESHOLD TUNING FOR OPTION D")
    print("="*80 + "\n")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for threshold in thresholds:
        metrics = evaluate_with_threshold(model, val_loader, threshold)
        
        results.append({
            'threshold': threshold,
            'disease_f1': metrics['disease']['f1_macro'],
            'severity_f1': metrics['severity']['f1_macro'],
            'hierarchical_acc': metrics['hierarchical_accuracy'],
            'combined_score': metrics['disease']['f1_macro'] + metrics['severity']['f1_macro'],
            'num_masked': metrics['masking_stats']['masked_predictions']
        })
        
        print(f"Threshold {threshold:.1f}: "
              f"Disease F1={metrics['disease']['f1_macro']:.4f}, "
              f"Severity F1={metrics['severity']['f1_macro']:.4f}, "
              f"Hierarchical={metrics['hierarchical_accuracy']:.4f}, "
              f"Masked={metrics['masking_stats']['masked_predictions']}")
    
    # Select best threshold by hierarchical accuracy
    best = max(results, key=lambda x: x['hierarchical_acc'])
    
    print(f"\n✅ Best threshold: {best['threshold']:.1f}")
    print(f"   Hierarchical Accuracy: {best['hierarchical_acc']:.4f}")
    print(f"   Disease F1: {best['disease_f1']:.4f}")
    print(f"   Severity F1: {best['severity_f1']:.4f}")
    print(f"   Masked predictions: {best['num_masked']}\n")
    
    return best['threshold'], results


def train_option_d(fold_idx, config, save_dir):
    """
    Train Option D model (same architecture as Option C)
    The only difference is in inference, where we apply severity masking
    """
    device = DEVICE
    print(f"\nUsing device: {device}")
    
    # Load metadata and splits
    metadata = pd.read_csv(PROJECT_ROOT / "metadata.csv")
    folds_df = pd.read_csv(PROJECT_ROOT / "splits" / "folds.csv")
    
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
    
    # Create model (same as Option C)
    model = MultiTaskHierarchicalModel(dropout=config['dropout']).to(device)
    
    # Update config with weights
    config['disease_weights'] = disease_weights
    config['severity_weights'] = severity_weights
    
    # Train using Option C's training function
    print(f"\n{'='*80}")
    print(f"Training Option D Model (Same architecture as Option C)")
    print(f"{'='*80}")
    
    # Loss functions
    disease_criterion = nn.CrossEntropyLoss(weight=disease_weights.to(device))
    severity_criterion = nn.CrossEntropyLoss(
        weight=severity_weights.to(device),
        ignore_index=-1
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
    
    # PHASE 1: Warmup (frozen backbone)
    print(f"\nPhase 1: Warmup (frozen backbone) - {config['warmup_epochs']} epochs")
    model.freeze_backbone()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['head_lr'],
        weight_decay=config['weight_decay']
    )
    
    for epoch in range(config['warmup_epochs']):
        train_metrics = c_train_one_epoch(
            model, train_loader, disease_criterion, severity_criterion,
            optimizer, device, config['lambda_severity']
        )
        val_metrics = c_validate(
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
    
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['backbone_lr']},
        {'params': model.disease_head.parameters(), 'lr': config['head_lr']},
        {'params': model.severity_head.parameters(), 'lr': config['head_lr']}
    ], weight_decay=config['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True, min_lr=1e-6
    )
    
    for epoch in range(config['warmup_epochs'], config['warmup_epochs'] + config['max_epochs']):
        train_metrics = c_train_one_epoch(
            model, train_loader, disease_criterion, severity_criterion,
            optimizer, device, config['lambda_severity']
        )
        val_metrics = c_validate(
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
    
    # Save model
    torch.save(model.state_dict(), save_dir / f"model_fold{fold_idx}.pth")
    
    # Evaluate with Option D inference (masking)
    print(f"\n{'='*80}")
    print("EVALUATING WITH OPTION D INFERENCE (SEVERITY MASKING)")
    print(f"{'='*80}")
    
    best_threshold, threshold_results = tune_threshold_on_validation(model, val_loader)
    
    # Final evaluation with best threshold
    final_metrics = evaluate_with_threshold(model, val_loader, best_threshold)
    
    # Save results
    results = {
        'training': {
            'history': history,
            'best_combined_f1': float(best_val_f1_combined),
            'best_epoch': int(best_epoch)
        },
        'inference': {
            'best_threshold': float(best_threshold),
            'threshold_tuning_results': threshold_results,
            'final_metrics': final_metrics
        }
    }
    
    with open(save_dir / f"results_fold{fold_idx}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("FINAL OPTION D RESULTS")
    print(f"{'='*80}")
    print(f"Disease Accuracy: {final_metrics['disease']['accuracy']:.4f}")
    print(f"Disease F1: {final_metrics['disease']['f1_macro']:.4f}")
    print(f"Severity Accuracy: {final_metrics['severity']['accuracy']:.4f}")
    print(f"Severity F1: {final_metrics['severity']['f1_macro']:.4f}")
    print(f"Hierarchical Accuracy: {final_metrics['hierarchical_accuracy']:.4f}")
    print(f"Best threshold: {best_threshold:.2f}")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPTION D - MULTI-TASK WITH SEVERITY MASKING")
    print("Same architecture as Option C with intelligent inference")
    print("="*80 + "\n")
    
    # Configuration
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
    
    print(f"Training on Fold {fold_idx}")
    print(f"Results will be saved to: {save_dir}\n")
    
    start_time = time.time()
    results = train_option_d(fold_idx, config, save_dir)
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Training and evaluation completed in {elapsed_time/60:.2f} minutes")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}")
