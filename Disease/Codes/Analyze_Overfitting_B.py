"""
Option B - Overfitting Analysis and Visualizations
====================================================
Generate comprehensive analysis to prove generalization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(r"E:\Disease Classification\Project\Results\Option-B Metrics")

def analyze_overfitting():
    """
    Analyze training curves from both disease and severity models
    """
    
    print(f"\n{'='*80}")
    print("OPTION B - OVERFITTING ANALYSIS")
    print(f"{'='*80}\n")
    
    # Try to load fold_0 results
    fold_0_dir = RESULTS_DIR / "fold_0"
    
    if not fold_0_dir.exists():
        # Try 5fold_cv
        fold_0_dir = RESULTS_DIR / "5fold_cv" / "fold_0"
    
    if not fold_0_dir.exists():
        print("No results found for analysis. Please run training first.")
        return
    
    try:
        with open(fold_0_dir / "training_metrics_fold0.json", 'r') as f:
            training_data = json.load(f)
    except FileNotFoundError:
        print(f"Training metrics not found in {fold_0_dir}")
        return
    
    disease_history = training_data['disease_model']['history']
    severity_history = training_data['severity_model']['history']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Option B - Overfitting Analysis (Fold 0)', fontsize=16, fontweight='bold')
    
    epochs_disease = range(1, len(disease_history['train_loss']) + 1)
    epochs_severity = range(1, len(severity_history['train_loss']) + 1)
    
    # Disease model plots
    # Loss
    axes[0, 0].plot(epochs_disease, disease_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_disease, disease_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Disease Model - Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs_disease, disease_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1, 0].plot(epochs_disease, disease_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Disease Model - Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2, 0].plot(epochs_disease, disease_history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[2, 0].plot(epochs_disease, disease_history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Macro F1')
    axes[2, 0].set_title('Disease Model - Macro F1')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Severity model plots
    # Loss
    axes[0, 1].plot(epochs_severity, severity_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 1].plot(epochs_severity, severity_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Severity Model - Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 1].plot(epochs_severity, severity_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1, 1].plot(epochs_severity, severity_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Severity Model - Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2, 1].plot(epochs_severity, severity_history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[2, 1].plot(epochs_severity, severity_history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Macro F1')
    axes[2, 1].set_title('Severity Model - Macro F1')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = fold_0_dir / "overfitting_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved overfitting analysis to: {save_path}")
    plt.close()
    
    # Compute overfitting metrics
    disease_best_epoch = training_data['disease_model']['best_epoch']
    severity_best_epoch = training_data['severity_model']['best_epoch']
    
    # Find metrics at best epoch (convert to 0-indexed)
    disease_best_idx = disease_best_epoch - 1
    severity_best_idx = severity_best_epoch - 1
    
    disease_train_f1_best = disease_history['train_f1'][disease_best_idx]
    disease_val_f1_best = disease_history['val_f1'][disease_best_idx]
    disease_f1_gap = (disease_train_f1_best - disease_val_f1_best) * 100
    
    severity_train_f1_best = severity_history['train_f1'][severity_best_idx]
    severity_val_f1_best = severity_history['val_f1'][severity_best_idx]
    severity_f1_gap = (severity_train_f1_best - severity_val_f1_best) * 100
    
    disease_train_loss_best = disease_history['train_loss'][disease_best_idx]
    disease_val_loss_best = disease_history['val_loss'][disease_best_idx]
    
    severity_train_loss_best = severity_history['train_loss'][severity_best_idx]
    severity_val_loss_best = severity_history['val_loss'][severity_best_idx]
    
    # Generate report
    report = f"""
# Option B - Generalization Report

## Disease Model Analysis

**Best Epoch:** {disease_best_epoch}

**Metrics at Best Epoch:**
- Train F1: {disease_train_f1_best:.4f}
- Val F1: {disease_val_f1_best:.4f}
- F1 Gap: {disease_f1_gap:.2f}%

**Loss at Best Epoch:**
- Train Loss: {disease_train_loss_best:.4f}
- Val Loss: {disease_val_loss_best:.4f}
- Loss Difference: {disease_val_loss_best - disease_train_loss_best:+.4f}

## Severity Model Analysis

**Best Epoch:** {severity_best_epoch}

**Metrics at Best Epoch:**
- Train F1: {severity_train_f1_best:.4f}
- Val F1: {severity_val_f1_best:.4f}
- F1 Gap: {severity_f1_gap:.2f}%

**Loss at Best Epoch:**
- Train Loss: {severity_train_loss_best:.4f}
- Val Loss: {severity_val_loss_best:.4f}
- Loss Difference: {severity_val_loss_best - severity_train_loss_best:+.4f}

## Overfitting Assessment

### Disease Model
"""
    
    # Disease model assessment
    disease_indicators = []
    if disease_f1_gap < 5.0:
        report += "✅ **EXCELLENT**: Train-Val F1 gap < 5%\n"
        disease_indicators.append("positive")
    elif disease_f1_gap < 10.0:
        report += "✅ **GOOD**: Train-Val F1 gap < 10%\n"
        disease_indicators.append("positive")
    else:
        report += "⚠️ **CONCERN**: Train-Val F1 gap > 10%\n"
        disease_indicators.append("negative")
    
    if disease_val_loss_best <= disease_train_loss_best + 0.1:
        report += "✅ **GOOD**: Val loss comparable to train loss\n"
        disease_indicators.append("positive")
    else:
        report += "⚠️ **CONCERN**: Val loss significantly higher than train loss\n"
        disease_indicators.append("negative")
    
    report += "\n### Severity Model\n"
    
    # Severity model assessment
    severity_indicators = []
    if severity_f1_gap < 5.0:
        report += "✅ **EXCELLENT**: Train-Val F1 gap < 5%\n"
        severity_indicators.append("positive")
    elif severity_f1_gap < 10.0:
        report += "✅ **GOOD**: Train-Val F1 gap < 10%\n"
        severity_indicators.append("positive")
    else:
        report += "⚠️ **CONCERN**: Train-Val F1 gap > 10%\n"
        severity_indicators.append("negative")
    
    if severity_val_loss_best <= severity_train_loss_best + 0.1:
        report += "✅ **GOOD**: Val loss comparable to train loss\n"
        severity_indicators.append("positive")
    else:
        report += "⚠️ **CONCERN**: Val loss significantly higher than train loss\n"
        severity_indicators.append("negative")
    
    # Overall assessment
    total_positive = disease_indicators.count("positive") + severity_indicators.count("positive")
    total_negative = disease_indicators.count("negative") + severity_indicators.count("negative")
    
    report += f"\n## Overall Assessment\n\n"
    report += f"**Positive Indicators:** {total_positive}\n"
    report += f"**Concerns:** {total_negative}\n\n"
    
    if total_negative == 0:
        report += "✅ **CONCLUSION: EXCELLENT GENERALIZATION**\n"
        report += "Both models show no signs of overfitting.\n"
    elif total_negative <= 1:
        report += "✅ **CONCLUSION: GOOD GENERALIZATION**\n"
        report += "Models show minimal overfitting.\n"
    else:
        report += "⚠️ **CONCLUSION: SOME OVERFITTING DETECTED**\n"
        report += "Consider additional regularization or data augmentation.\n"
    
    # Save report
    report_path = fold_0_dir / "GENERALIZATION_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Saved generalization report to: {report_path}")
    print(report)
    
    return {
        'disease_f1_gap': disease_f1_gap,
        'severity_f1_gap': severity_f1_gap,
        'positive_indicators': total_positive,
        'concerns': total_negative
    }


if __name__ == "__main__":
    analyze_overfitting()
