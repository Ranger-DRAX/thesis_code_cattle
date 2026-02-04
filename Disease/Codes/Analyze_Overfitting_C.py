"""
Option C - Overfitting Analysis and Visualizations
====================================================
Generate comprehensive analysis to prove generalization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(r"E:\Disease Classification\Project\Results\Option-C Metrics")

def analyze_overfitting():
    """
    Analyze training curves from multi-task model
    """
    
    print(f"\n{'='*80}")
    print("OPTION C - OVERFITTING ANALYSIS")
    print(f"{'='*80}\n")
    
    # Try to load fold_0 results
    fold_0_dir = RESULTS_DIR / "5fold_cv" / "fold_0"
    
    if not fold_0_dir.exists():
        print("No results found for analysis. Please run training first.")
        return
    
    try:
        with open(fold_0_dir / "results_fold0.json", 'r') as f:
            training_data = json.load(f)
    except FileNotFoundError:
        print(f"Training results not found in {fold_0_dir}")
        return
    
    history = training_data['training']['history']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Option C - Overfitting Analysis (Fold 0)', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # Total Loss
    axes[0, 0].plot(epochs, history['train_total_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_total_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Combined Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Disease Loss
    axes[0, 1].plot(epochs, history['train_disease_loss'], 'b-', label='Train Disease Loss', linewidth=2)
    axes[0, 1].plot(epochs, history['val_disease_loss'], 'r-', label='Val Disease Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Disease Head Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Severity Loss
    axes[1, 0].plot(epochs, history['train_severity_loss'], 'b-', label='Train Severity Loss', linewidth=2)
    axes[1, 0].plot(epochs, history['val_severity_loss'], 'r-', label='Val Severity Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Severity Head Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Disease F1
    axes[1, 1].plot(epochs, history['train_disease_f1'], 'b-', label='Train Disease F1', linewidth=2)
    axes[1, 1].plot(epochs, history['val_disease_f1'], 'r-', label='Val Disease F1', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Macro F1')
    axes[1, 1].set_title('Disease Head F1')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Severity F1
    axes[2, 0].plot(epochs, history['train_severity_f1'], 'b-', label='Train Severity F1', linewidth=2)
    axes[2, 0].plot(epochs, history['val_severity_f1'], 'r-', label='Val Severity F1', linewidth=2)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Macro F1')
    axes[2, 0].set_title('Severity Head F1')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2, 1].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Learning Rate')
    axes[2, 1].set_title('Learning Rate Schedule')
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = fold_0_dir / "overfitting_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved overfitting analysis to: {save_path}")
    plt.close()
    
    # Compute overfitting metrics
    best_epoch = training_data['training']['best_epoch']
    best_idx = best_epoch - 1
    
    train_disease_f1_best = history['train_disease_f1'][best_idx]
    val_disease_f1_best = history['val_disease_f1'][best_idx]
    disease_f1_gap = (train_disease_f1_best - val_disease_f1_best) * 100
    
    train_severity_f1_best = history['train_severity_f1'][best_idx]
    val_severity_f1_best = history['val_severity_f1'][best_idx]
    severity_f1_gap = (train_severity_f1_best - val_severity_f1_best) * 100
    
    train_total_loss_best = history['train_total_loss'][best_idx]
    val_total_loss_best = history['val_total_loss'][best_idx]
    
    train_disease_loss_best = history['train_disease_loss'][best_idx]
    val_disease_loss_best = history['val_disease_loss'][best_idx]
    
    train_severity_loss_best = history['train_severity_loss'][best_idx]
    val_severity_loss_best = history['val_severity_loss'][best_idx]
    
    # Generate report
    report = f"""
# Option C - Generalization Report

## Multi-Task Model Analysis

**Best Epoch:** {best_epoch}

### Disease Head

**Metrics at Best Epoch:**
- Train F1: {train_disease_f1_best:.4f}
- Val F1: {val_disease_f1_best:.4f}
- F1 Gap: {disease_f1_gap:.2f}%

**Loss at Best Epoch:**
- Train Loss: {train_disease_loss_best:.4f}
- Val Loss: {val_disease_loss_best:.4f}
- Loss Difference: {val_disease_loss_best - train_disease_loss_best:+.4f}

### Severity Head

**Metrics at Best Epoch:**
- Train F1: {train_severity_f1_best:.4f}
- Val F1: {val_severity_f1_best:.4f}
- F1 Gap: {severity_f1_gap:.2f}%

**Loss at Best Epoch:**
- Train Loss: {train_severity_loss_best:.4f}
- Val Loss: {val_severity_loss_best:.4f}
- Loss Difference: {val_severity_loss_best - train_severity_loss_best:+.4f}

### Combined Model

**Total Loss at Best Epoch:**
- Train Total Loss: {train_total_loss_best:.4f}
- Val Total Loss: {val_total_loss_best:.4f}
- Loss Difference: {val_total_loss_best - train_total_loss_best:+.4f}

## Overfitting Assessment

### Disease Head
"""
    
    # Disease head assessment
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
    
    if val_disease_loss_best <= train_disease_loss_best + 0.1:
        report += "✅ **GOOD**: Val loss comparable to train loss\n"
        disease_indicators.append("positive")
    else:
        report += "⚠️ **CONCERN**: Val loss significantly higher than train loss\n"
        disease_indicators.append("negative")
    
    report += "\n### Severity Head\n"
    
    # Severity head assessment
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
    
    if val_severity_loss_best <= train_severity_loss_best + 0.1:
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
        report += "Multi-task model shows no signs of overfitting on both heads.\n"
    elif total_negative <= 1:
        report += "✅ **CONCLUSION: GOOD GENERALIZATION**\n"
        report += "Multi-task model shows minimal overfitting.\n"
    else:
        report += "⚠️ **CONCLUSION: SOME OVERFITTING DETECTED**\n"
        report += "Consider additional regularization or lambda adjustment.\n"
    
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
