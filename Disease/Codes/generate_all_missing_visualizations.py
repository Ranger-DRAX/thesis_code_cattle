"""
Generate All Missing Training Visualizations
Generates training curves for Option E and MobileNetV3
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def generate_option_e_visualizations():
    """Generate all missing visualizations for Option E"""
    print(f"\n{'='*80}")
    print(f"GENERATING OPTION E VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    base_path = Path("Results/Option-E Metrics/5fold_cv")
    
    # Generate for each fold
    all_fold_data = []
    for fold_num in range(5):
        fold_path = base_path / f"fold_{fold_num}"
        results_file = fold_path / f"results_fold{fold_num}.json"
        
        if not results_file.exists():
            print(f"⚠️  Skipping fold {fold_num} - results not found")
            continue
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        all_fold_data.append((fold_num, data))
        
        history = data['training']['history']
        best_epoch = data['training']['best_epoch']
        epochs = range(1, len(history['train_total_loss']) + 1)
        
        # Figure 1: Training Curves (4-panel)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Option E Training Curves - Fold {fold_num}', fontsize=16, fontweight='bold')
        
        # Total Loss
        ax = axes[0, 0]
        ax.plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, history['val_total_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Total Loss', fontweight='bold')
        ax.set_title('Total Multi-Task Loss (Ordinal)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Disease Loss
        ax = axes[0, 1]
        ax.plot(epochs, history['train_disease_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, history['val_disease_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Disease Loss', fontweight='bold')
        ax.set_title('Disease Classification Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Severity Loss (Ordinal)
        ax = axes[1, 0]
        ax.plot(epochs, history['train_severity_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, history['val_severity_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Ordinal Severity Loss', fontweight='bold')
        ax.set_title('Severity Ordinal Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 1]
        ax.plot(epochs, history['lr'], 'g-', linewidth=2, marker='D', markersize=5)
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        output_path = fold_path / 'training_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
        
        # Figure 2: Detailed Metrics (4-panel - only F1 and hierarchical available)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Option E Detailed Metrics - Fold {fold_num}', fontsize=16, fontweight='bold')
        
        # Disease F1
        ax = axes[0, 0]
        ax.plot(epochs, history['train_disease_f1'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, history['val_disease_f1'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Macro F1-Score', fontweight='bold')
        ax.set_title('Disease Macro-F1', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Severity F1
        ax = axes[0, 1]
        ax.plot(epochs, history['train_severity_f1'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, history['val_severity_f1'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Macro F1-Score', fontweight='bold')
        ax.set_title('Severity Macro-F1 (Ordinal)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Hierarchical Accuracy
        ax = axes[1, 0]
        ax.plot(epochs, history['val_hierarchical_acc'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Hierarchical Accuracy', fontweight='bold')
        ax.set_title('Hierarchical Accuracy (Val Only)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Loss Comparison
        ax = axes[1, 1]
        ax.plot(epochs, history['train_disease_loss'], 'b--', label='Train Disease', linewidth=1.5, alpha=0.7)
        ax.plot(epochs, history['train_severity_loss'], 'b:', label='Train Severity', linewidth=1.5, alpha=0.7)
        ax.plot(epochs, history['val_disease_loss'], 'r--', label='Val Disease', linewidth=1.5, alpha=0.7)
        ax.plot(epochs, history['val_severity_loss'], 'r:', label='Val Severity', linewidth=1.5, alpha=0.7)
        ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Disease vs Severity Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = fold_path / 'detailed_training_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    # Generate all-folds comparison
    if all_fold_data:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Option E: 5-Fold Cross-Validation Comparison', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Total Loss
        ax = axes[0, 0]
        for fold_num, data in all_fold_data:
            history = data['training']['history']
            epochs = range(1, len(history['train_total_loss']) + 1)
            ax.plot(epochs, history['val_total_loss'], color=colors[fold_num], 
                    label=f'Fold {fold_num}', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Total Loss', fontweight='bold')
        ax.set_title('Total Loss (All Folds)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Disease F1
        ax = axes[0, 1]
        for fold_num, data in all_fold_data:
            history = data['training']['history']
            epochs = range(1, len(history['train_disease_f1']) + 1)
            ax.plot(epochs, history['val_disease_f1'], color=colors[fold_num], 
                    label=f'Fold {fold_num}', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Disease F1', fontweight='bold')
        ax.set_title('Disease Macro-F1 (All Folds)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Severity F1
        ax = axes[1, 0]
        for fold_num, data in all_fold_data:
            history = data['training']['history']
            epochs = range(1, len(history['train_severity_f1']) + 1)
            ax.plot(epochs, history['val_severity_f1'], color=colors[fold_num], 
                    label=f'Fold {fold_num}', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Severity F1', fontweight='bold')
        ax.set_title('Severity Macro-F1 (All Folds)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Best Epoch Distribution
        ax = axes[1, 1]
        best_epochs = [data['training']['best_epoch'] for _, data in all_fold_data]
        ax.bar(range(len(best_epochs)), best_epochs, color=colors[:len(best_epochs)], alpha=0.7)
        ax.set_xlabel('Fold', fontweight='bold')
        ax.set_ylabel('Best Epoch', fontweight='bold')
        ax.set_title('Best Epoch per Fold', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(best_epochs)))
        ax.set_xticklabels([f'Fold {i}' for i in range(len(best_epochs))])
        ax.grid(True, alpha=0.3, axis='y')
        
        mean_epoch = np.mean(best_epochs)
        ax.axhline(mean_epoch, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_epoch:.1f}')
        ax.legend(loc='best')
        
        plt.tight_layout()
        output_path = base_path / 'all_folds_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

def generate_mobilenetv3_visualizations():
    """Generate all missing visualizations for MobileNetV3"""
    print(f"\n{'='*80}")
    print(f"GENERATING MOBILENETV3 VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    base_path = Path("Results/MobileNetV3 Metrics")
    history_file = base_path / "training_history_fold0.json"
    
    if not history_file.exists():
        print(f"❌ Training history not found: {history_file}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    best_epoch = history.get('best_epoch', len(history['train_loss']))
    epochs = history['epoch']
    
    # Figure 1: Training Curves (3-panel - loss and F1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('MobileNetV3 Training Curves - Lightweight Model', fontsize=16, fontweight='bold')
    
    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Total Loss', fontweight='bold')
    ax.set_title('Total Multi-Task Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Disease F1
    ax = axes[0, 1]
    ax.plot(epochs, history['train_disease_f1'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, history['val_disease_f1'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Macro F1-Score', fontweight='bold')
    ax.set_title('Disease Macro-F1', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Severity F1
    ax = axes[1, 0]
    ax.plot(epochs, history['train_severity_f1'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, history['val_severity_f1'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Macro F1-Score', fontweight='bold')
    ax.set_title('Severity Macro-F1', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2, marker='D', markersize=5)
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate', fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = base_path / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # Figure 2: Detailed Metrics (2-panel - F1 comparison and overfitting)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('MobileNetV3 Detailed Metrics - Lightweight Model', fontsize=16, fontweight='bold')
    
    # F1 Comparison
    ax = axes[0]
    ax.plot(epochs, history['train_disease_f1'], 'b-', label='Train Disease F1', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, history['val_disease_f1'], 'r-', label='Val Disease F1', linewidth=2, marker='s', markersize=3)
    ax.plot(epochs, history['train_severity_f1'], 'b--', label='Train Severity F1', linewidth=1.5, marker='^', markersize=3, alpha=0.7)
    ax.plot(epochs, history['val_severity_f1'], 'r--', label='Val Severity F1', linewidth=1.5, marker='v', markersize=3, alpha=0.7)
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Macro F1-Score', fontweight='bold')
    ax.set_title('Disease vs Severity F1 Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Overfitting Gap
    ax = axes[1]
    gap_loss = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
    gap_disease_f1 = [v - t for v, t in zip(history['val_disease_f1'], history['train_disease_f1'])]
    gap_severity_f1 = [v - t for v, t in zip(history['val_severity_f1'], history['train_severity_f1'])]
    
    ax.plot(epochs, gap_loss, 'purple', label='Loss Gap (Val-Train)', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, gap_disease_f1, 'orange', label='Disease F1 Gap', linewidth=1.5, marker='s', markersize=3, alpha=0.7)
    ax.plot(epochs, gap_severity_f1, 'cyan', label='Severity F1 Gap', linewidth=1.5, marker='^', markersize=3, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Gap (Val - Train)', fontweight='bold')
    ax.set_title('Generalization Gap Analysis', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = base_path / 'detailed_training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"GENERATING ALL MISSING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Generate Option E visualizations
    generate_option_e_visualizations()
    
    # Generate MobileNetV3 visualizations
    generate_mobilenetv3_visualizations()
    
    print(f"\n{'='*80}")
    print(f"✅ ALL MISSING VISUALIZATIONS GENERATED")
    print(f"{'='*80}\n")
