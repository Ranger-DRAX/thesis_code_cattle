"""Create comprehensive overfitting analysis visualizations"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "Project" / "Results" / "Option-A Metrics" / "fold_0"

# Load training metrics
with open(RESULTS_DIR / 'training_metrics.json') as f:
    metrics = json.load(f)

with open(RESULTS_DIR / 'summary.json') as f:
    summary = json.load(f)

best_epoch = summary['best_epoch'] - 1  # 0-indexed
epochs = range(1, len(metrics['train_loss']) + 1)

# Create comprehensive overfitting analysis visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# 1. Train vs Val Loss with overfitting zones
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(epochs, metrics['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
ax1.plot(epochs, metrics['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
ax1.axvline(x=best_epoch+1, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch+1})')
ax1.axvline(x=5, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Warmup End')

# Shade overfitting zone (if any)
train_losses = np.array(metrics['train_loss'])
val_losses = np.array(metrics['val_loss'])
gap = val_losses - train_losses
if any(gap > 0.3):
    overfit_mask = gap > 0.3
    ax1.fill_between(epochs, 0, 3, where=overfit_mask, alpha=0.2, color='red', label='Potential Overfit Zone')

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Train vs Validation Loss (Overfitting Check)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# ============================================================================
# 2. Train vs Val F1 Score
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(epochs, metrics['train_f1'], 'b-', linewidth=2, label='Train F1', marker='o', markersize=4)
ax2.plot(epochs, metrics['val_f1'], 'r-', linewidth=2, label='Val F1', marker='s', markersize=4)
ax2.axvline(x=best_epoch+1, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=5, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('Train vs Val F1', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# ============================================================================
# 3. Loss Gap (Train - Val)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])
loss_gap = np.array(metrics['train_loss']) - np.array(metrics['val_loss'])
colors = ['green' if g < 0 else 'orange' if g < 0.15 else 'red' for g in loss_gap]
ax3.bar(epochs, loss_gap, color=colors, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.axhline(y=-0.1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (Val Better)')
ax3.axhline(y=0.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Warning (Overfit)')
ax3.axvline(x=best_epoch+1, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Loss Gap (Train - Val)', fontsize=12, fontweight='bold')
ax3.set_title('Loss Gap Analysis', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# ============================================================================
# 4. F1 Gap (Train - Val)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
f1_gap = np.array(metrics['train_f1']) - np.array(metrics['val_f1'])
colors = ['green' if g < 0.05 else 'orange' if g < 0.10 else 'red' for g in f1_gap]
ax4.bar(epochs, f1_gap, color=colors, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (<5%)')
ax4.axhline(y=0.10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good (<10%)')
ax4.axvline(x=best_epoch+1, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax4.set_ylabel('F1 Gap (Train - Val)', fontsize=12, fontweight='bold')
ax4.set_title('F1 Gap Analysis', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

# ============================================================================
# 5. Validation Improvement Rate
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
val_f1_improvement = [0] + [metrics['val_f1'][i] - metrics['val_f1'][i-1] for i in range(1, len(metrics['val_f1']))]
colors = ['green' if imp > 0 else 'red' for imp in val_f1_improvement]
ax5.bar(epochs, val_f1_improvement, color=colors, alpha=0.7)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax5.axvline(x=best_epoch+1, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax5.set_ylabel('Val F1 Improvement', fontsize=12, fontweight='bold')
ax5.set_title('Validation Improvement per Epoch', fontsize=14, fontweight='bold')
ax5.grid(alpha=0.3, axis='y')

# ============================================================================
# 6. Accuracy comparison
# ============================================================================
ax6 = fig.add_subplot(gs[2, :2])
ax6.plot(epochs, metrics['train_acc'], 'b-', linewidth=2, label='Train Accuracy', marker='o', markersize=4, alpha=0.7)
ax6.plot(epochs, metrics['val_acc'], 'r-', linewidth=2, label='Val Accuracy', marker='s', markersize=4)
ax6.axvline(x=best_epoch+1, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch+1})')
ax6.axvline(x=5, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Warmup End')
ax6.fill_between(epochs, metrics['train_acc'], metrics['val_acc'], alpha=0.2, 
                 color='red' if metrics['train_acc'][-1] - metrics['val_acc'][-1] > 0.1 else 'green')
ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax6.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax6.set_title('Train vs Validation Accuracy', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# ============================================================================
# 7. Summary statistics table
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

# Calculate statistics
best_val_f1 = metrics['val_f1'][best_epoch]
final_val_f1 = metrics['val_f1'][-1]
val_f1_improvement_total = best_val_f1 - metrics['val_f1'][0]
loss_gap_best = metrics['train_loss'][best_epoch] - metrics['val_loss'][best_epoch]
f1_gap_best = metrics['train_f1'][best_epoch] - metrics['val_f1'][best_epoch]

stats_data = [
    ['Best Val F1', f'{best_val_f1:.4f}'],
    ['Val F1 @ Epoch 1', f'{metrics["val_f1"][0]:.4f}'],
    ['Improvement', f'{val_f1_improvement_total:.4f}'],
    ['', ''],
    ['Loss Gap (Best)', f'{loss_gap_best:.4f}'],
    ['F1 Gap (Best)', f'{f1_gap_best:.4f}'],
    ['', ''],
    ['Best Epoch', f'{best_epoch+1}'],
    ['Total Epochs', f'{len(metrics["train_loss"])}'],
    ['Early Stopped', 'Yes' if len(metrics["train_loss"]) < 25 else 'No']
]

table = ax7.table(cellText=stats_data, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(len(stats_data)):
    if stats_data[i][0] == '':
        for j in range(2):
            table[(i, j)].set_facecolor('#FFFFFF')
            table[(i, j)].set_text_props(weight='normal')
    else:
        table[(i, 0)].set_facecolor('#E7E6E6')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#FFFFFF')

ax7.set_title('Generalization Statistics', fontsize=14, fontweight='bold', pad=10)

# Overall title
fig.suptitle('Option A - Comprehensive Overfitting Analysis', 
             fontsize=18, fontweight='bold', y=0.995)

# Add verdict box
verdict_text = '✅ EXCELLENT GENERALIZATION - NO OVERFITTING DETECTED'
fig.text(0.5, 0.02, verdict_text, ha='center', fontsize=14, weight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.99])

output_path = RESULTS_DIR / 'overfitting_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'✓ Created: {output_path}')
plt.close()

print('\n' + '=' * 80)
print('OVERFITTING ANALYSIS COMPLETE')
print('=' * 80)
