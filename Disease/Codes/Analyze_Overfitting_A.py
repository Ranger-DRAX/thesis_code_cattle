"""Analyze Option A for overfitting"""
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "Project" / "Results" / "Option-A Metrics" / "fold_0"

# Load training metrics
with open(RESULTS_DIR / 'training_metrics.json') as f:
    metrics = json.load(f)

# Load summary
with open(RESULTS_DIR / 'summary.json') as f:
    summary = json.load(f)

print('=' * 80)
print('OPTION A - OVERFITTING ANALYSIS')
print('=' * 80)

# Best epoch info
best_epoch = summary['best_epoch']
total_epochs = len(metrics['train_loss'])

print(f'\n📊 TRAINING SUMMARY:')
print(f'  Total epochs trained: {total_epochs}')
print(f'  Best epoch (by val F1): {best_epoch}')
print(f'  Early stopping patience: 5 epochs')
print(f'  Stopped at epoch: {total_epochs} (after {total_epochs - best_epoch} epochs of no improvement)')

# Get metrics at different stages
print(f'\n\n📈 METRICS PROGRESSION:')
print(f'{"Stage":<20} {"Train Loss":<12} {"Val Loss":<12} {"Gap":<12} {"Train F1":<12} {"Val F1":<12} {"Gap":<12}')
print('-' * 100)

# Epoch 1 (initial)
idx = 0
train_loss, val_loss = metrics['train_loss'][idx], metrics['val_loss'][idx]
train_f1, val_f1 = metrics['train_f1'][idx], metrics['val_f1'][idx]
print(f'{"Epoch 1 (Initial)":<20} {train_loss:<12.4f} {val_loss:<12.4f} {train_loss-val_loss:<12.4f} {train_f1:<12.4f} {val_f1:<12.4f} {train_f1-val_f1:<12.4f}')

# Epoch 5 (end of warmup)
idx = 4
train_loss, val_loss = metrics['train_loss'][idx], metrics['val_loss'][idx]
train_f1, val_f1 = metrics['train_f1'][idx], metrics['val_f1'][idx]
print(f'{"Epoch 5 (Warmup End)":<20} {train_loss:<12.4f} {val_loss:<12.4f} {train_loss-val_loss:<12.4f} {train_f1:<12.4f} {val_f1:<12.4f} {train_f1-val_f1:<12.4f}')

# Best epoch
idx = best_epoch - 1
train_loss, val_loss = metrics['train_loss'][idx], metrics['val_loss'][idx]
train_f1, val_f1 = metrics['train_f1'][idx], metrics['val_f1'][idx]
print(f'{f"Epoch {best_epoch} (Best)":<20} {train_loss:<12.4f} {val_loss:<12.4f} {train_loss-val_loss:<12.4f} {train_f1:<12.4f} {val_f1:<12.4f} {train_f1-val_f1:<12.4f}')

# Final epoch
idx = -1
train_loss, val_loss = metrics['train_loss'][idx], metrics['val_loss'][idx]
train_f1, val_f1 = metrics['train_f1'][idx], metrics['val_f1'][idx]
print(f'{f"Epoch {total_epochs} (Final)":<20} {train_loss:<12.4f} {val_loss:<12.4f} {train_loss-val_loss:<12.4f} {train_f1:<12.4f} {val_f1:<12.4f} {train_f1-val_f1:<12.4f}')

# Calculate overfitting indicators
train_losses = np.array(metrics['train_loss'])
val_losses = np.array(metrics['val_loss'])
train_f1s = np.array(metrics['train_f1'])
val_f1s = np.array(metrics['val_f1'])

# Best epoch metrics
best_idx = best_epoch - 1
loss_gap_best = train_losses[best_idx] - val_losses[best_idx]
f1_gap_best = train_f1s[best_idx] - val_f1s[best_idx]

# Final epoch metrics
loss_gap_final = train_losses[-1] - val_losses[-1]
f1_gap_final = train_f1s[-1] - val_f1s[-1]

print(f'\n\n🎯 OVERFITTING INDICATORS:')
print(f'  Loss gap at best epoch: {loss_gap_best:.4f} (negative = validation better)')
print(f'  F1 gap at best epoch: {f1_gap_best:.4f} (positive = train better)')
print(f'  Loss gap at final epoch: {loss_gap_final:.4f}')
print(f'  F1 gap at final epoch: {f1_gap_final:.4f}')

# Check if validation improved after best
val_improved_after_best = any(val_f1s[best_idx:] > val_f1s[best_idx])
print(f'\n  Val F1 improved after best epoch: {val_improved_after_best}')

# Overfitting diagnosis
print(f'\n\n✅ GENERALIZATION ASSESSMENT:')

issues = []
good_signs = []

# Check 1: Train-Val gap at best epoch
if abs(f1_gap_best) < 0.05:
    good_signs.append('Train-Val F1 gap is very small (<5%) at best epoch')
elif abs(f1_gap_best) < 0.10:
    good_signs.append('Train-Val F1 gap is reasonable (<10%) at best epoch')
else:
    issues.append(f'Large Train-Val F1 gap ({f1_gap_best:.2%}) at best epoch')

# Check 2: Validation trend
if val_losses[-1] < val_losses[best_idx] + 0.1:
    good_signs.append('Validation loss remained stable after best epoch')
else:
    issues.append('Validation loss degraded significantly after best epoch')

# Check 3: Early stopping worked
if total_epochs < summary['config']['max_epochs']:
    good_signs.append('Early stopping triggered (prevented overtraining)')
else:
    issues.append('Trained to max epochs (might need more patience)')

# Check 4: Validation improvement over time
val_improvement = val_f1s[best_idx] - val_f1s[0]
if val_improvement > 0.25:
    good_signs.append(f'Strong validation improvement: {val_improvement:.2%}')
elif val_improvement > 0.15:
    good_signs.append(f'Good validation improvement: {val_improvement:.2%}')
else:
    issues.append(f'Limited validation improvement: {val_improvement:.2%}')

# Check 5: Loss behavior
if val_losses[best_idx] <= train_losses[best_idx] + 0.2:
    good_signs.append('Validation loss close to train loss (good generalization)')

print(f'\n✓ POSITIVE INDICATORS ({len(good_signs)}):')
for sign in good_signs:
    print(f'  • {sign}')

if issues:
    print(f'\n⚠ POTENTIAL CONCERNS ({len(issues)}):')
    for issue in issues:
        print(f'  • {issue}')
else:
    print(f'\n⚠ No significant concerns detected')

# Final verdict
print(f'\n\n🏆 VERDICT:')
if len(issues) == 0:
    print('  ✅ Model shows EXCELLENT generalization')
    print('  ✅ No signs of overfitting')
elif len(issues) <= 1:
    print('  ✅ Model shows GOOD generalization')
    print('  ⚠ Minor concerns, but overall well-generalized')
else:
    print('  ⚠ Model shows signs of overfitting')
    print('  📝 Recommendations: Increase dropout, add more augmentation, reduce capacity')

print(f'\n' + '=' * 80)
