"""Generate 10x10 confusion matrix and classification report table"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "Project" / "Results" / "Option-A Metrics" / "fold_0"

# Load metrics
with open(RESULTS_DIR / 'per_class_metrics.json') as f:
    per_class = json.load(f)

# Get confusion matrix and labels
cm = np.array(per_class['confusion_matrix'])
classes = ['fmd_s1', 'fmd_s2', 'fmd_s3', 'healthy', 'ibk_s1', 'ibk_s2', 'ibk_s3', 'lsd_s1', 'lsd_s2', 'lsd_s3']

print("=" * 80)
print("GENERATING 10-CLASS VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# 1. Create 10x10 confusion matrix
# ============================================================================
print("\n1. Creating 10x10 confusion matrix...")
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.title('10-Class Confusion Matrix (Option A - Flat Classification)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

output_path = RESULTS_DIR / 'confusion_matrix_10x10.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 2. Create classification report table
# ============================================================================
print("\n2. Creating classification report table...")

# Prepare data
data = []
for cls in classes:
    metrics = per_class['per_class'][cls]
    data.append([
        cls,
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['support']
    ])

# Add separator and averages
data.append(['', '', '', '', ''])  
data.append([
    'Macro Avg',
    per_class['macro']['precision'],
    per_class['macro']['recall'],
    per_class['macro']['f1'],
    sum([per_class['per_class'][c]['support'] for c in classes])
])
data.append([
    'Weighted Avg',
    per_class['weighted']['precision'],
    per_class['weighted']['recall'],
    per_class['weighted']['f1'],
    sum([per_class['per_class'][c]['support'] for c in classes])
])
data.append([
    'Micro Avg',
    per_class['micro']['precision'],
    per_class['micro']['recall'],
    per_class['micro']['f1'],
    sum([per_class['per_class'][c]['support'] for c in classes])
])

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('tight')
ax.axis('off')

# Create table
table_data = []
headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']

for row in data:
    if row[0] == '':  # Separator
        table_data.append(['─' * 10, '─' * 10, '─' * 10, '─' * 10, '─' * 10])
    else:
        formatted_row = [
            row[0],
            f'{row[1]:.4f}' if isinstance(row[1], float) else str(row[1]),
            f'{row[2]:.4f}' if isinstance(row[2], float) else str(row[2]),
            f'{row[3]:.4f}' if isinstance(row[3], float) else str(row[3]),
            str(row[4])
        ]
        table_data.append(formatted_row)

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

# Style per-class rows
for i in range(1, 11):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

# Style average rows
for i in range(12, 15):
    for j in range(5):
        table[(i, j)].set_facecolor('#FFF2CC')
        table[(i, j)].set_text_props(weight='bold')

# Add overall accuracy
total_samples = sum([per_class['per_class'][c]['support'] for c in classes])
correct = int(per_class['accuracy'] * total_samples)
accuracy_text = f'Overall Accuracy: {per_class["accuracy"]:.4f} ({correct}/{total_samples} samples)'
plt.figtext(0.5, 0.05, accuracy_text, ha='center', fontsize=12, weight='bold', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.title('10-Class Classification Report (Option A)', fontsize=16, weight='bold', pad=20)

output_path = RESULTS_DIR / 'classification_report_table.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {output_path}")
plt.close()

# ============================================================================
# 3. Save text-based classification report
# ============================================================================
print("\n3. Creating text classification report...")

with open(RESULTS_DIR / 'classification_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("10-CLASS CLASSIFICATION REPORT (OPTION A)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
    f.write("-" * 80 + "\n")
    
    for cls in classes:
        metrics = per_class['per_class'][cls]
        f.write(f"{cls:<12} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                f"{metrics['f1']:<12.4f} {metrics['support']:<10}\n")
    
    f.write("-" * 80 + "\n")
    f.write(f"{'Macro Avg':<12} {per_class['macro']['precision']:<12.4f} "
            f"{per_class['macro']['recall']:<12.4f} {per_class['macro']['f1']:<12.4f} "
            f"{total_samples:<10}\n")
    f.write(f"{'Weighted Avg':<12} {per_class['weighted']['precision']:<12.4f} "
            f"{per_class['weighted']['recall']:<12.4f} {per_class['weighted']['f1']:<12.4f} "
            f"{total_samples:<10}\n")
    f.write(f"{'Micro Avg':<12} {per_class['micro']['precision']:<12.4f} "
            f"{per_class['micro']['recall']:<12.4f} {per_class['micro']['f1']:<12.4f} "
            f"{total_samples:<10}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"Overall Accuracy: {per_class['accuracy']:.4f} ({correct}/{total_samples} samples)\n")
    f.write("=" * 80 + "\n")

print(f"   ✓ Saved: {RESULTS_DIR / 'classification_report.txt'}")

print("\n" + "=" * 80)
print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\nFiles created in: {RESULTS_DIR}")
print("  1. confusion_matrix_10x10.png")
print("  2. classification_report_table.png")
print("  3. classification_report.txt")
