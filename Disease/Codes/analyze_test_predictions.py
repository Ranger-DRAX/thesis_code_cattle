"""
Analyze Test Set Predictions - Compare all options
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

PROJECT_ROOT = Path(r"E:\Disease Classification\Project")
CSV_PATH = PROJECT_ROOT / "test_set_check.csv"
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Test_Set_Analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)

print("=" * 80)
print("TEST SET PREDICTIONS ANALYSIS")
print("=" * 80)

# Calculate metrics for each option
options = ['a', 'b', 'c', 'd', 'e']
results_summary = []

for option in options:
    pred_col = f'predicted_option_{option}'
    
    # Overall accuracy
    accuracy = accuracy_score(df['actual_class'], df[pred_col])
    
    # Per-class accuracy
    correct_per_class = df[df['actual_class'] == df[pred_col]].groupby('actual_class').size()
    total_per_class = df.groupby('actual_class').size()
    class_accuracy = (correct_per_class / total_per_class).fillna(0)
    
    results_summary.append({
        'Option': option.upper(),
        'Accuracy': f"{accuracy*100:.2f}%",
        'Correct': (df['actual_class'] == df[pred_col]).sum(),
        'Total': len(df),
        'Errors': (df['actual_class'] != df[pred_col]).sum()
    })

# Print summary table
summary_df = pd.DataFrame(results_summary)
print("\n📊 ACCURACY SUMMARY:")
print(summary_df.to_string(index=False))

# Find images where all options agree but are wrong
all_wrong = df[
    (df['predicted_option_a'] == df['predicted_option_b']) &
    (df['predicted_option_b'] == df['predicted_option_c']) &
    (df['predicted_option_c'] == df['predicted_option_d']) &
    (df['predicted_option_d'] == df['predicted_option_e']) &
    (df['predicted_option_a'] != df['actual_class'])
]

print(f"\n🔴 Images where ALL options agree but are WRONG: {len(all_wrong)}")
if len(all_wrong) > 0:
    print("\nSample hard cases (all models misclassify):")
    print(all_wrong[['filepath', 'actual_class', 'predicted_option_a']].head(10).to_string(index=False))

# Find images where all options are correct
all_correct = df[
    (df['predicted_option_a'] == df['actual_class']) &
    (df['predicted_option_b'] == df['actual_class']) &
    (df['predicted_option_c'] == df['actual_class']) &
    (df['predicted_option_d'] == df['actual_class']) &
    (df['predicted_option_e'] == df['actual_class'])
]

print(f"\n🟢 Images where ALL options are CORRECT: {len(all_correct)} ({len(all_correct)/len(df)*100:.1f}%)")

# Find images where only best option (E) is correct but others are wrong
only_e_correct = df[
    (df['predicted_option_e'] == df['actual_class']) &
    ((df['predicted_option_a'] != df['actual_class']) |
     (df['predicted_option_b'] != df['actual_class']) |
     (df['predicted_option_c'] != df['actual_class']) |
     (df['predicted_option_d'] != df['actual_class']))
]

print(f"\n⭐ Images where Option E is correct but at least one other option is wrong: {len(only_e_correct)}")

# Per-class performance comparison
print(f"\n📈 PER-CLASS ACCURACY COMPARISON:")
class_comparison = []

for cls in sorted(df['actual_class'].unique()):
    cls_df = df[df['actual_class'] == cls]
    row = {'Class': cls, 'Count': len(cls_df)}
    
    for option in options:
        pred_col = f'predicted_option_{option}'
        acc = (cls_df['actual_class'] == cls_df[pred_col]).mean() * 100
        row[f'Option {option.upper()}'] = f"{acc:.1f}%"
    
    class_comparison.append(row)

class_comp_df = pd.DataFrame(class_comparison)
print(class_comp_df.to_string(index=False))

# Visualization: Accuracy comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Overall accuracy bar chart
ax = axes[0, 0]
accuracies = [accuracy_score(df['actual_class'], df[f'predicted_option_{opt}']) * 100 
              for opt in options]
bars = ax.bar([opt.upper() for opt in options], accuracies, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Option', fontsize=12, fontweight='bold')
ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([75, 95])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

# 2. Error counts
ax = axes[0, 1]
errors = [(df['actual_class'] != df[f'predicted_option_{opt}']).sum() 
          for opt in options]
bars = ax.bar([opt.upper() for opt in options], errors, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
ax.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
ax.set_xlabel('Option', fontsize=12, fontweight='bold')
ax.set_title('Total Misclassifications', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 3. Per-class accuracy heatmap
ax = axes[1, 0]
class_accuracies = []
class_names = sorted(df['actual_class'].unique())

for cls in class_names:
    cls_df = df[df['actual_class'] == cls]
    accs = [(cls_df['actual_class'] == cls_df[f'predicted_option_{opt}']).mean() * 100 
            for opt in options]
    class_accuracies.append(accs)

sns.heatmap(class_accuracies, annot=True, fmt='.1f', cmap='RdYlGn', 
            xticklabels=[opt.upper() for opt in options],
            yticklabels=class_names, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
ax.set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Option', fontsize=12, fontweight='bold')
ax.set_ylabel('Class', fontsize=12, fontweight='bold')

# 4. Agreement analysis
ax = axes[1, 1]
agreement_data = {
    'All Correct': len(all_correct),
    'All Wrong': len(all_wrong),
    'Mixed': len(df) - len(all_correct) - len(all_wrong)
}

colors_pie = ['#98D8C8', '#FF6B6B', '#FFA07A']
wedges, texts, autotexts = ax.pie(agreement_data.values(), labels=agreement_data.keys(), 
                                   autopct='%1.1f%%', startangle=90, colors=colors_pie)
ax.set_title('Model Agreement on Predictions', fontsize=14, fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'test_set_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {OUTPUT_DIR / 'test_set_comparison.png'}")
plt.close()

# Save detailed comparison report
report_path = OUTPUT_DIR / 'detailed_comparison.txt'
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DETAILED TEST SET PREDICTIONS COMPARISON\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("OVERALL SUMMARY:\n")
    f.write(summary_df.to_string(index=False) + "\n\n")
    
    f.write("PER-CLASS ACCURACY:\n")
    f.write(class_comp_df.to_string(index=False) + "\n\n")
    
    f.write(f"All models correct: {len(all_correct)} ({len(all_correct)/len(df)*100:.2f}%)\n")
    f.write(f"All models wrong: {len(all_wrong)} ({len(all_wrong)/len(df)*100:.2f}%)\n")
    f.write(f"Mixed results: {len(df) - len(all_correct) - len(all_wrong)} ({(len(df) - len(all_correct) - len(all_wrong))/len(df)*100:.2f}%)\n")

print(f"✅ Detailed report saved to: {report_path}")
print("\n" + "=" * 80)
