"""
Create comparison table of all model results
"""
import json
import pandas as pd
from pathlib import Path

FINAL_DIR = Path(r"E:\Disease Classification\Final")

# Collect all metrics
results = []

for metrics_file in FINAL_DIR.rglob("metrics.json"):
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    results.append({
        'Option': data['option'],
        'Backbone': data['backbone'],
        'Disease_Accuracy': f"{data['disease']['accuracy']:.4f}",
        'Disease_Precision': f"{data['disease']['precision_macro']:.4f}",
        'Disease_Recall': f"{data['disease']['recall_macro']:.4f}",
        'Disease_F1': f"{data['disease']['f1_macro']:.4f}",
        'Severity_Accuracy': f"{data['severity']['accuracy']:.4f}",
        'Severity_F1': f"{data['severity']['f1_macro']:.4f}",
        'Hierarchical_Accuracy': f"{data['hierarchical']['accuracy']:.4f}",
        'Hierarchical_F1': f"{data['hierarchical']['f1_macro']:.4f}",
        # Raw values for sorting
        '_disease_f1': data['disease']['f1_macro'],
        '_hier_acc': data['hierarchical']['accuracy']
    })

# Create DataFrame and sort by Disease F1 (descending)
df = pd.DataFrame(results)
df = df.sort_values('_disease_f1', ascending=False)

# Drop sorting columns
df_display = df.drop(columns=['_disease_f1', '_hier_acc'])

print("\n" + "="*120)
print("COMPREHENSIVE MODEL COMPARISON - ALL OPTIONS AND BACKBONES")
print("="*120)
print(df_display.to_string(index=False))
print("="*120)

# Find best models
print("\n" + "="*120)
print("BEST MODELS")
print("="*120)

best_disease_f1 = df.iloc[0]
best_hier_acc = df.sort_values('_hier_acc', ascending=False).iloc[0]

print(f"\n🏆 BEST DISEASE CLASSIFICATION (F1-Score):")
print(f"   Option: {best_disease_f1['Option']}")
print(f"   Backbone: {best_disease_f1['Backbone']}")
print(f"   Disease F1: {best_disease_f1['Disease_F1']}")
print(f"   Disease Accuracy: {best_disease_f1['Disease_Accuracy']}")
print(f"   Hierarchical Accuracy: {best_disease_f1['Hierarchical_Accuracy']}")

print(f"\n🏆 BEST HIERARCHICAL CLASSIFICATION (Accuracy):")
print(f"   Option: {best_hier_acc['Option']}")
print(f"   Backbone: {best_hier_acc['Backbone']}")
print(f"   Hierarchical Accuracy: {best_hier_acc['Hierarchical_Accuracy']}")
print(f"   Hierarchical F1: {best_hier_acc['Hierarchical_F1']}")
print(f"   Disease F1: {best_hier_acc['Disease_F1']}")

print("\n" + "="*120)

# Save to file
output_file = FINAL_DIR / "COMPARISON_TABLE.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*120 + "\n")
    f.write("COMPREHENSIVE MODEL COMPARISON - ALL OPTIONS AND BACKBONES\n")
    f.write("="*120 + "\n")
    f.write(df_display.to_string(index=False) + "\n")
    f.write("="*120 + "\n\n")
    
    f.write("="*120 + "\n")
    f.write("BEST MODELS\n")
    f.write("="*120 + "\n\n")
    
    f.write(f"🏆 BEST DISEASE CLASSIFICATION (F1-Score):\n")
    f.write(f"   Option: {best_disease_f1['Option']}\n")
    f.write(f"   Backbone: {best_disease_f1['Backbone']}\n")
    f.write(f"   Disease F1: {best_disease_f1['Disease_F1']}\n")
    f.write(f"   Disease Accuracy: {best_disease_f1['Disease_Accuracy']}\n")
    f.write(f"   Hierarchical Accuracy: {best_disease_f1['Hierarchical_Accuracy']}\n\n")
    
    f.write(f"🏆 BEST HIERARCHICAL CLASSIFICATION (Accuracy):\n")
    f.write(f"   Option: {best_hier_acc['Option']}\n")
    f.write(f"   Backbone: {best_hier_acc['Backbone']}\n")
    f.write(f"   Hierarchical Accuracy: {best_hier_acc['Hierarchical_Accuracy']}\n")
    f.write(f"   Hierarchical F1: {best_hier_acc['Hierarchical_F1']}\n")
    f.write(f"   Disease F1: {best_hier_acc['Disease_F1']}\n")
    f.write("\n" + "="*120 + "\n")

print(f"\n✓ Comparison table saved to: {output_file}")
