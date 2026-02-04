"""
Generate 5-Fold CV Summary Table for All Options (A-E)
=======================================================
Creates comprehensive comparison table with:
- Disease 4-class metrics (Accuracy, Precision, Recall, Macro F1)
- Severity 3-class metrics (Accuracy, Precision, Recall, Macro F1)
- Hierarchical 10-class metrics (Accuracy, Precision, Recall, Macro F1)
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "Results"

# Load calculated metrics
def load_calculated_metrics():
    """Load calculated missing metrics"""
    calc_path = RESULTS_DIR / "calculated_missing_metrics.json"
    if calc_path.exists():
        with open(calc_path) as f:
            return json.load(f)
    return None

CALCULATED_METRICS = load_calculated_metrics()

def load_option_a_metrics():
    """Load Option-A CV results"""
    with open(RESULTS_DIR / "Option-A Metrics" / "5fold_cv" / "cv_results.json") as f:
        data = json.load(f)
    
    stats = data['statistics']
    
    # Use calculated metrics
    calc = CALCULATED_METRICS['option_a'] if CALCULATED_METRICS else None
    
    return {
        'disease': {
            'accuracy': (stats['disease_accuracy']['mean'], stats['disease_accuracy']['std']),
            'precision': (calc['disease']['precision_mean'], calc['disease']['precision_std']) if calc else (None, None),
            'recall': (calc['disease']['recall_mean'], calc['disease']['recall_std']) if calc else (None, None),
            'f1': (stats['disease_macro_f1']['mean'], stats['disease_macro_f1']['std'])
        },
        'severity': {
            'accuracy': (stats['severity_accuracy']['mean'], stats['severity_accuracy']['std']),
            'precision': (calc['severity']['precision_mean'], calc['severity']['precision_std']) if calc else (None, None),
            'recall': (calc['severity']['recall_mean'], calc['severity']['recall_std']) if calc else (None, None),
            'f1': (stats['severity_macro_f1']['mean'], stats['severity_macro_f1']['std'])
        },
        'hierarchical': {
            'accuracy': (stats['hierarchical_accuracy']['mean'], stats['hierarchical_accuracy']['std']),
            'precision': (calc['hierarchical']['precision_mean'], calc['hierarchical']['precision_std']) if calc else (None, None),
            'recall': (calc['hierarchical']['recall_mean'], calc['hierarchical']['recall_std']) if calc else (None, None),
            'f1': (stats['flat_10class_f1']['mean'], stats['flat_10class_f1']['std'])
        }
    }


def load_option_b_metrics():
    """Load Option-B CV results"""
    with open(RESULTS_DIR / "Option-B Metrics" / "5fold_cv" / "cv_results.json") as f:
        data = json.load(f)
    
    stats = data['statistics']
    
    # Use calculated hierarchical metrics
    calc_hier = CALCULATED_METRICS['option_b_hierarchical'] if CALCULATED_METRICS else None
    
    return {
        'disease': {
            'accuracy': (stats['disease_accuracy']['mean'], stats['disease_accuracy']['std']),
            'precision': (stats['disease_precision']['mean'], stats['disease_precision']['std']),
            'recall': (stats['disease_recall']['mean'], stats['disease_recall']['std']),
            'f1': (stats['disease_f1']['mean'], stats['disease_f1']['std'])
        },
        'severity': {
            'accuracy': (stats['severity_accuracy']['mean'], stats['severity_accuracy']['std']),
            'precision': (stats['severity_precision']['mean'], stats['severity_precision']['std']),
            'recall': (stats['severity_recall']['mean'], stats['severity_recall']['std']),
            'f1': (stats['severity_f1']['mean'], stats['severity_f1']['std'])
        },
        'hierarchical': {
            'accuracy': (stats['hierarchical_accuracy']['mean'], stats['hierarchical_accuracy']['std']),
            'precision': (calc_hier['precision_mean'], calc_hier['precision_std']) if calc_hier else (None, None),
            'recall': (calc_hier['recall_mean'], calc_hier['recall_std']) if calc_hier else (None, None),
            'f1': (calc_hier['f1_mean'], calc_hier['f1_std']) if calc_hier else (None, None)
        }
    }


def load_option_c_metrics():
    """Load Option-C CV results"""
    with open(RESULTS_DIR / "Option-C Metrics" / "5fold_cv" / "cv_results.json") as f:
        data = json.load(f)
    
    stats = data['statistics']
    
    # Use calculated metrics
    calc = CALCULATED_METRICS['option_c'] if CALCULATED_METRICS else None
    calc_hier = CALCULATED_METRICS['option_c_hierarchical'] if CALCULATED_METRICS else None
    
    return {
        'disease': {
            'accuracy': (stats['disease_accuracy']['mean'], stats['disease_accuracy']['std']),
            'precision': (calc['disease']['precision_mean'], calc['disease']['precision_std']) if calc else (None, None),
            'recall': (calc['disease']['recall_mean'], calc['disease']['recall_std']) if calc else (None, None),
            'f1': (stats['disease_f1']['mean'], stats['disease_f1']['std'])
        },
        'severity': {
            'accuracy': (stats['severity_accuracy']['mean'], stats['severity_accuracy']['std']),
            'precision': (calc['severity']['precision_mean'], calc['severity']['precision_std']) if calc else (None, None),
            'recall': (calc['severity']['recall_mean'], calc['severity']['recall_std']) if calc else (None, None),
            'f1': (stats['severity_f1']['mean'], stats['severity_f1']['std'])
        },
        'hierarchical': {
            'accuracy': (stats['hierarchical_accuracy']['mean'], stats['hierarchical_accuracy']['std']),
            'precision': (calc_hier['precision_mean'], calc_hier['precision_std']) if calc_hier else (None, None),
            'recall': (calc_hier['recall_mean'], calc_hier['recall_std']) if calc_hier else (None, None),
            'f1': (calc_hier['f1_mean'], calc_hier['f1_std']) if calc_hier else (None, None)
        }
    }


def load_option_d_metrics():
    """
    Load Option-D metrics
    Option-D uses Option-C's model, so we use Option-C's CV results
    """
    # Option-D uses Option-C's trained model, so CV metrics are same as Option-C
    return load_option_c_metrics()


def load_option_e_metrics():
    """Load Option-E CV results"""
    with open(RESULTS_DIR / "Option-E Metrics" / "5fold_cv" / "cv_results.json") as f:
        data = json.load(f)
    
    folds = data['folds']
    stats = data['statistics']
    
    # Calculate precision and recall means/stds from folds
    disease_precision = [f['disease']['precision_macro'] for f in folds]
    disease_recall = [f['disease']['recall_macro'] for f in folds]
    disease_accuracy = [f['disease']['accuracy'] for f in folds]
    
    severity_precision = [f['severity']['precision_macro'] for f in folds]
    severity_recall = [f['severity']['recall_macro'] for f in folds]
    severity_accuracy = [f['severity']['accuracy'] for f in folds]
    
    # Use calculated hierarchical metrics
    calc_hier = CALCULATED_METRICS['option_e_hierarchical'] if CALCULATED_METRICS else None
    
    return {
        'disease': {
            'accuracy': (np.mean(disease_accuracy), np.std(disease_accuracy)),
            'precision': (np.mean(disease_precision), np.std(disease_precision)),
            'recall': (np.mean(disease_recall), np.std(disease_recall)),
            'f1': (stats['disease_f1']['mean'], stats['disease_f1']['std'])
        },
        'severity': {
            'accuracy': (np.mean(severity_accuracy), np.std(severity_accuracy)),
            'precision': (np.mean(severity_precision), np.std(severity_precision)),
            'recall': (np.mean(severity_recall), np.std(severity_recall)),
            'f1': (stats['severity_f1']['mean'], stats['severity_f1']['std'])
        },
        'hierarchical': {
            'accuracy': (stats['hierarchical_accuracy']['mean'], stats['hierarchical_accuracy']['std']),
            'precision': (calc_hier['precision_mean'], calc_hier['precision_std']) if calc_hier else (None, None),
            'recall': (calc_hier['recall_mean'], calc_hier['recall_std']) if calc_hier else (None, None),
            'f1': (calc_hier['f1_mean'], calc_hier['f1_std']) if calc_hier else (None, None)
        }
    }


def recalculate_option_e_accuracies():
    """Recalculate accuracy statistics for Option-E from all folds"""
    with open(RESULTS_DIR / "Option-E Metrics" / "5fold_cv" / "cv_results.json") as f:
        data = json.load(f)
    
    folds = data['folds']
    
    disease_acc = [f['disease']['accuracy'] for f in folds]
    severity_acc = [f['severity']['accuracy'] for f in folds]
    
    return {
        'disease_accuracy': (np.mean(disease_acc), np.std(disease_acc)),
        'severity_accuracy': (np.mean(severity_acc), np.std(severity_acc))
    }


def format_metric(mean, std):
    """Format metric as 'mean ± std' with percentages"""
    if mean is None or std is None:
        return "—"
    return f"{mean*100:.2f} ± {std*100:.2f}"


def generate_table():
    """Generate comprehensive CV summary table"""
    
    print("=" * 100)
    print("LOADING 5-FOLD CV RESULTS")
    print("=" * 100)
    
    # Load all metrics
    option_a = load_option_a_metrics()
    option_b = load_option_b_metrics()
    option_c = load_option_c_metrics()
    option_d = load_option_d_metrics()
    option_e = load_option_e_metrics()
    
    print("\n* Loaded all CV results (including calculated metrics)")
    
    # Create table data
    table_data = []
    
    for option_name, metrics in [
        ('Option-A', option_a),
        ('Option-B', option_b),
        ('Option-C', option_c),
        ('Option-D', option_d),
        ('Option-E', option_e)
    ]:
        row = {
            'Option': option_name,
            # Disease 4-class
            'Disease Accuracy': format_metric(*metrics['disease']['accuracy']),
            'Disease Precision': format_metric(*metrics['disease']['precision']),
            'Disease Recall': format_metric(*metrics['disease']['recall']),
            'Disease F1': format_metric(*metrics['disease']['f1']),
            # Severity 3-class
            'Severity Accuracy': format_metric(*metrics['severity']['accuracy']),
            'Severity Precision': format_metric(*metrics['severity']['precision']),
            'Severity Recall': format_metric(*metrics['severity']['recall']),
            'Severity F1': format_metric(*metrics['severity']['f1']),
            # Hierarchical 10-class
            'Hierarchical Accuracy': format_metric(*metrics['hierarchical']['accuracy']),
            'Hierarchical Precision': format_metric(*metrics['hierarchical']['precision']),
            'Hierarchical Recall': format_metric(*metrics['hierarchical']['recall']),
            'Hierarchical F1': format_metric(*metrics['hierarchical']['f1'])
        }
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Print tables
    print("\n" + "=" * 100)
    print("5-FOLD CROSS-VALIDATION SUMMARY (Mean ± Std %)")
    print("=" * 100)
    
    # Full table
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    csv_path = PROJECT_DIR / "Results" / "5fold_cv_summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n* Saved CSV: {csv_path}")
    
    # Create separate tables for each classification task
    print("\n" + "=" * 100)
    print("DISEASE CLASSIFICATION (4-class)")
    print("=" * 100)
    disease_df = df[['Option', 'Disease Accuracy', 'Disease Precision', 'Disease Recall', 'Disease F1']]
    print("\n" + disease_df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("SEVERITY CLASSIFICATION (3-class, Diseased Only)")
    print("=" * 100)
    severity_df = df[['Option', 'Severity Accuracy', 'Severity Precision', 'Severity Recall', 'Severity F1']]
    print("\n" + severity_df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("HIERARCHICAL CLASSIFICATION (10-class)")
    print("=" * 100)
    hierarchical_df = df[['Option', 'Hierarchical Accuracy', 'Hierarchical Precision', 'Hierarchical Recall', 'Hierarchical F1']]
    print("\n" + hierarchical_df.to_string(index=False))
    
    # Generate Markdown table
    markdown_table = generate_markdown_table(df)
    md_path = PROJECT_DIR / "Results" / "5FOLD_CV_SUMMARY_TABLE.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    print(f"\n* Saved Markdown: {md_path}")
    
    print("\n" + "=" * 100)
    print("✅ TABLE GENERATION COMPLETE")
    print("=" * 100)
    print("\nNotes:")
    print("  - Option-A: Flat 10-class classification with disease/severity extraction")
    print("  - Option-B: Two-stage cascade (separate disease and severity models)")
    print("  - Option-C: Shared backbone multi-task learning")
    print("  - Option-D: Option-C + severity masking (uses Option-C's CV results)")
    print("  - Option-E: Knowledge distillation from Option-C")
    print("  - All missing metrics calculated from CV fold results")
    print("  - All values are percentages (Mean ± Std) from 5-fold cross-validation")


def generate_markdown_table(df):
    """Generate formatted Markdown table"""
    
    md = """# 5-Fold Cross-Validation Summary Table

## Full Comparison (All Options)

All values are reported as **Mean ± Std (%)** from 5-fold cross-validation.

### Complete Metrics Table

"""
    
    # Create full markdown table manually
    md += df_to_markdown(df)
    
    md += "\n\n## By Classification Task\n\n"
    
    # Disease Classification
    md += "### Disease Classification (4-class)\n\n"
    md += "**Classes:** Healthy, LSD, FMD, IBK\n\n"
    disease_df = df[['Option', 'Disease Accuracy', 'Disease Precision', 'Disease Recall', 'Disease F1']]
    md += df_to_markdown(disease_df)
    
    # Severity Classification
    md += "\n\n### Severity Classification (3-class, Diseased Only)\n\n"
    md += "**Classes:** Stage 1, Stage 2, Stage 3 (excludes healthy samples)\n\n"
    severity_df = df[['Option', 'Severity Accuracy', 'Severity Precision', 'Severity Recall', 'Severity F1']]
    md += df_to_markdown(severity_df)
    
    # Hierarchical Classification
    md += "\n\n### Hierarchical Classification (10-class)\n\n"
    md += "**Classes:** Healthy, LSD-S1/S2/S3, FMD-S1/S2/S3, IBK-S1/S2/S3\n\n"
    hierarchical_df = df[['Option', 'Hierarchical Accuracy', 'Hierarchical Precision', 'Hierarchical Recall', 'Hierarchical F1']]
    md += df_to_markdown(hierarchical_df)
    
    md += "\n\n## Notes\n\n"
    md += "- **Option-A:** Flat 10-class classification with post-hoc disease/severity extraction\n"
    md += "- **Option-B:** Two-stage cascade (separate disease and severity models)\n"
    md += "- **Option-C:** Shared backbone multi-task learning (joint optimization)\n"
    md += "- **Option-D:** Option-C + severity masking (uses Option-C's trained model and CV results)\n"
    md += "- **Option-E:** Knowledge distillation from Option-C (student-teacher framework)\n"
    md += "- All missing precision/recall/F1 metrics calculated from CV fold results\n"
    md += "- All values are percentages (Mean ± Std) from 5-fold cross-validation\n"
    
    return md


def df_to_markdown(df):
    """Convert DataFrame to markdown table without tabulate dependency"""
    lines = []
    
    # Header
    header = "| " + " | ".join(df.columns) + " |"
    lines.append(header)
    
    # Separator
    separator = "|" + "|".join([" --- " for _ in df.columns]) + "|"
    lines.append(separator)
    
    # Rows
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(v) for v in row.values) + " |"
        lines.append(row_str)
    
    return "\n".join(lines)


if __name__ == "__main__":
    generate_table()
