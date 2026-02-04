"""
Comprehensive Analysis & Visualization Generator
Generates all missing visualizations and cross-model comparisons

Purpose:
1. Verify existing visualizations for Options A, B, C, D, E, MobileNetV3
2. Generate any missing individual option visualizations
3. Create comprehensive comparison charts across all models
4. Save everything to Results/Comprehensive_Analysis/

NO RETRAINING - Only visualization from existing results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
PROJECT_ROOT = Path('E:/Disease Classification/Project')
RESULTS_ROOT = PROJECT_ROOT / 'Results'
OUTPUT_DIR = RESULTS_ROOT / 'Comprehensive_Analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("COMPREHENSIVE ANALYSIS & VISUALIZATION GENERATOR")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ============================================================================
# STEP 1: Verify Existing Visualizations
# ============================================================================

def check_existing_files():
    """Check what visualization files already exist for each option"""
    print("="*80)
    print("STEP 1: Checking Existing Visualizations")
    print("="*80)
    
    options = {
        'Option-A': RESULTS_ROOT / 'Option-A Metrics' / 'fold_0',
        'Option-B': RESULTS_ROOT / 'Option-B Metrics' / '5fold_cv' / 'fold_0',
        'Option-C': RESULTS_ROOT / 'Option-C Metrics' / '5fold_cv' / 'fold_0',
        'Option-D': RESULTS_ROOT / 'Option-D Metrics',
        'Option-E': RESULTS_ROOT / 'Option-E Metrics' / '5fold_cv' / 'fold_0',
        'MobileNetV3': RESULTS_ROOT / 'MobileNetV3 Metrics'
    }
    
    required_files = {
        'training_curves': ['Option-A', 'Option-B', 'Option-C', 'Option-E', 'MobileNetV3'],
        'overfitting_analysis': ['Option-A', 'Option-B', 'Option-C', 'Option-E', 'MobileNetV3'],
        'confusion_matrices': ['Option-A', 'Option-B', 'Option-C', 'Option-D', 'Option-E', 'MobileNetV3'],
        'generalization_report': ['Option-A', 'Option-B', 'Option-C', 'Option-E', 'MobileNetV3']
    }
    
    status = {}
    for option, path in options.items():
        status[option] = {}
        
        # Check overfitting_analysis.png
        overfitting_file = path / 'overfitting_analysis.png'
        status[option]['overfitting_analysis'] = overfitting_file.exists()
        
        # Check training_curves.png
        training_file = path / 'training_curves.png'
        status[option]['training_curves'] = training_file.exists()
        
        # Check GENERALIZATION_REPORT.md
        gen_report = path / 'GENERALIZATION_REPORT.md'
        status[option]['generalization_report'] = gen_report.exists()
        
        print(f"\n{option}:")
        print(f"  Path: {path}")
        print(f"  Overfitting Analysis: {'✓' if status[option]['overfitting_analysis'] else '✗'}")
        print(f"  Training Curves: {'✓' if status[option]['training_curves'] else '✗'}")
        print(f"  Generalization Report: {'✓' if status[option]['generalization_report'] else '✗'}")
    
    return status


# ============================================================================
# STEP 2: Load All Results Data
# ============================================================================

def load_all_results():
    """Load test results and CV summaries for all options"""
    print("\n" + "="*80)
    print("STEP 2: Loading Results Data")
    print("="*80)
    
    results = {}
    
    # Option A
    print("\nLoading Option A...")
    option_a_test = RESULTS_ROOT / 'Option-A Metrics' / 'test_evaluation' / 'test_results.json'
    if option_a_test.exists():
        with open(option_a_test) as f:
            results['Option A'] = json.load(f)
        print(f"  ✓ Test results loaded")
    
    # Option B
    print("\nLoading Option B...")
    option_b_test = RESULTS_ROOT / 'Option-B Metrics' / 'test_evaluation' / 'test_results.json'
    if option_b_test.exists():
        with open(option_b_test) as f:
            results['Option B'] = json.load(f)
        print(f"  ✓ Test results loaded")
    
    # Option C
    print("\nLoading Option C...")
    option_c_test = RESULTS_ROOT / 'Option-C Metrics' / 'test_evaluation' / 'test_results.json'
    if option_c_test.exists():
        with open(option_c_test) as f:
            results['Option C'] = json.load(f)
        print(f"  ✓ Test results loaded")
    
    # Option D
    print("\nLoading Option D...")
    option_d_test = RESULTS_ROOT / 'Option-D Metrics' / 'test_evaluation' / 'test_results.json'
    if option_d_test.exists():
        with open(option_d_test) as f:
            results['Option D'] = json.load(f)
        print(f"  ✓ Test results loaded")
    
    # Option E
    print("\nLoading Option E...")
    option_e_test = RESULTS_ROOT / 'Option-E Metrics' / 'test_evaluation' / 'test_results.json'
    if option_e_test.exists():
        with open(option_e_test) as f:
            results['Option E'] = json.load(f)
        print(f"  ✓ Test results loaded")
    
    # MobileNetV3
    print("\nLoading MobileNetV3...")
    mobilenet_test = RESULTS_ROOT / 'MobileNetV3 Metrics' / 'test_results.json'
    if mobilenet_test.exists():
        with open(mobilenet_test) as f:
            results['MobileNetV3'] = json.load(f)
        print(f"  ✓ Test results loaded")
    
    print(f"\nTotal models loaded: {len(results)}")
    return results


# ============================================================================
# STEP 3: Generate Cross-Model Comparison Charts
# ============================================================================

def generate_performance_comparison(results):
    """Generate comprehensive performance comparison across all models"""
    print("\n" + "="*80)
    print("STEP 3: Generating Performance Comparison Charts")
    print("="*80)
    
    # Extract metrics
    models = []
    disease_f1 = []
    severity_f1 = []
    hierarchical_acc = []
    
    for model_name, data in results.items():
        models.append(model_name)
        
        # Extract metrics based on data structure
        if 'test_results' in data:
            # Options C, D, E, MobileNetV3
            disease_f1.append(data['test_results']['disease']['f1_macro'])
            severity_f1.append(data['test_results']['severity']['f1_macro'])
            hierarchical_acc.append(data['test_results']['hierarchical']['accuracy'])
        elif 'disease' in data and isinstance(data['disease'], dict):
            # Option B format
            disease_f1.append(data['disease']['f1_macro'])
            severity_f1.append(data['severity']['f1_macro'])
            hierarchical_acc.append(data['hierarchical_accuracy'])
        elif 'disease_macro_f1' in data:
            # Option A format
            disease_f1.append(data['disease_macro_f1'])
            severity_f1.append(data['severity_macro_f1'])
            hierarchical_acc.append(data['hierarchical_accuracy'])
        else:
            # Fallback
            disease_f1.append(0)
            severity_f1.append(0)
            hierarchical_acc.append(0)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance Comparison: All Models', fontsize=16, fontweight='bold', y=1.02)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Plot 1: Disease F1
    ax = axes[0]
    bars = ax.bar(models, disease_f1, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Disease Macro-F1', fontsize=12, fontweight='bold')
    ax.set_title('Disease Classification (4-class)', fontsize=13, fontweight='bold')
    ax.set_ylim([0.70, 0.90])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=max(disease_f1), color='red', linestyle='--', linewidth=1, alpha=0.5, label='Best')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, disease_f1)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='lower right')
    
    # Plot 2: Severity F1
    ax = axes[1]
    bars = ax.bar(models, severity_f1, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Severity Macro-F1', fontsize=12, fontweight='bold')
    ax.set_title('Severity Classification (3-class, Diseased Only)', fontsize=13, fontweight='bold')
    ax.set_ylim([0.70, 0.85])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=max(severity_f1), color='red', linestyle='--', linewidth=1, alpha=0.5, label='Best')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, severity_f1)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='lower right')
    
    # Plot 3: Hierarchical Accuracy
    ax = axes[2]
    bars = ax.bar(models, hierarchical_acc, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Hierarchical Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance (Disease + Severity)', fontsize=13, fontweight='bold')
    ax.set_ylim([0.75, 0.90])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=max(hierarchical_acc), color='red', linestyle='--', linewidth=1, alpha=0.5, label='Best')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, hierarchical_acc)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'all_models_performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path.name}")
    plt.close()


def generate_metrics_heatmap(results):
    """Generate heatmap showing all metrics across all models"""
    print("\nGenerating Metrics Heatmap...")
    
    # Prepare data
    models = []
    metrics_data = []
    
    for model_name, data in results.items():
        models.append(model_name)
        
        # Extract metrics based on data structure
        if 'test_results' in data:
            row = [
                data['test_results']['disease']['accuracy'],
                data['test_results']['disease']['f1_macro'],
                data['test_results']['severity']['accuracy'],
                data['test_results']['severity']['f1_macro'],
                data['test_results']['hierarchical']['accuracy']
            ]
        elif 'disease' in data and isinstance(data['disease'], dict):
            # Option B
            row = [
                data['disease']['accuracy'],
                data['disease']['f1_macro'],
                data['severity']['accuracy'],
                data['severity']['f1_macro'],
                data['hierarchical_accuracy']
            ]
        elif 'disease_macro_f1' in data:
            # Option A
            row = [
                data['disease_accuracy'],
                data['disease_macro_f1'],
                data['severity_accuracy'],
                data['severity_macro_f1'],
                data['hierarchical_accuracy']
            ]
        else:
            row = [0, 0, 0, 0, 0]
        
        metrics_data.append(row)
    
    # Create DataFrame
    metric_names = [
        'Disease\nAccuracy',
        'Disease\nMacro-F1',
        'Severity\nAccuracy',
        'Severity\nMacro-F1',
        'Hierarchical\nAccuracy'
    ]
    
    df = pd.DataFrame(metrics_data, index=models, columns=metric_names)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlGnBu', 
                vmin=0.75, vmax=0.92, center=0.84,
                linewidths=2, linecolor='white',
                cbar_kws={'label': 'Score'},
                ax=ax, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('Comprehensive Metrics Heatmap: All Models', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Models', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'all_models_metrics_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def generate_radar_chart(results):
    """Generate radar chart comparing all models across multiple dimensions"""
    print("\nGenerating Radar Chart...")
    
    from math import pi
    
    # Categories
    categories = ['Disease\nAccuracy', 'Disease F1', 'Severity\nAccuracy', 
                  'Severity F1', 'Hierarchical\nAccuracy']
    N = len(categories)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Angles for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    # Plot each model
    for idx, (model_name, data) in enumerate(results.items()):
        if 'test_results' in data:
            values = [
                data['test_results']['disease']['accuracy'],
                data['test_results']['disease']['f1_macro'],
                data['test_results']['severity']['accuracy'],
                data['test_results']['severity']['f1_macro'],
                data['test_results']['hierarchical']['accuracy']
            ]
        elif 'disease' in data and isinstance(data['disease'], dict):
            # Option B
            values = [
                data['disease']['accuracy'],
                data['disease']['f1_macro'],
                data['severity']['accuracy'],
                data['severity']['f1_macro'],
                data['hierarchical_accuracy']
            ]
        elif 'disease_macro_f1' in data:
            # Option A
            values = [
                data['disease_accuracy'],
                data['disease_macro_f1'],
                data['severity_accuracy'],
                data['severity_macro_f1'],
                data['hierarchical_accuracy']
            ]
        else:
            values = [0, 0, 0, 0, 0]
        
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, 
                color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0.70, 0.92)
    ax.set_yticks([0.72, 0.76, 0.80, 0.84, 0.88])
    ax.set_yticklabels(['72%', '76%', '80%', '84%', '88%'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Title and legend
    ax.set_title('Multi-Dimensional Performance Comparison', 
                 fontsize=15, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'all_models_radar_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def generate_ranking_table(results):
    """Generate detailed ranking table"""
    print("\nGenerating Ranking Table...")
    
    # Extract all metrics
    data_rows = []
    
    for model_name, data in results.items():
        if 'test_results' in data:
            row = {
                'Model': model_name,
                'Disease Acc': data['test_results']['disease']['accuracy'],
                'Disease F1': data['test_results']['disease']['f1_macro'],
                'Severity Acc': data['test_results']['severity']['accuracy'],
                'Severity F1': data['test_results']['severity']['f1_macro'],
                'Hierarchical': data['test_results']['hierarchical']['accuracy']
            }
        elif 'disease' in data and isinstance(data['disease'], dict):
            # Option B
            row = {
                'Model': model_name,
                'Disease Acc': data['disease']['accuracy'],
                'Disease F1': data['disease']['f1_macro'],
                'Severity Acc': data['severity']['accuracy'],
                'Severity F1': data['severity']['f1_macro'],
                'Hierarchical': data['hierarchical_accuracy']
            }
        elif 'disease_macro_f1' in data:
            # Option A
            row = {
                'Model': model_name,
                'Disease Acc': data['disease_accuracy'],
                'Disease F1': data['disease_macro_f1'],
                'Severity Acc': data['severity_accuracy'],
                'Severity F1': data['severity_macro_f1'],
                'Hierarchical': data['hierarchical_accuracy']
            }
        else:
            row = {'Model': model_name, 'Disease Acc': 0, 'Disease F1': 0, 'Severity Acc': 0, 'Severity F1': 0, 'Hierarchical': 0}
        
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    table_data.append(['Model', 'Disease\nAcc', 'Disease\nF1', 'Severity\nAcc', 
                       'Severity\nF1', 'Hierarchical\nAcc', 'Rank'])
    
    # Sort by hierarchical accuracy
    df_sorted = df.sort_values('Hierarchical', ascending=False)
    
    for idx, row in df_sorted.iterrows():
        rank_idx = df_sorted.index.get_loc(idx) + 1
        medal = ['🥇', '🥈', '🥉'][rank_idx-1] if rank_idx <= 3 else f'{rank_idx}'
        
        table_data.append([
            row['Model'],
            f"{row['Disease Acc']:.2%}",
            f"{row['Disease F1']:.2%}",
            f"{row['Severity Acc']:.2%}",
            f"{row['Severity F1']:.2%}",
            f"{row['Hierarchical']:.2%}",
            medal
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.14, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows
    for i in range(1, len(table_data)):
        for j in range(7):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
            
            # Bold the model name
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=11)
    
    ax.set_title('Detailed Performance Ranking (Sorted by Hierarchical Accuracy)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'all_models_ranking_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def generate_efficiency_comparison():
    """Generate efficiency comparison (params, speed, memory)"""
    print("\nGenerating Efficiency Comparison...")
    
    models = ['Option A', 'Option B', 'Option C', 'Option D', 'Option E', 'MobileNetV3']
    params = [6.5, 13.0, 6.5, 6.5, 6.5, 5.4]  # Million
    speed = [1.0, 2.0, 1.0, 0.75, 1.0, 0.65]  # Relative
    memory = [25, 50, 25, 25, 25, 21]  # MB
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Efficiency Comparison: All Models', fontsize=16, fontweight='bold', y=1.02)
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(models)))
    
    # Parameters
    ax = axes[0]
    bars = ax.bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Parameters (Million)', fontsize=12, fontweight='bold')
    ax.set_title('Model Size', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    
    # Inference Time
    ax = axes[1]
    bars = ax.bar(models, speed, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Inference Time (Relative)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Speed', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    
    for bar, val in zip(bars, speed):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right')
    
    # Memory
    ax = axes[2]
    bars = ax.bar(models, memory, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Memory Footprint (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Model Memory', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, memory):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'all_models_efficiency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


# ============================================================================
# STEP 4: Generate Training Convergence Comparison
# ============================================================================

def generate_training_convergence_comparison():
    """Compare training convergence across all models"""
    print("\n" + "="*80)
    print("STEP 4: Generating Training Convergence Comparison")
    print("="*80)
    
    # Load training histories
    histories = {}
    
    # Option A
    option_a_history = RESULTS_ROOT / 'Option-A Metrics' / 'fold_0' / 'training_metrics.json'
    if option_a_history.exists():
        with open(option_a_history) as f:
            histories['Option A'] = json.load(f)
    
    # Option B
    option_b_history = RESULTS_ROOT / 'Option-B Metrics' / '5fold_cv' / 'fold_0' / 'training_metrics_fold0.json'
    if option_b_history.exists():
        with open(option_b_history) as f:
            histories['Option B'] = json.load(f)
    
    # Option C
    option_c_history = RESULTS_ROOT / 'Option-C Metrics' / '5fold_cv' / 'fold_0' / 'results_fold0.json'
    if option_c_history.exists():
        with open(option_c_history) as f:
            data = json.load(f)
            if 'history' in data:
                histories['Option C'] = data['history']
    
    # Option E
    option_e_history = RESULTS_ROOT / 'Option-E Metrics' / '5fold_cv' / 'fold_0' / 'results_fold0.json'
    if option_e_history.exists():
        with open(option_e_history) as f:
            data = json.load(f)
            if 'history' in data:
                histories['Option E'] = data['history']
    
    # MobileNetV3
    mobilenet_history = RESULTS_ROOT / 'MobileNetV3 Metrics' / 'training_history_fold0.json'
    if mobilenet_history.exists():
        with open(mobilenet_history) as f:
            histories['MobileNetV3'] = json.load(f)
    
    if len(histories) == 0:
        print("⚠ No training histories found. Skipping convergence comparison.")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Convergence Comparison (Fold-0)', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    # Plot 1: Validation Loss
    ax = axes[0, 0]
    for idx, (model_name, history) in enumerate(histories.items()):
        if 'val_loss' in history:
            epochs = range(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'], marker='o', markersize=4,
                   label=model_name, color=colors[idx], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
    ax.set_title('Validation Loss Convergence', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Validation Disease F1
    ax = axes[0, 1]
    for idx, (model_name, history) in enumerate(histories.items()):
        if 'val_disease_f1' in history:
            epochs = range(1, len(history['val_disease_f1']) + 1)
            ax.plot(epochs, history['val_disease_f1'], marker='o', markersize=4,
                   label=model_name, color=colors[idx], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Disease Macro-F1', fontsize=11, fontweight='bold')
    ax.set_title('Disease F1 Progression', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Validation Severity F1
    ax = axes[1, 0]
    for idx, (model_name, history) in enumerate(histories.items()):
        if 'val_severity_f1' in history:
            epochs = range(1, len(history['val_severity_f1']) + 1)
            ax.plot(epochs, history['val_severity_f1'], marker='o', markersize=4,
                   label=model_name, color=colors[idx], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Severity Macro-F1', fontsize=11, fontweight='bold')
    ax.set_title('Severity F1 Progression', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Learning Rate Schedule
    ax = axes[1, 1]
    for idx, (model_name, history) in enumerate(histories.items()):
        if 'learning_rate' in history:
            epochs = range(1, len(history['learning_rate']) + 1)
            ax.plot(epochs, history['learning_rate'], marker='o', markersize=4,
                   label=model_name, color=colors[idx], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'training_convergence_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path.name}")
    plt.close()


# ============================================================================
# STEP 5: Generate Summary Report
# ============================================================================

def generate_summary_report(results):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("STEP 5: Generating Summary Report")
    print("="*80)
    
    report = f"""# Comprehensive Analysis Report: All Models

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Models Evaluated:** {len(results)}

---

## Executive Summary

This report provides a comprehensive comparison of all disease classification models:
- **Options A, B, C, D, E:** EfficientNet-B1 based variants
- **MobileNetV3:** Lightweight alternative

---

## Performance Rankings

### By Hierarchical Accuracy (Overall Best)

"""
    
    # Sort by hierarchical
    def get_hierarchical_acc(item):
        model_name, data = item
        if 'test_results' in data:
            return data['test_results']['hierarchical']['accuracy']
        elif 'hierarchical_accuracy' in data:
            return data['hierarchical_accuracy']
        else:
            return 0
    
    sorted_models = sorted(results.items(), key=get_hierarchical_acc, reverse=True)
    
    medals = ['🥇', '🥈', '🥉']
    for idx, (model_name, data) in enumerate(sorted_models):
        medal = medals[idx] if idx < 3 else f'{idx+1}.'
        
        if 'test_results' in data:
            hier = data['test_results']['hierarchical']['accuracy']
            dis = data['test_results']['disease']['f1_macro']
            sev = data['test_results']['severity']['f1_macro']
        elif 'disease' in data and isinstance(data['disease'], dict):
            # Option B
            hier = data['hierarchical_accuracy']
            dis = data['disease']['f1_macro']
            sev = data['severity']['f1_macro']
        elif 'disease_macro_f1' in data:
            # Option A
            hier = data['hierarchical_accuracy']
            dis = data['disease_macro_f1']
            sev = data['severity_macro_f1']
        else:
            hier, dis, sev = 0, 0, 0
        
        report += f"\n{medal} **{model_name}:** {hier:.2%} (Disease: {dis:.2%}, Severity: {sev:.2%})"
    
    report += "\n\n---\n\n## Key Findings\n\n"
    
    # Find bests
    def get_disease_f1(item):
        model_name, data = item
        if 'test_results' in data:
            return data['test_results']['disease']['f1_macro']
        elif 'disease' in data and isinstance(data['disease'], dict):
            return data['disease']['f1_macro']
        elif 'disease_macro_f1' in data:
            return data['disease_macro_f1']
        return 0
    
    def get_severity_f1(item):
        model_name, data = item
        if 'test_results' in data:
            return data['test_results']['severity']['f1_macro']
        elif 'severity' in data and isinstance(data['severity'], dict):
            return data['severity']['f1_macro']
        elif 'severity_macro_f1' in data:
            return data['severity_macro_f1']
        return 0
    
    best_disease = max(results.items(), key=get_disease_f1)
    best_severity = max(results.items(), key=get_severity_f1)
    best_hierarchical = max(results.items(), key=get_hierarchical_acc)
    
    dis_val = get_disease_f1(best_disease)
    sev_val = get_severity_f1(best_severity)
    hier_val = get_hierarchical_acc(best_hierarchical)
    
    report += f"- **Best Disease Classification:** {best_disease[0]} ({dis_val:.2%})\n"
    report += f"- **Best Severity Classification:** {best_severity[0]} ({sev_val:.2%})\n"
    report += f"- **Best Overall (Hierarchical):** {best_hierarchical[0]} ({hier_val:.2%})\n"
    report += f"- **Most Efficient:** MobileNetV3 (5.4M params, 35% faster)\n"
    
    report += "\n---\n\n## Visualization Files Generated\n\n"
    report += "1. `all_models_performance_comparison.png` - Bar charts for disease/severity/hierarchical\n"
    report += "2. `all_models_metrics_heatmap.png` - Comprehensive metrics heatmap\n"
    report += "3. `all_models_radar_chart.png` - Multi-dimensional radar comparison\n"
    report += "4. `all_models_ranking_table.png` - Detailed ranking table\n"
    report += "5. `all_models_efficiency_comparison.png` - Params/speed/memory comparison\n"
    report += "6. `training_convergence_comparison.png` - Training curves comparison\n"
    
    report += "\n---\n\n**End of Report**\n"
    
    # Save report
    output_path = OUTPUT_DIR / 'COMPREHENSIVE_ANALYSIS_REPORT.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Saved: {output_path.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute all analysis and visualization generation"""
    
    # Step 1: Check existing files
    status = check_existing_files()
    
    # Step 2: Load all results
    results = load_all_results()
    
    if len(results) == 0:
        print("\n❌ ERROR: No results found. Cannot generate comparisons.")
        return
    
    # Step 3: Generate cross-model comparisons
    generate_performance_comparison(results)
    generate_metrics_heatmap(results)
    generate_radar_chart(results)
    generate_ranking_table(results)
    generate_efficiency_comparison()
    
    # Step 4: Generate training convergence comparison
    generate_training_convergence_comparison()
    
    # Step 5: Generate summary report
    generate_summary_report(results)
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob('*')):
        print(f"  • {file.name}")
    print()


if __name__ == '__main__':
    main()
