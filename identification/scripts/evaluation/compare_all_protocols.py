import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_protocol_results(base_dir: Path, backbone: str, loss: str = "combined", fold: int = 0):
    """Load all protocol results for a given backbone."""
    results_dir = base_dir / "results"
    
    results = {}
    
    if backbone == "resnet50":
        pa_yolo = results_dir / f"protocolA_fold{fold}_yolo.csv"
        pa_gt = results_dir / f"protocolA_fold{fold}_gt.csv"
    else:
        pa_yolo = results_dir / f"protocolA_fold{fold}_{backbone}_{loss}_yolo.csv"
        pa_gt = results_dir / f"protocolA_fold{fold}_{backbone}_{loss}_gt.csv"
    
    if pa_yolo.exists():
        df_a = pd.read_csv(pa_yolo)
        avg_a = df_a[df_a['round'] == 'Average'].iloc[0]
        results['A_YOLO'] = {'top1': avg_a['top1'], 'top5': avg_a['top5'], 'mAP': avg_a['mAP']}
    
    if pa_gt.exists():
        df_a_gt = pd.read_csv(pa_gt)
        avg_a_gt = df_a_gt[df_a_gt['round'] == 'Average'].iloc[0]
        results['A_GT'] = {'top1': avg_a_gt['top1'], 'top5': avg_a_gt['top5'], 'mAP': avg_a_gt['mAP']}
    
    for protocol in ['B', 'C', 'D']:
        yolo_path = results_dir / f"protocol{protocol}_fold{fold}_{backbone}_{loss}_yolo.csv"
        if yolo_path.exists():
            df = pd.read_csv(yolo_path)
            results[f'{protocol}_YOLO'] = {
                'top1': (df['correct'].sum() / len(df)) * 100,
                'top5': 0,  # Not directly stored, compute if needed
                'num': len(df)
            }
        
        gt_path = results_dir / f"protocol{protocol}_fold{fold}_{backbone}_{loss}_gt.csv"
        if gt_path.exists():
            df = pd.read_csv(gt_path)
            results[f'{protocol}_GT'] = {
                'top1': (df['correct'].sum() / len(df)) * 100,
                'top5': 0,
                'num': len(df)
            }
    
    summary_path = results_dir / f"protocols_summary_fold{fold}_{backbone}_{loss}.csv"
    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        for _, row in df_summary.iterrows():
            protocol_name = row['protocol']
            if 'YOLO' in protocol_name:
                key = protocol_name.split()[-2] + '_YOLO'  # e.g., "Protocol B YOLO" -> "B_YOLO"
            else:
                key = protocol_name.split()[-2] + '_GT'
            results[key] = {
                'top1': row['top1'],
                'top5': row['top5'],
                'mAP': row['mAP'],
                'num': int(row['num_queries'])
            }
    
    return results

def compare_all_protocols():
    """Generate comprehensive comparison across all protocols."""
    
    base_dir = Path(r"d:\identification")
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"
    
    print("="*70)
    print("Comprehensive Protocol Comparison: ResNet-50 vs ConvNeXt-Tiny")
    print("="*70)
    
    print("\n[Loading Results]")
    resnet_results = load_protocol_results(base_dir, "resnet50")
    convnext_results = load_protocol_results(base_dir, "convnext_tiny")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    ax1 = axes[0, 0]
    protocols = ['A\nYOLO', 'A\nGT']
    resnet_a = [resnet_results.get('A_YOLO', {}).get('top1', 0),
                resnet_results.get('A_GT', {}).get('top1', 0)]
    convnext_a = [convnext_results.get('A_YOLO', {}).get('top1', 0),
                  convnext_results.get('A_GT', {}).get('top1', 0)]
    
    x = np.arange(len(protocols))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, resnet_a, width, label='ResNet-50', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, convnext_a, width, label='ConvNeXt-Tiny', alpha=0.8, color='#2ecc71')
    
    ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Protocol A (Within-ID Rotation)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(protocols)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2 = axes[0, 1]
    protocols_b = ['B\nYOLO', 'B\nGT']
    resnet_b = [resnet_results.get('B_YOLO', {}).get('top1', 0),
                resnet_results.get('B_GT', {}).get('top1', 0)]
    convnext_b = [convnext_results.get('B_YOLO', {}).get('top1', 0),
                  convnext_results.get('B_GT', {}).get('top1', 0)]
    
    x_b = np.arange(len(protocols_b))
    
    bars3 = ax2.bar(x_b - width/2, resnet_b, width, label='ResNet-50', alpha=0.8, color='#3498db')
    bars4 = ax2.bar(x_b + width/2, convnext_b, width, label='ConvNeXt-Tiny', alpha=0.8, color='#2ecc71')
    
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Protocol B (Cross-Angle)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(protocols_b)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax3 = axes[0, 2]
    protocols_c = ['C\nYOLO', 'C\nGT']
    resnet_c = [resnet_results.get('C_YOLO', {}).get('top1', 0),
                resnet_results.get('C_GT', {}).get('top1', 0)]
    convnext_c = [convnext_results.get('C_YOLO', {}).get('top1', 0),
                  convnext_results.get('C_GT', {}).get('top1', 0)]
    
    x_c = np.arange(len(protocols_c))
    
    bars5 = ax3.bar(x_c - width/2, resnet_c, width, label='ResNet-50', alpha=0.8, color='#3498db')
    bars6 = ax3.bar(x_c + width/2, convnext_c, width, label='ConvNeXt-Tiny', alpha=0.8, color='#2ecc71')
    
    ax3.set_ylabel('Top-1 Accuracy (%)', fontsize=10, fontweight='bold')
    ax3.set_title('Protocol C (Hard Cases)', fontsize=11, fontweight='bold')
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(protocols_c)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)
    
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax4 = axes[1, 0]
    protocols_d = ['D\nYOLO', 'D\nGT']
    resnet_d = [resnet_results.get('D_YOLO', {}).get('top1', 0),
                resnet_results.get('D_GT', {}).get('top1', 0)]
    convnext_d = [convnext_results.get('D_YOLO', {}).get('top1', 0),
                  convnext_results.get('D_GT', {}).get('top1', 0)]
    
    x_d = np.arange(len(protocols_d))
    
    bars7 = ax4.bar(x_d - width/2, resnet_d, width, label='ResNet-50', alpha=0.8, color='#3498db')
    bars8 = ax4.bar(x_d + width/2, convnext_d, width, label='ConvNeXt-Tiny', alpha=0.8, color='#2ecc71')
    
    ax4.set_ylabel('Top-1 Accuracy (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Protocol D (Unknown IDs)', fontsize=11, fontweight='bold')
    ax4.set_xticks(x_d)
    ax4.set_xticklabels(protocols_d)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 100)
    
    for bars in [bars7, bars8]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax5 = axes[1, 1]
    protocols_all = ['A', 'B', 'C', 'D']
    resnet_all_yolo = [
        resnet_results.get('A_YOLO', {}).get('top1', 0),
        resnet_results.get('B_YOLO', {}).get('top1', 0),
        resnet_results.get('C_YOLO', {}).get('top1', 0),
        resnet_results.get('D_YOLO', {}).get('top1', 0)
    ]
    convnext_all_yolo = [
        convnext_results.get('A_YOLO', {}).get('top1', 0),
        convnext_results.get('B_YOLO', {}).get('top1', 0),
        convnext_results.get('C_YOLO', {}).get('top1', 0),
        convnext_results.get('D_YOLO', {}).get('top1', 0)
    ]
    
    x_all = np.arange(len(protocols_all))
    
    bars9 = ax5.bar(x_all - width/2, resnet_all_yolo, width, label='ResNet-50', alpha=0.8, color='#3498db')
    bars10 = ax5.bar(x_all + width/2, convnext_all_yolo, width, label='ConvNeXt-Tiny', alpha=0.8, color='#2ecc71')
    
    ax5.set_ylabel('Top-1 Accuracy (%)', fontsize=10, fontweight='bold')
    ax5.set_title('All Protocols (YOLO - End-to-End)', fontsize=11, fontweight='bold')
    ax5.set_xticks(x_all)
    ax5.set_xticklabels(protocols_all)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_ylim(0, 100)
    
    for bars in [bars9, bars10]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    table_data = [
        ['Protocol', 'ResNet-50', 'ConvNeXt', 'Winner'],
        ['A (YOLO)', f"{resnet_results.get('A_YOLO', {}).get('top1', 0):.1f}%", 
         f"{convnext_results.get('A_YOLO', {}).get('top1', 0):.1f}%",
         '✓ CN' if convnext_results.get('A_YOLO', {}).get('top1', 0) > resnet_results.get('A_YOLO', {}).get('top1', 0) else '✓ RN'],
        ['B (YOLO)', f"{resnet_results.get('B_YOLO', {}).get('top1', 0):.1f}%",
         f"{convnext_results.get('B_YOLO', {}).get('top1', 0):.1f}%",
         '✓ CN' if convnext_results.get('B_YOLO', {}).get('top1', 0) > resnet_results.get('B_YOLO', {}).get('top1', 0) else '✓ RN'],
        ['C (YOLO)', f"{resnet_results.get('C_YOLO', {}).get('top1', 0):.1f}%",
         f"{convnext_results.get('C_YOLO', {}).get('top1', 0):.1f}%",
         '✓ CN' if convnext_results.get('C_YOLO', {}).get('top1', 0) > resnet_results.get('C_YOLO', {}).get('top1', 0) else '✓ RN'],
        ['D (YOLO)', f"{resnet_results.get('D_YOLO', {}).get('top1', 0):.1f}%",
         f"{convnext_results.get('D_YOLO', {}).get('top1', 0):.1f}%",
         '✓ CN' if convnext_results.get('D_YOLO', {}).get('top1', 0) > resnet_results.get('D_YOLO', {}).get('top1', 0) else '✓ RN'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    ax6.set_title('Summary (Top-1 Accuracy)', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle('Complete Protocol Evaluation: ResNet-50 vs ConvNeXt-Tiny', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = figures_dir / "all_protocols_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive comparison to: {output_path}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("FINAL SUMMARY: All Protocols")
    print("="*70)
    
    for protocol in ['A', 'B', 'C', 'D']:
        print(f"\nProtocol {protocol}:")
        print(f"  ResNet-50 (YOLO):      {resnet_results.get(f'{protocol}_YOLO', {}).get('top1', 0):.2f}%")
        print(f"  ConvNeXt-Tiny (YOLO):  {convnext_results.get(f'{protocol}_YOLO', {}).get('top1', 0):.2f}%")
        diff = convnext_results.get(f'{protocol}_YOLO', {}).get('top1', 0) - resnet_results.get(f'{protocol}_YOLO', {}).get('top1', 0)
        winner = "ConvNeXt-Tiny" if diff > 0 else "ResNet-50"
        print(f"  Winner: {winner} ({diff:+.2f}%)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    compare_all_protocols()
