import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def compare_backbones():
    """Compare ResNet-50 vs ConvNeXt-Tiny on Protocol A results."""
    
    base_dir = Path(r"d:\identification")
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"
    
    df_resnet_yolo = pd.read_csv(results_dir / "protocolA_fold0_yolo.csv")  # Old naming
    df_convnext_yolo = pd.read_csv(results_dir / "protocolA_fold0_convnext_tiny_combined_yolo.csv")
    
    df_resnet_gt = pd.read_csv(results_dir / "protocolA_fold0_gt.csv")  # Old naming
    df_convnext_gt = pd.read_csv(results_dir / "protocolA_fold0_convnext_tiny_combined_gt.csv")
    
    print("="*70)
    print("Step-9 Lite: Backbone Comparison (ResNet-50 vs ConvNeXt-Tiny)")
    print("="*70)
    
    resnet_avg_yolo = df_resnet_yolo[df_resnet_yolo['round'] == 'Average'].iloc[0]
    convnext_avg_yolo = df_convnext_yolo[df_convnext_yolo['round'] == 'Average'].iloc[0]
    
    resnet_avg_gt = df_resnet_gt[df_resnet_gt['round'] == 'Average'].iloc[0]
    convnext_avg_gt = df_convnext_gt[df_convnext_gt['round'] == 'Average'].iloc[0]
    
    print("\n[1] Test Set Performance (Protocol A - 4-Round Rotation)")
    print("-"*70)
    print("\nYOLO Crops (End-to-End Deployment):")
    print(f"  ResNet-50:      Top-1={resnet_avg_yolo['top1']:.2f}%, Top-5={resnet_avg_yolo['top5']:.2f}%, mAP={resnet_avg_yolo['mAP']:.2f}%")
    print(f"  ConvNeXt-Tiny:  Top-1={convnext_avg_yolo['top1']:.2f}%, Top-5={convnext_avg_yolo['top5']:.2f}%, mAP={convnext_avg_yolo['mAP']:.2f}%")
    print(f"  Δ (ConvNeXt-ResNet): Top-1={convnext_avg_yolo['top1']-resnet_avg_yolo['top1']:+.2f}%, Top-5={convnext_avg_yolo['top5']-resnet_avg_yolo['top5']:+.2f}%, mAP={convnext_avg_yolo['mAP']-resnet_avg_yolo['mAP']:+.2f}%")
    
    print("\nGT Crops (Oracle Upper Bound):")
    print(f"  ResNet-50:      Top-1={resnet_avg_gt['top1']:.2f}%, Top-5={resnet_avg_gt['top5']:.2f}%, mAP={resnet_avg_gt['mAP']:.2f}%")
    print(f"  ConvNeXt-Tiny:  Top-1={convnext_avg_gt['top1']:.2f}%, Top-5={convnext_avg_gt['top5']:.2f}%, mAP={convnext_avg_gt['mAP']:.2f}%")
    print(f"  Δ (ConvNeXt-ResNet): Top-1={convnext_avg_gt['top1']-resnet_avg_gt['top1']:+.2f}%, Top-5={convnext_avg_gt['top5']-resnet_avg_gt['top5']:+.2f}%, mAP={convnext_avg_gt['mAP']-resnet_avg_gt['mAP']:+.2f}%")
    
    print("\n[2] Validation Performance (During Training)")
    print("-"*70)
    print("  ResNet-50:      Val Top-1=76.00% (Epoch 25, stopped at 37)")
    print("  ConvNeXt-Tiny:  Val Top-1=93.00% (Epoch 36, stopped at 48)")
    print("  Δ (ConvNeXt-ResNet): Val Top-1=+17.00%")
    
    print("\n[3] Model Statistics")
    print("-"*70)
    print("  ResNet-50:")
    print("    Parameters: 26.13M (2.63M trainable when frozen)")
    print("    Training time: ~19 min (37 epochs)")
    print("    Best epoch: 25")
    print("\n  ConvNeXt-Tiny:")
    print("    Parameters: 28.31M (493K trainable when frozen)")
    print("    Training time: ~23 min (48 epochs)")
    print("    Best epoch: 36")
    
    print("\n[4] Winner Analysis")
    print("-"*70)
    
    yolo_winner = "ConvNeXt-Tiny" if convnext_avg_yolo['top1'] > resnet_avg_yolo['top1'] else "ResNet-50" if resnet_avg_yolo['top1'] > convnext_avg_yolo['top1'] else "Tie"
    gt_winner = "ConvNeXt-Tiny" if convnext_avg_gt['top1'] > resnet_avg_gt['top1'] else "ResNet-50" if resnet_avg_gt['top1'] > convnext_avg_gt['top1'] else "Tie"
    
    print(f"\n  YOLO (End-to-End):  {yolo_winner} wins")
    print(f"  GT (Oracle):        {gt_winner} wins")
    print(f"  Validation:         ConvNeXt-Tiny wins (+17% absolute)")
    
    print("\n[5] Recommendation")
    print("-"*70)
    
    if convnext_avg_yolo['top1'] > resnet_avg_yolo['top1']:
        print("\n  ✅ BEST MODEL: ConvNeXt-Tiny")
        print("     Reasons:")
        print(f"       • Higher test Top-1: {convnext_avg_yolo['top1']:.2f}% vs {resnet_avg_yolo['top1']:.2f}% (+{convnext_avg_yolo['top1']-resnet_avg_yolo['top1']:.2f}%)")
        print(f"       • Higher mAP: {convnext_avg_yolo['mAP']:.2f}% vs {resnet_avg_yolo['mAP']:.2f}% (+{convnext_avg_yolo['mAP']-resnet_avg_yolo['mAP']:.2f}%)")
        print(f"       • Much better validation: 93% vs 76% (+17%)")
        print(f"       • Similar parameters: 28.31M vs 26.13M")
        best_backbone = "convnext_tiny"
    else:
        print("\n  ✅ BEST MODEL: ResNet-50")
        print("     Reasons:")
        print(f"       • Higher test Top-1: {resnet_avg_yolo['top1']:.2f}% vs {convnext_avg_yolo['top1']:.2f}%")
        best_backbone = "resnet50"
    
    print("\n[6] Generating Comparison Visualization")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    metrics = ['Top-1', 'Top-5', 'mAP']
    resnet_vals = [resnet_avg_yolo['top1'], resnet_avg_yolo['top5'], resnet_avg_yolo['mAP']]
    convnext_vals = [convnext_avg_yolo['top1'], convnext_avg_yolo['top5'], convnext_avg_yolo['mAP']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, resnet_vals, width, label='ResNet-50', alpha=0.8, color='#3498db')
    bars2 = ax1.bar(x + width/2, convnext_vals, width, label='ConvNeXt-Tiny', alpha=0.8, color='#2ecc71')
    
    ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Test Performance (YOLO Crops)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[0, 1]
    
    resnet_rounds = df_resnet_yolo[df_resnet_yolo['round'] != 'Average']
    convnext_rounds = df_convnext_yolo[df_convnext_yolo['round'] != 'Average']
    
    rounds = ['R1\n(Front)', 'R2\n(Back)', 'R3\n(Left)', 'R4\n(Right)']
    x_rounds = np.arange(len(rounds))
    
    bars3 = ax2.bar(x_rounds - width/2, resnet_rounds['top1'].values, width, 
                    label='ResNet-50', alpha=0.8, color='#3498db')
    bars4 = ax2.bar(x_rounds + width/2, convnext_rounds['top1'].values, width,
                    label='ConvNeXt-Tiny', alpha=0.8, color='#2ecc71')
    
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Per-Round Performance (YOLO)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_rounds)
    ax2.set_xticklabels(rounds)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 100)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax3 = axes[1, 0]
    
    categories = ['Val\nTop-1', 'Test\nTop-1\n(YOLO)', 'Test\nmAP\n(YOLO)']
    resnet_train_test = [76.0, resnet_avg_yolo['top1'], resnet_avg_yolo['mAP']]
    convnext_train_test = [93.0, convnext_avg_yolo['top1'], convnext_avg_yolo['mAP']]
    
    x_cat = np.arange(len(categories))
    
    bars5 = ax3.bar(x_cat - width/2, resnet_train_test, width,
                    label='ResNet-50', alpha=0.8, color='#3498db')
    bars6 = ax3.bar(x_cat + width/2, convnext_train_test, width,
                    label='ConvNeXt-Tiny', alpha=0.8, color='#2ecc71')
    
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Validation vs Test Performance', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_cat)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)
    
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'ResNet-50', 'ConvNeXt-Tiny', 'Winner'],
        ['Val Top-1', '76.0%', '93.0%', '✓ ConvNeXt'],
        ['Test Top-1 (YOLO)', f"{resnet_avg_yolo['top1']:.1f}%", f"{convnext_avg_yolo['top1']:.1f}%", 
         '✓ ConvNeXt' if convnext_avg_yolo['top1'] > resnet_avg_yolo['top1'] else '✓ ResNet'],
        ['Test mAP (YOLO)', f"{resnet_avg_yolo['mAP']:.1f}%", f"{convnext_avg_yolo['mAP']:.1f}%",
         '✓ ConvNeXt' if convnext_avg_yolo['mAP'] > resnet_avg_yolo['mAP'] else '✓ ResNet'],
        ['Parameters', '26.13M', '28.31M', '✓ ResNet'],
        ['Training Time', '~19 min', '~23 min', '✓ ResNet'],
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    for i in range(1, 6):
        cell = table[(i, 3)]
        if '✓ ConvNeXt' in cell.get_text().get_text():
            cell.set_facecolor('#d5f4e6')
        elif '✓ ResNet' in cell.get_text().get_text():
            cell.set_facecolor('#fadbd8')
    
    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Step-9 Lite: Backbone Comparison (Fold 0, Combined Loss)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = figures_dir / "step9_backbone_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("Step-9 Lite Complete!")
    print("="*70)
    print(f"\n✅ SELECTED BEST BACKBONE: {best_backbone.upper()}")
    print(f"\nNext Steps:")
    print(f"  1. Use '{best_backbone}' for folds 1-4 training (Step-10)")
    print(f"  2. Compute 5-fold mean ± std for final results")
    print(f"  3. Proceed to Protocols B/C/D with best model")
    print("="*70)
    
    return best_backbone

if __name__ == "__main__":
    best_backbone = compare_backbones()
