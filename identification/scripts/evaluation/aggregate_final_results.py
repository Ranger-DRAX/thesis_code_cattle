import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'results'
    
    resnet50_path = results_dir / 'final_protocolA_5fold_resnet50.csv'
    convnext_path = results_dir / 'final_protocolA_5fold_convnext_tiny.csv'
    
    if not resnet50_path.exists():
        print(f"Error: ResNet-50 results not found: {resnet50_path}")
        print("Run: python train_5fold_final.py --backbone resnet50")
        return
    
    if not convnext_path.exists():
        print(f"Error: ConvNeXt-Tiny results not found: {convnext_path}")
        print("Run: python train_5fold_final.py --backbone convnext_tiny")
        return
    
    df_resnet = pd.read_csv(resnet50_path)
    df_convnext = pd.read_csv(convnext_path)
    
    df_combined = pd.concat([df_resnet, df_convnext], ignore_index=True)
    
    output_path = results_dir / 'final_protocolA_mean_std.csv'
    df_combined.to_csv(output_path, index=False)
    
    print("="*70)
    print("Final Protocol A Results - 5-Fold Cross-Validation")
    print("="*70)
    
    print("\n" + "="*70)
    print("Table: GT vs YOLO Crop Performance")
    print("="*70)
    print(f"{'Backbone':<15} {'Crop':<8} {'Top-1 (%)':<15} {'Top-5 (%)':<15} {'mAP (%)':<15}")
    print("-"*70)
    
    for backbone in ['resnet50', 'convnext_tiny']:
        df_bb = df_combined[df_combined['backbone'] == backbone]
        
        for crop_type in ['YOLO', 'GT']:
            df_crop = df_bb[df_bb['crop_type'] == crop_type]
            
            mean_row = df_crop[df_crop['fold'] == 'mean'].iloc[0]
            std_row = df_crop[df_crop['fold'] == 'std'].iloc[0]
            
            top1_str = f"{mean_row['top1']:.2f} ± {std_row['top1']:.2f}"
            top5_str = f"{mean_row['top5']:.2f} ± {std_row['top5']:.2f}"
            map_str = f"{mean_row['mAP']:.2f} ± {std_row['mAP']:.2f}"
            
            print(f"{backbone:<15} {crop_type:<8} {top1_str:<15} {top5_str:<15} {map_str:<15}")
        
        print("-"*70)
    
    print("\n✓ Saved combined results to:", output_path)
    print("="*70)

if __name__ == "__main__":
    main()
