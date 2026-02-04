import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import yaml
from itertools import product
from tqdm import tqdm
import sys

from models.reid_model import build_reid_model
from losses.triplet_loss import TripletLoss
from losses.supcon_loss import SupConLoss
from losses.combined_loss import CombinedLoss
from samplers.pk_sampler import PKBatchSampler
from transforms import get_train_transforms, get_eval_transforms
from train import ReIDDataset, train_one_epoch, validate

def get_mild_transforms():
    """Mild augmentation: reduced ColorJitter and RandomErasing."""
    import torchvision.transforms as T
    from PIL import Image
    
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.1),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

def create_loss_function(method: str, num_classes: int, embed_dim: int):
    """Create loss function based on method name."""
    if method == 'triplet':
        return TripletLoss(margin=0.30)
    elif method == 'supcon':
        return SupConLoss(temperature=0.07)
    elif method == 'supcon+arcface':
        return CombinedLoss(
            num_classes=num_classes,
            embedding_dim=embed_dim,
            supcon_temperature=0.07,
            arcface_margin=0.30,
            arcface_scale=30.0,
            weight_supcon=1.0,
            weight_arcface=1.0
        )
    else:
        raise ValueError(f"Unknown method: {method}")

def train_one_config(config: dict, fold: int = 0):
    """Train one configuration and return validation metrics."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = Path(__file__).parent
    
    train_csv = base_dir / 'splits' / f'fold{fold}_train.csv'
    val_csv = base_dir / 'splits' / f'fold{fold}_val.csv'
    yolo_preds_json = base_dir / 'outputs' / 'yolo_preds' / 'fold0_main.json'
    
    if config['aug_strength'] == 'mild':
        train_transform = get_mild_transforms()
    else:
        train_transform = get_train_transforms()
    
    val_transform = get_eval_transforms()
    
    train_dataset = ReIDDataset(
        train_csv, base_dir, train_transform,
        use_yolo_crop=True, yolo_preds_path=yolo_preds_json
    )
    val_dataset = ReIDDataset(
        val_csv, base_dir, val_transform,
        use_yolo_crop=True, yolo_preds_path=yolo_preds_json
    )
    
    num_classes = len(train_dataset.label_to_idx)
    
    pk_sampler = PKBatchSampler(train_dataset, p=16, k=4)
    train_loader = DataLoader(train_dataset, batch_sampler=pk_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = build_reid_model(config['backbone'], embed_dim=512, dropout=0.2, pretrained=True)
    model = model.to(device)
    
    loss_fn = create_loss_function(config['method'], num_classes, embed_dim=512)
    loss_fn = loss_fn.to(device)
    
    params = [
        {'params': model.backbone.parameters(), 'lr': config['lr_backbone']},
        {'params': model.embedding_head.parameters(), 'lr': config['lr_head']}
    ]
    
    if isinstance(loss_fn, CombinedLoss):
        params.append({'params': loss_fn.parameters(), 'lr': config['lr_head']})
    
    optimizer = torch.optim.AdamW(params, weight_decay=config['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
    
    best_top1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    model.freeze_backbone()
    
    for epoch in range(1, config['epochs'] + 1):
        if epoch == 6:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': config['lr_backbone']},
                {'params': model.embedding_head.parameters(), 'lr': config['lr_head']}
            ], weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs'] - 5, eta_min=1e-6
            )
        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        
        val_metrics = validate(model, val_loader, device)
        
        scheduler.step()
        
        if val_metrics['top1'] > best_top1:
            best_top1 = val_metrics['top1']
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return {
        'top1': best_top1,
        'best_epoch': best_epoch,
    }

def main():
    """Main tuning loop."""
    
    print("="*70)
    print("Step-9 Phase-1: Hyperparameter Tuning - ConvNeXt-Tiny")
    print("="*70)
    
    tuning_space = {
        'backbone': ['convnext_tiny'],
        'method': ['supcon+arcface'],  # Only best method
        'lr_backbone': [3e-5, 1e-4],   # Test both backbone LRs
        'lr_head': [3e-4],             # Fixed to best value
        'weight_decay': [1e-4, 3e-4],  # Test regularization
        'aug_strength': ['default', 'mild'],  # Test augmentation
    }
    
    fixed_params = {
        'epochs': 25,      # Reduced from 80
        'patience': 6,     # Reduced from 12
    }
    
    keys = list(tuning_space.keys())
    values = list(tuning_space.values())
    all_configs = [dict(zip(keys, combo)) for combo in product(*values)]
    
    print(f"\nTotal configurations: {len(all_configs)}")
    print(f"Estimated time: {len(all_configs) * 10 / 60:.1f} hours (assuming 10 min per config)")
    print("Option B-2: Fast tuning with best method + key hyperparameters\n")
    print("\nStarting grid search...\n")
    
    results = []
    
    for i, config in enumerate(all_configs, 1):
        print(f"\n{'='*70}")
        print(f"Configuration {i}/{len(all_configs)}")
        print(f"{'='*70}")
        print(f"Method: {config['method']}")
        print(f"LR Backbone: {config['lr_backbone']:.0e}, LR Head: {config['lr_head']:.0e}")
        print(f"Weight Decay: {config['weight_decay']:.0e}")
        print(f"Aug Strength: {config['aug_strength']}")
        
        full_config = {**config, **fixed_params}
        
        try:
            metrics = train_one_config(full_config, fold=0)
            
            result = {
                'config_id': i,
                'backbone': config['backbone'],
                'method': config['method'],
                'lr_backbone': config['lr_backbone'],
                'lr_head': config['lr_head'],
                'weight_decay': config['weight_decay'],
                'aug_strength': config['aug_strength'],
                'val_top1': metrics['top1'],
                'best_epoch': metrics['best_epoch'],
                'status': 'success'
            }
            
            print(f"\n✓ Best Val Top-1: {metrics['top1']:.2f}% (epoch {metrics['best_epoch']})")
            
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            result = {
                'config_id': i,
                'backbone': config['backbone'],
                'method': config['method'],
                'lr_backbone': config['lr_backbone'],
                'lr_head': config['lr_head'],
                'weight_decay': config['weight_decay'],
                'aug_strength': config['aug_strength'],
                'val_top1': 0.0,
                'best_epoch': 0,
                'status': f'failed: {str(e)[:50]}'
            }
        
        results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_top1', ascending=False)
        
        base_dir = Path(__file__).parent
        results_dir = base_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results_df.to_csv(results_dir / 'tuning_convnext_tiny_fold0.csv', index=False)
        
        print(f"\n✓ Saved intermediate results")
        print(f"Current best: {results_df.iloc[0]['method']} - "
              f"{results_df.iloc[0]['val_top1']:.2f}%")
    
    print("\n" + "="*70)
    print("TUNING COMPLETE - ConvNeXt-Tiny")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_top1', ascending=False)
    
    results_df.to_csv(results_dir / 'tuning_convnext_tiny_fold0.csv', index=False)
    
    print("\nTop 5 Configurations:")
    print(results_df.head(5).to_string(index=False))
    
    best_config = results_df.iloc[0]
    
    config_dict = {
        'backbone': best_config['backbone'],
        'method': best_config['method'],
        'lr_backbone': float(best_config['lr_backbone']),
        'lr_head': float(best_config['lr_head']),
        'weight_decay': float(best_config['weight_decay']),
        'aug_strength': best_config['aug_strength'],
        'embed_dim': 512,
        'dropout': 0.2,
        'epochs': 80,
        'patience': 12,
        'batch_size': 64,
        'P': 16,
        'K': 4,
        'val_top1': float(best_config['val_top1']),
        'best_epoch': int(best_config['best_epoch']),
    }
    
    configs_dir = base_dir / 'configs'
    configs_dir.mkdir(exist_ok=True)
    
    with open(configs_dir / 'best_convnext_tiny.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"\n✓ Saved best config to: configs/best_convnext_tiny.yaml")
    print(f"✓ Best Val Top-1: {best_config['val_top1']:.2f}%")
    print(f"✓ Method: {best_config['method']}")
    print(f"✓ LR Backbone: {best_config['lr_backbone']:.0e}")
    print(f"✓ LR Head: {best_config['lr_head']:.0e}")
    print(f"✓ Weight Decay: {best_config['weight_decay']:.0e}")
    print(f"✓ Aug Strength: {best_config['aug_strength']}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
