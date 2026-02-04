import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Literal
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from models.reid_model import build_reid_model
from losses.triplet_loss import TripletLoss
from losses.supcon_loss import SupConLoss
from losses.combined_loss import CombinedLoss
from samplers.pk_sampler import PKBatchSampler
from transforms import get_train_transforms, get_eval_transforms
from cropper import CowCropper, load_yolo_predictions
import cv2


class ReIDDataset(Dataset):
    def __init__(self, csv_path, base_dir, transform=None, 
                 use_yolo_crop=True, yolo_preds_path=None):
        self.df = pd.read_csv(csv_path, dtype={'cow_id': str})
        self.base_dir = base_dir
        self.transform = transform
        self.use_yolo_crop = use_yolo_crop
        
        self.cropper = CowCropper(target_size=(224, 224), padding=0.15)
        
        if use_yolo_crop and yolo_preds_path:
            self.yolo_preds = load_yolo_predictions(yolo_preds_path)
        else:
            self.yolo_preds = None
        
        unique_ids = sorted(self.df['cow_id'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_ids)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[cow_id] for cow_id in self.df['cow_id']]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_dir / row['filepath']
        cow_id_str = row['cow_id']
        label = self.label_to_idx[cow_id_str]
        
        if self.use_yolo_crop and self.yolo_preds:
            filename = img_path.name
            pred = self.yolo_preds.get(filename)
            cropped = self.cropper.crop_from_yolo_prediction(img_path, pred) if pred else None
        else:
            bbox_path = self.base_dir / row['bbox_txt_path']
            cropped = self.cropper.crop_from_gt_bbox(img_path, bbox_path)
        
        if cropped is None:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cropped = cv2.resize(img, (224, 224))
        
        from PIL import Image
        cropped_pil = Image.fromarray(cropped)
        
        if self.transform:
            cropped_pil = self.transform(cropped_pil)
        
        return cropped_pil, label, cow_id_str


def train_one_epoch(model, train_loader, criterion, optimizer, device, 
                    epoch, use_amp=True, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = GradScaler() if use_amp else None
    
    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                embeddings = model(images)
                if isinstance(criterion, CombinedLoss):
                    loss, _ = criterion(embeddings, labels)
                else:
                    loss = criterion(embeddings, labels)
            
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            if isinstance(criterion, CombinedLoss):
                loss, _ = criterion(embeddings, labels)
            else:
                loss = criterion(embeddings, labels)
            
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, val_loader, device):
    model.eval()
    all_embeddings, all_labels, all_cow_ids = [], [], []
    
    with torch.no_grad():
        for images, labels, cow_ids in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.tolist())
            all_cow_ids.extend(cow_ids)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = np.array(all_labels)
    all_cow_ids = np.array(all_cow_ids)
    
    unique_ids = np.unique(all_cow_ids)
    top1_scores, top5_scores, ap_scores = [], [], []
    
    for cow_id in unique_ids:
        mask = all_cow_ids == cow_id
        id_embeddings = all_embeddings[mask]
        id_label = all_labels[mask][0]
        
        if len(id_embeddings) != 4:
            continue
        
        for query_idx in range(4):
            gallery_indices = [i for i in range(4) if i != query_idx]
            query_emb = id_embeddings[query_idx:query_idx+1]
            gallery_id_embs = id_embeddings[gallery_indices]
            
            other_mask = all_cow_ids != cow_id
            other_embeddings = all_embeddings[other_mask]
            other_labels = all_labels[other_mask]
            
            gallery_embs = torch.cat([gallery_id_embs, other_embeddings], dim=0)
            gallery_labels = np.concatenate([[id_label] * 3, other_labels])
            
            similarities = torch.matmul(query_emb, gallery_embs.T).squeeze(0)
            ranked_indices = torch.argsort(similarities, descending=True).numpy()
            ranked_labels = gallery_labels[ranked_indices]
            
            top1 = 1.0 if ranked_labels[0] == id_label else 0.0
            top5 = 1.0 if id_label in ranked_labels[:5] else 0.0
            top1_scores.append(top1)
            top5_scores.append(top5)
            
            relevant_mask = ranked_labels == id_label
            if relevant_mask.sum() > 0:
                precisions = np.cumsum(relevant_mask) / (np.arange(len(relevant_mask)) + 1)
                ap = (precisions * relevant_mask).sum() / relevant_mask.sum()
                ap_scores.append(ap)
    
    return {
        'top1': np.mean(top1_scores) * 100,
        'top5': np.mean(top5_scores) * 100,
        'mAP': np.mean(ap_scores) * 100
    }


def train_reid(fold=0, backbone="resnet50", loss_method="combined", epochs=80,
               lr_head=3e-4, lr_backbone=1e-4, weight_decay=3e-4, patience=12,
               use_amp=True, grad_clip=1.0, device="cuda"):
    base_dir = Path(r"d:\identification")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print(f"Training ReID Model - Fold {fold}")
    print(f"Backbone: {backbone}, Loss: {loss_method}, Device: {device}")
    
    train_csv = base_dir / "splits" / f"fold{fold}_train.csv"
    val_csv = base_dir / "splits" / f"fold{fold}_val.csv"
    yolo_preds_json = base_dir / "outputs" / "yolo_preds" / "fold0_main.json"
    
    train_dataset = ReIDDataset(train_csv, base_dir, get_train_transforms(),
                                use_yolo_crop=True, yolo_preds_path=yolo_preds_json)
    val_dataset = ReIDDataset(val_csv, base_dir, get_eval_transforms(),
                              use_yolo_crop=True, yolo_preds_path=yolo_preds_json)
    
    print(f"Train: {len(train_dataset)} images, {len(train_dataset.label_to_idx)} IDs")
    print(f"Val: {len(val_dataset)} images, {len(val_dataset.label_to_idx)} IDs")
    
    pk_sampler = PKBatchSampler(train_dataset, p=16, k=4)
    train_loader = DataLoader(train_dataset, batch_sampler=pk_sampler, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    num_train_classes = len(train_dataset.label_to_idx)
    model = build_reid_model(backbone, embed_dim=512, dropout=0.2, pretrained=True)
    model = model.to(device)
    
    if loss_method == "triplet":
        criterion = TripletLoss(margin=0.30)
    elif loss_method == "supcon":
        criterion = SupConLoss(temperature=0.07)
    elif loss_method == "combined":
        criterion = CombinedLoss(num_train_classes, 512, 0.07, 0.30, 30.0, 1.0, 1.0)
    criterion = criterion.to(device)
    
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_head, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler() if use_amp else None
    
    best_top1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_top1': [], 'val_top5': [], 'val_mAP': [], 'lr': []}
    
    for epoch in range(1, epochs + 1):
        if epoch == 6:
            print(f"Epoch {epoch}: Unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': lr_backbone},
                {'params': model.embedding_head.parameters(), 'lr': lr_head}
            ], weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - 5, eta_min=1e-6)
        
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    embeddings = model(images)
                    if loss_method == "combined":
                        loss, _ = criterion(embeddings, labels)
                    else:
                        loss = criterion(embeddings, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                embeddings = model(images)
                if loss_method == "combined":
                    loss, _ = criterion(embeddings, labels)
                else:
                    loss = criterion(embeddings, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_top1'].append(val_metrics['top1'])
        history['val_top5'].append(val_metrics['top5'])
        history['val_mAP'].append(val_metrics['mAP'])
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Top1={val_metrics['top1']:.2f}%, "
              f"Top5={val_metrics['top5']:.2f}%, mAP={val_metrics['mAP']:.2f}%")
        
        if val_metrics['top1'] > best_top1:
            best_top1 = val_metrics['top1']
            patience_counter = 0
            
            checkpoints_dir = base_dir / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoints_dir / f"reid_fold{fold}_{backbone}_{loss_method}_best.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_top1': best_top1,
                'history': history,
                'config': {
                    'fold': fold,
                    'backbone': backbone,
                    'loss_method': loss_method,
                    'lr_head': lr_head,
                    'lr_backbone': lr_backbone,
                    'weight_decay': weight_decay
                }
            }, checkpoint_path)
            
            print(f"  New best: {best_top1:.2f}%")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}, best Top-1: {best_top1:.2f}%")
                break
    
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['train_loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['val_top1'], color='green')
    axes[0, 1].axhline(y=best_top1, color='red', linestyle='--')
    axes[0, 1].set_title('Top-1 Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['val_top5'], label='Top-5')
    axes[1, 0].plot(history['val_mAP'], label='mAP')
    axes[1, 0].set_title('Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Fold {fold} - {backbone} - {loss_method}', fontsize=16)
    plt.tight_layout()
    
    curve_path = figures_dir / f"learning_curves_fold{fold}_{backbone}_{loss_method}.png"
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"Saved curves to: {curve_path}")
    
    print(f"\nTraining complete! Best Top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='resnet50', 
                       choices=['resnet50', 'convnext_tiny'])
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['triplet', 'supcon', 'combined'])
    parser.add_argument('--lr_head', type=float, default=3e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train_reid(fold=args.fold, backbone=args.backbone, loss_method=args.loss,
               epochs=args.epochs, lr_head=args.lr_head, lr_backbone=args.lr_backbone,
               weight_decay=args.weight_decay, patience=args.patience,
               use_amp=True, grad_clip=1.0, device=args.device)
