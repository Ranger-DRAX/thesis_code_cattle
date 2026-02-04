import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent))

from models.reid_model import build_reid_model
from train import ReIDDataset
from transforms import get_eval_transforms

def extract_embeddings(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_cow_ids = []
    all_filenames = []
    
    with torch.no_grad():
        for images, labels, cow_ids in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.tolist())
            all_cow_ids.extend(cow_ids)
    
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    
    return embeddings, labels, all_cow_ids, all_filenames

def evaluate_single_round(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_prototypes: np.ndarray,
    gallery_labels: np.ndarray
) -> Dict[str, float]:
    """
    query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12)
    gallery_prototypes = gallery_prototypes / (np.linalg.norm(gallery_prototypes, axis=1, keepdims=True) + 1e-12)
    
    similarities = np.dot(query_embeddings, gallery_prototypes.T)  # (N_query, N_gallery)
    
    top1_correct = 0
    top5_correct = 0
    ap_scores = []
    
    for i in range(len(query_labels)):
        query_label = query_labels[i]
        query_sims = similarities[i]  # (N_gallery,)
        
        ranked_indices = np.argsort(query_sims)[::-1]
        ranked_labels = gallery_labels[ranked_indices]
        
        if ranked_labels[0] == query_label:
            top1_correct += 1
        
        if query_label in ranked_labels[:5]:
            top5_correct += 1
        
        relevant_mask = (ranked_labels == query_label)
        if relevant_mask.sum() > 0:
            precisions = np.cumsum(relevant_mask) / (np.arange(len(relevant_mask)) + 1)
            ap = (precisions * relevant_mask).sum() / relevant_mask.sum()
            ap_scores.append(ap)
    
    metrics = {
        'top1': (top1_correct / len(query_labels)) * 100,
        'top5': (top5_correct / len(query_labels)) * 100,
        'mAP': np.mean(ap_scores) * 100 if ap_scores else 0.0
    }
    
    return metrics

def evaluate_protocol_a(
    model: nn.Module,
    test_csv: Path,
    base_dir: Path,
    device: torch.device,
    use_yolo_crop: bool = True,
    yolo_preds_path: Path = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    print("="*70)
    print("Protocol A: Within-ID Rotation Evaluation")
    print("="*70)
    
    eval_transforms = get_eval_transforms()
    test_dataset = ReIDDataset(
        test_csv, base_dir, eval_transforms,
        use_yolo_crop=use_yolo_crop,
        yolo_preds_path=yolo_preds_path
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nTest set: {len(test_dataset)} images, {len(test_dataset.label_to_idx)} IDs")
    print(f"Crop mode: {'YOLO (end-to-end)' if use_yolo_crop else 'GT (oracle)'}")
    
    print("\n[1] Extracting Embeddings")
    embeddings, labels, cow_ids, _ = extract_embeddings(model, test_loader, device)
    print(f"  Extracted {len(embeddings)} embeddings")
    
    df_test = pd.read_csv(test_csv, dtype={'cow_id': str})
    
    unique_cow_ids = sorted(test_dataset.label_to_idx.keys())
    print(f"\n[2] Grouping by Cow ID")
    print(f"  Unique IDs: {len(unique_cow_ids)}")
    
    cow_to_views = {}
    
    for idx, cow_id in enumerate(cow_ids):
        row = df_test.iloc[idx]
        view = row['view']  # front, back, left, right
        
        if cow_id not in cow_to_views:
            cow_to_views[cow_id] = {}
        
        cow_to_views[cow_id][view] = {
            'embedding': embeddings[idx],
            'label': labels[idx]
        }
    
    valid_cows = [cow_id for cow_id, views in cow_to_views.items() if len(views) == 4]
    print(f"  Valid cows (4 views): {len(valid_cows)}")
    
    if len(valid_cows) < len(unique_cow_ids):
        print(f"  Warning: {len(unique_cow_ids) - len(valid_cows)} cows excluded (missing views)")
    
    rounds = [
        {'name': 'Round-1', 'enroll': ['back', 'left', 'right'], 'query': 'front'},
        {'name': 'Round-2', 'enroll': ['front', 'left', 'right'], 'query': 'back'},
        {'name': 'Round-3', 'enroll': ['front', 'back', 'right'], 'query': 'left'},
        {'name': 'Round-4', 'enroll': ['front', 'back', 'left'], 'query': 'right'}
    ]
    
    print(f"\n[3] Evaluating 4 Rounds")
    print("-"*70)
    
    round_results = []
    
    for round_info in rounds:
        round_name = round_info['name']
        enroll_views = round_info['enroll']
        query_view = round_info['query']
        
        print(f"\n{round_name}:")
        print(f"  Enroll: {enroll_views}")
        print(f"  Query: {query_view}")
        
        query_embeddings = []
        query_labels = []
        gallery_prototypes = []
        gallery_labels = []
        
        for cow_id in valid_cows:
            views = cow_to_views[cow_id]
            label = views[query_view]['label']
            
            query_emb = views[query_view]['embedding']
            query_embeddings.append(query_emb)
            query_labels.append(label)
            
            enroll_embs = [views[v]['embedding'] for v in enroll_views]
            prototype = np.mean(enroll_embs, axis=0)
            gallery_prototypes.append(prototype)
            gallery_labels.append(label)
        
        query_embeddings = np.array(query_embeddings)
        query_labels = np.array(query_labels)
        gallery_prototypes = np.array(gallery_prototypes)
        gallery_labels = np.array(gallery_labels)
        
        metrics = evaluate_single_round(
            query_embeddings, query_labels,
            gallery_prototypes, gallery_labels
        )
        
        print(f"  Top-1: {metrics['top1']:.2f}%")
        print(f"  Top-5: {metrics['top5']:.2f}%")
        print(f"  mAP: {metrics['mAP']:.2f}%")
        
        round_results.append({
            'round': round_name,
            'enroll_views': '+'.join(enroll_views),
            'query_view': query_view,
            'num_queries': len(query_labels),
            'top1': metrics['top1'],
            'top5': metrics['top5'],
            'mAP': metrics['mAP']
        })
    
    avg_top1 = np.mean([r['top1'] for r in round_results])
    avg_top5 = np.mean([r['top5'] for r in round_results])
    avg_mAP = np.mean([r['mAP'] for r in round_results])
    
    round_results.append({
        'round': 'Average',
        'enroll_views': 'All rotations',
        'query_view': 'All views',
        'num_queries': len(valid_cows) * 4,
        'top1': avg_top1,
        'top5': avg_top5,
        'mAP': avg_mAP
    })
    
    print(f"\n{'='*70}")
    print("Protocol A Results Summary")
    print(f"{'='*70}")
    print(f"Average Top-1: {avg_top1:.2f}%")
    print(f"Average Top-5: {avg_top5:.2f}%")
    print(f"Average mAP: {avg_mAP:.2f}%")
    
    results_df = pd.DataFrame(round_results)
    avg_metrics = {
        'top1': avg_top1,
        'top5': avg_top5,
        'mAP': avg_mAP
    }
    
    return results_df, avg_metrics

def main():
    """Run Protocol A evaluation on fold0 test set."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Protocol A')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4)')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                       choices=['resnet50', 'convnext_tiny'],
                       help='Backbone architecture')
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['triplet', 'supcon', 'combined'],
                       help='Loss function used during training')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (auto-detected if not specified)')
    args = parser.parse_args()
    
    base_dir = Path(r"d:\identification")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("Step 8: Protocol A Evaluation")
    print("="*70)
    
    fold = args.fold
    backbone = args.backbone
    loss_method = args.loss
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = base_dir / "checkpoints" / f"reid_fold{fold}_{backbone}_{loss_method}_best.pt"
    
    test_csv = base_dir / "splits" / f"fold{fold}_test.csv"
    yolo_preds_json = base_dir / "outputs" / "yolo_preds" / f"fold{fold}_main.json"
    
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    print(f"\nConfiguration:")
    print(f"  Fold: {fold}")
    print(f"  Backbone: {backbone}")
    print(f"  Loss: {loss_method}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    
    print(f"\n[1] Loading Model")
    print("-"*70)
    
    model = build_reid_model(backbone, embed_dim=512, dropout=0.2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  ✓ Best validation Top-1: {checkpoint['best_top1']:.2f}%")
    
    print(f"\n[2] Evaluating with YOLO Crops (End-to-End)")
    print("="*70)
    
    results_yolo, avg_yolo = evaluate_protocol_a(
        model, test_csv, base_dir, device,
        use_yolo_crop=True,
        yolo_preds_path=yolo_preds_json
    )
    
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_path_yolo = results_dir / f"protocolA_fold{fold}_{backbone}_{loss_method}_yolo.csv"
    results_yolo.to_csv(output_path_yolo, index=False)
    print(f"\n✓ Saved YOLO results to: {output_path_yolo}")
    
    print(f"\n[3] Evaluating with GT Crops (Oracle)")
    print("="*70)
    
    results_gt, avg_gt = evaluate_protocol_a(
        model, test_csv, base_dir, device,
        use_yolo_crop=False,
        yolo_preds_path=None
    )
    
    output_path_gt = results_dir / f"protocolA_fold{fold}_{backbone}_{loss_method}_gt.csv"
    results_gt.to_csv(output_path_gt, index=False)
    print(f"\n✓ Saved GT results to: {output_path_gt}")
    
    print(f"\n{'='*70}")
    print("YOLO vs GT Comparison")
    print(f"{'='*70}")
    print(f"\n{'Metric':<15} {'YOLO (End-to-End)':<20} {'GT (Oracle)':<20} {'Gap':<15}")
    print("-"*70)
    print(f"{'Top-1':<15} {avg_yolo['top1']:>18.2f}% {avg_gt['top1']:>18.2f}% {avg_gt['top1']-avg_yolo['top1']:>13.2f}%")
    print(f"{'Top-5':<15} {avg_yolo['top5']:>18.2f}% {avg_gt['top5']:>18.2f}% {avg_gt['top5']-avg_yolo['top5']:>13.2f}%")
    print(f"{'mAP':<15} {avg_yolo['mAP']:>18.2f}% {avg_gt['mAP']:>18.2f}% {avg_gt['mAP']-avg_yolo['mAP']:>13.2f}%")
    
    print(f"\n{'='*70}")
    print("Protocol A Evaluation Complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
