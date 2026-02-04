import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from models.reid_model import build_reid_model
from transforms import get_eval_transforms
from ultralytics import YOLO

class YOLOCropper:
    """Simple YOLO-based cropper for cow detection."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def get_bbox(self, image) -> list:
        """Get bbox from image using YOLO. Returns [x1, y1, x2, y2] or None."""
        import numpy as np
        
        if hasattr(image, 'convert'):
            img_np = np.array(image)
        else:
            img_np = image
        
        results = self.model(img_np, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            
            if confs.max() >= self.conf_threshold:
                best_idx = confs.argmax()
                bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
                return bbox
        
        return None

def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained ReID model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = build_reid_model(
        config['backbone'],
        embed_dim=config['embed_dim'],
        dropout=config['dropout'],
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config

def load_yolo_bbox(bbox_txt_path: Path, img_width: int, img_height: int):
    """Load YOLO format bbox and convert to pixel coordinates."""
    if not bbox_txt_path.exists():
        return None
    
    with open(bbox_txt_path, 'r') as f:
        line = f.readline().strip()
        if not line:
            return None
        
        parts = line.split()
        if len(parts) < 5:
            return None
        
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        w = float(parts[3]) * img_width
        h = float(parts[4]) * img_height
        
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        return [x1, y1, x2, y2]

def crop_with_padding(image: Image.Image, bbox: list, padding: float = 0.15) -> Image.Image:
    """Crop image with optional padding."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(image.width, x2 + pad_w)
    y2 = min(image.height, y2 + pad_h)
    
    return image.crop((x1, y1, x2, y2))

def build_gallery_prototypes(model, device, main_images_dir: Path, 
                             main_bboxes_dir: Path, cow_ids: set,
                             transform, use_yolo: bool = False,
                             yolo_cropper=None) -> tuple:
    """Build 4-view prototypes for gallery cows."""
    prototypes = {}
    gallery_ids = []
    
    view_names = {
        'F': ['front', 'Front', 'FRONT', 'F'],
        'B': ['back', 'Back', 'BACK', 'B'],
        'L': ['left', 'Left', 'LEFT', 'L'],
        'R': ['right', 'Right', 'RIGHT', 'R']
    }
    
    print(f"\nBuilding gallery prototypes for {len(cow_ids)} cows...")
    
    for cow_id in tqdm(sorted(cow_ids), desc="Gallery"):
        embeddings = []
        valid_views = 0
        
        cow_dir = main_images_dir / str(cow_id)
        if not cow_dir.exists():
            continue
        
        for view_key, view_variants in view_names.items():
            img_path = None
            
            for view_name in view_variants:
                for ext in ['.JPG', '.jpg', '.png', '.PNG', '.jpeg']:
                    candidates = [
                        cow_dir / f"{cow_id}_{view_name}{ext}",
                        cow_dir / f"{cow_id}{view_name}{ext}",
                        cow_dir / f"{view_name}{ext}",
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            img_path = candidate
                            break
                    if img_path:
                        break
                if img_path:
                    break
            
            if img_path is None:
                continue
            
            image = Image.open(img_path).convert('RGB')
            
            bbox = None
            if use_yolo and yolo_cropper is not None:
                bbox = yolo_cropper.get_bbox(image)
            else:
                for view_name in view_variants:
                    for ext in ['.txt']:
                        bbox_candidates = [
                            main_bboxes_dir / str(cow_id) / f"{cow_id}_{view_name}{ext}",
                            main_bboxes_dir / f"{cow_id}_{view_name}{ext}",
                            main_bboxes_dir / str(cow_id) / f"{view_name}{ext}",
                        ]
                        for bbox_txt in bbox_candidates:
                            if bbox_txt.exists():
                                bbox = load_yolo_bbox(bbox_txt, image.width, image.height)
                                if bbox:
                                    break
                        if bbox:
                            break
                    if bbox:
                        break
            
            if bbox is not None:
                image = crop_with_padding(image, bbox, padding=0.15)
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = model(img_tensor)
                embeddings.append(emb.cpu())
                valid_views += 1
        
        if valid_views >= 2:  # Need at least 2 views
            avg_emb = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)
            prototype = F.normalize(avg_emb, p=2, dim=1)
            prototypes[cow_id] = prototype
            gallery_ids.append(cow_id)
    
    print(f"  Built {len(gallery_ids)} valid prototypes")
    
    if len(gallery_ids) == 0:
        raise ValueError("No valid gallery prototypes could be built!")
    
    proto_matrix = torch.cat([prototypes[cid] for cid in gallery_ids], dim=0)
    
    return proto_matrix, gallery_ids

def evaluate_protocol(model, device, query_dir: Path, labels_csv: Path,
                      bboxes_dir: Path, gallery_prototypes: torch.Tensor,
                      gallery_ids: list, transform, use_yolo: bool = False,
                      yolo_cropper=None) -> dict:
    """Evaluate a protocol with GT or YOLO crops."""
    
    df = pd.read_csv(labels_csv)
    
    if 'filename' in df.columns:
        filename_col = 'filename'
    else:
        filename_col = df.columns[0]
    
    if 'cow_id' in df.columns:
        id_col = 'cow_id'
    else:
        id_col = df.columns[1]
    
    results = []
    yolo_failures = 0
    total_queries = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating", leave=False):
        filename = str(row[filename_col])
        true_id = int(row[id_col])
        
        if true_id not in gallery_ids:
            continue
        
        total_queries += 1
        
        img_path = None
        for ext in ['.JPG', '.jpg', '.png', '.PNG']:
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            candidate = query_dir / f"{base_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            img_path = query_dir / filename
            if not img_path.exists():
                continue
        
        image = Image.open(img_path).convert('RGB')
        
        bbox = None
        if use_yolo and yolo_cropper is not None:
            bbox = yolo_cropper.get_bbox(image)
            if bbox is None:
                yolo_failures += 1
        else:
            bbox_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            bbox_txt = bboxes_dir / f"{bbox_name}.txt"
            if bbox_txt.exists():
                bbox = load_yolo_bbox(bbox_txt, image.width, image.height)
        
        if bbox is not None:
            image = crop_with_padding(image, bbox, padding=0.15)
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_emb = model(img_tensor)
            query_emb = F.normalize(query_emb, p=2, dim=1)
        
        similarities = torch.mm(query_emb, gallery_prototypes.to(device).t()).squeeze(0)
        
        sorted_indices = torch.argsort(similarities, descending=True)
        ranked_ids = [gallery_ids[i] for i in sorted_indices.tolist()]
        
        if true_id in ranked_ids:
            rank = ranked_ids.index(true_id)
        else:
            rank = len(ranked_ids)
        
        results.append({
            'filename': filename,
            'true_id': true_id,
            'rank': rank,
            'top1': 1 if rank == 0 else 0,
            'top5': 1 if rank < 5 else 0,
            'ap': 1.0 / (rank + 1) if rank < len(ranked_ids) else 0.0,
            'similarity': similarities[gallery_ids.index(true_id)].item() if true_id in gallery_ids else 0.0,
            'top5_ids': ranked_ids[:5]
        })
    
    if len(results) > 0:
        top1 = np.mean([r['top1'] for r in results]) * 100
        top5 = np.mean([r['top5'] for r in results]) * 100
        mAP = np.mean([r['ap'] for r in results]) * 100
    else:
        top1, top5, mAP = 0.0, 0.0, 0.0
    
    yolo_fail_rate = (yolo_failures / total_queries * 100) if total_queries > 0 else 0.0
    
    return {
        'top1': top1,
        'top5': top5,
        'mAP': mAP,
        'num_queries': total_queries,
        'yolo_fail_rate': yolo_fail_rate,
        'results': results
    }

def create_retrieval_visualization(results: list, query_dir: Path, main_images_dir: Path,
                                   output_path: Path, protocol_name: str, num_examples: int = 6):
    """Create visualization of query + top-5 retrieved results."""
    
    successes = [r for r in results if r['rank'] == 0]
    failures = [r for r in results if r['rank'] > 0]
    
    np.random.seed(42)
    selected_success = np.random.choice(len(successes), min(3, len(successes)), replace=False) if successes else []
    selected_failure = np.random.choice(len(failures), min(3, len(failures)), replace=False) if failures else []
    
    examples = [successes[i] for i in selected_success] + [failures[i] for i in selected_failure]
    
    if len(examples) == 0:
        print(f"  No examples to visualize for {protocol_name}")
        return
    
    fig, axes = plt.subplots(len(examples), 6, figsize=(18, 3 * len(examples)))
    if len(examples) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{protocol_name} Retrieval Examples', fontsize=14, fontweight='bold')
    
    for row_idx, result in enumerate(examples):
        query_path = query_dir / result['filename']
        if not query_path.exists():
            for ext in ['.JPG', '.jpg', '.png']:
                base = result['filename'].rsplit('.', 1)[0]
                candidate = query_dir / f"{base}{ext}"
                if candidate.exists():
                    query_path = candidate
                    break
        
        if query_path.exists():
            query_img = Image.open(query_path).convert('RGB')
            axes[row_idx, 0].imshow(query_img)
        
        status = "✓ SUCCESS" if result['rank'] == 0 else f"✗ FAIL (rank={result['rank']+1})"
        axes[row_idx, 0].set_title(f"Query: ID {result['true_id']}\n{status}", fontsize=9)
        axes[row_idx, 0].axis('off')
        
        for col_idx, retrieved_id in enumerate(result['top5_ids'][:5]):
            ax = axes[row_idx, col_idx + 1]
            
            id_dir = main_images_dir / str(retrieved_id)
            if id_dir.exists():
                imgs = list(id_dir.glob("*.JPG")) + list(id_dir.glob("*.jpg"))
                if imgs:
                    retrieved_img = Image.open(imgs[0]).convert('RGB')
                    ax.imshow(retrieved_img)
            
            is_correct = retrieved_id == result['true_id']
            border_color = 'green' if is_correct else 'red'
            
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            
            ax.set_title(f"#{col_idx+1}: ID {retrieved_id}", fontsize=9,
                        color='green' if is_correct else 'black')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_comparison_plot(results_A: dict, results_B: dict, results_C: dict,
                          output_path: Path, backbone: str):
    """Create bar plot comparing Protocol A vs B vs C."""
    
    protocols = ['A-YOLO', 'A-GT', 'B-YOLO', 'B-GT', 'C-YOLO', 'C-GT']
    top1_values = [
        results_A.get('yolo_top1', 0), results_A.get('gt_top1', 0),
        results_B.get('yolo_top1', 0), results_B.get('gt_top1', 0),
        results_C.get('yolo_top1', 0), results_C.get('gt_top1', 0)
    ]
    top5_values = [
        results_A.get('yolo_top5', 0), results_A.get('gt_top5', 0),
        results_B.get('yolo_top5', 0), results_B.get('gt_top5', 0),
        results_C.get('yolo_top5', 0), results_C.get('gt_top5', 0)
    ]
    
    x = np.arange(len(protocols))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, top1_values, width, label='Top-1', color='steelblue')
    bars2 = ax.bar(x + width/2, top5_values, width, label='Top-5', color='coral')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Protocol A vs B vs C Comparison - {backbone.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(protocols)
    ax.legend()
    ax.set_ylim(0, 105)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Step-11: Protocol B & C Evaluation')
    parser.add_argument('--backbone', type=str, required=True,
                       choices=['resnet50', 'convnext_tiny'],
                       help='Backbone architecture')
    parser.add_argument('--fold', type=int, default=0,
                       help='Which fold checkpoint to use (default: 0)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(f"Step-11: Protocol B & C Evaluation - {args.backbone.upper()}")
    print("="*70)
    
    results_dir = base_dir / 'results' / args.backbone
    figures_dir = base_dir / 'figures' / args.backbone
    prototypes_dir = base_dir / 'prototypes' / args.backbone
    
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prototypes_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = base_dir / 'checkpoints' / f'reid_fold{args.fold}_{args.backbone}_supcon+arcface_best.pt'
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nLoading model from: {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    transform = get_eval_transforms()
    
    yolo_path = base_dir / 'checkpoints' / 'yolo_fold0_best.pt'
    if not yolo_path.exists():
        yolo_path = base_dir / 'yolo_fold0' / 'weights' / 'best.pt'
    
    yolo_cropper = None
    if yolo_path.exists():
        yolo_cropper = YOLOCropper(str(yolo_path))
        print(f"Loaded YOLO from: {yolo_path}")
    else:
        print("Warning: YOLO weights not found, YOLO evaluation will skip")
    
    main_images_dir = base_dir / 'main' / 'images'
    main_bboxes_dir = base_dir / 'main' / 'Annotated' / 'labels'
    
    protocolB_query_dir = base_dir / 'test_cross_angle' / 'images'
    protocolB_labels = base_dir / 'test_cross_angle' / 'labels.csv'
    protocolB_bboxes = base_dir / 'test_cross_angle' / 'bboxes_txt'
    
    protocolC_query_dir = base_dir / 'test_hard' / 'images'
    protocolC_labels = base_dir / 'test_hard' / 'labels.csv'
    protocolC_bboxes = base_dir / 'test_hard' / 'bboxes_txt'
    
    df_B = pd.read_csv(protocolB_labels)
    df_C = pd.read_csv(protocolC_labels)
    
    id_col_B = 'cow_id' if 'cow_id' in df_B.columns else df_B.columns[1]
    id_col_C = 'cow_id' if 'cow_id' in df_C.columns else df_C.columns[1]
    
    gallery_ids_B = set(df_B[id_col_B].astype(int).tolist())
    gallery_ids_C = set(df_C[id_col_C].astype(int).tolist())
    all_gallery_ids = gallery_ids_B | gallery_ids_C
    
    print(f"\nProtocol B: {len(df_B)} queries, {len(gallery_ids_B)} unique IDs")
    print(f"Protocol C: {len(df_C)} queries, {len(gallery_ids_C)} unique IDs")
    print(f"Total gallery IDs needed: {len(all_gallery_ids)}")
    
    print(f"\n{'='*70}")
    print("Step 11.1: Building Gallery Prototypes (GT crops)")
    print(f"{'='*70}")
    
    gallery_prototypes, gallery_id_list = build_gallery_prototypes(
        model, device, main_images_dir, main_bboxes_dir, all_gallery_ids,
        transform, use_yolo=False, yolo_cropper=None
    )
    
    np.save(prototypes_dir / 'protocolB_gallery.npy', gallery_prototypes.numpy())
    np.save(prototypes_dir / 'protocolC_gallery.npy', gallery_prototypes.numpy())
    with open(prototypes_dir / 'gallery_ids.json', 'w') as f:
        json.dump(gallery_id_list, f)
    print(f"Saved prototypes:")
    print(f"  {prototypes_dir / 'protocolB_gallery.npy'}")
    print(f"  {prototypes_dir / 'protocolC_gallery.npy'}")
    print(f"  {prototypes_dir / 'gallery_ids.json'}")
    
    print(f"\n{'='*70}")
    print("Step 11.2: Protocol B Evaluation (Cross-Angle)")
    print(f"{'='*70}")
    
    print("\nProtocol B-GT (Oracle crops)...")
    results_B_GT = evaluate_protocol(
        model, device, protocolB_query_dir, protocolB_labels, protocolB_bboxes,
        gallery_prototypes, gallery_id_list, transform,
        use_yolo=False, yolo_cropper=None
    )
    print(f"  Top-1: {results_B_GT['top1']:.2f}%, Top-5: {results_B_GT['top5']:.2f}%, mAP: {results_B_GT['mAP']:.2f}%")
    
    print("\nProtocol B-YOLO (Deployment crops)...")
    if yolo_cropper is not None:
        results_B_YOLO = evaluate_protocol(
            model, device, protocolB_query_dir, protocolB_labels, protocolB_bboxes,
            gallery_prototypes, gallery_id_list, transform,
            use_yolo=True, yolo_cropper=yolo_cropper
        )
        print(f"  Top-1: {results_B_YOLO['top1']:.2f}%, Top-5: {results_B_YOLO['top5']:.2f}%, mAP: {results_B_YOLO['mAP']:.2f}%")
        print(f"  YOLO fail rate: {results_B_YOLO['yolo_fail_rate']:.2f}%")
    else:
        results_B_YOLO = {'top1': 0, 'top5': 0, 'mAP': 0, 'num_queries': 0, 'yolo_fail_rate': 100, 'results': []}
    
    metrics_B = pd.DataFrame([
        {'run_type': 'GT', 'Top1': results_B_GT['top1'], 'Top5': results_B_GT['top5'],
         'mAP': results_B_GT['mAP'], 'num_queries': results_B_GT['num_queries'],
         'yolo_fail_rate': 0.0},
        {'run_type': 'YOLO', 'Top1': results_B_YOLO['top1'], 'Top5': results_B_YOLO['top5'],
         'mAP': results_B_YOLO['mAP'], 'num_queries': results_B_YOLO['num_queries'],
         'yolo_fail_rate': results_B_YOLO['yolo_fail_rate']}
    ])
    metrics_B.to_csv(results_dir / 'protocolB_metrics.csv', index=False)
    print(f"\nSaved: {results_dir / 'protocolB_metrics.csv'}")
    
    if len(results_B_GT['results']) > 0:
        create_retrieval_visualization(
            results_B_GT['results'], protocolB_query_dir, main_images_dir,
            figures_dir / 'retrieval_examples_B.png', 'Protocol B (GT)'
        )
    
    print(f"\n{'='*70}")
    print("Step 11.3: Protocol C Evaluation (Hard Cases)")
    print(f"{'='*70}")
    
    print("\nProtocol C-GT (Oracle crops)...")
    results_C_GT = evaluate_protocol(
        model, device, protocolC_query_dir, protocolC_labels, protocolC_bboxes,
        gallery_prototypes, gallery_id_list, transform,
        use_yolo=False, yolo_cropper=None
    )
    print(f"  Top-1: {results_C_GT['top1']:.2f}%, Top-5: {results_C_GT['top5']:.2f}%, mAP: {results_C_GT['mAP']:.2f}%")
    
    print("\nProtocol C-YOLO (Deployment crops)...")
    if yolo_cropper is not None:
        results_C_YOLO = evaluate_protocol(
            model, device, protocolC_query_dir, protocolC_labels, protocolC_bboxes,
            gallery_prototypes, gallery_id_list, transform,
            use_yolo=True, yolo_cropper=yolo_cropper
        )
        print(f"  Top-1: {results_C_YOLO['top1']:.2f}%, Top-5: {results_C_YOLO['top5']:.2f}%, mAP: {results_C_YOLO['mAP']:.2f}%")
        print(f"  YOLO fail rate: {results_C_YOLO['yolo_fail_rate']:.2f}%")
    else:
        results_C_YOLO = {'top1': 0, 'top5': 0, 'mAP': 0, 'num_queries': 0, 'yolo_fail_rate': 100, 'results': []}
    
    metrics_C = pd.DataFrame([
        {'run_type': 'GT', 'Top1': results_C_GT['top1'], 'Top5': results_C_GT['top5'],
         'mAP': results_C_GT['mAP'], 'num_queries': results_C_GT['num_queries'],
         'yolo_fail_rate': 0.0},
        {'run_type': 'YOLO', 'Top1': results_C_YOLO['top1'], 'Top5': results_C_YOLO['top5'],
         'mAP': results_C_YOLO['mAP'], 'num_queries': results_C_YOLO['num_queries'],
         'yolo_fail_rate': results_C_YOLO['yolo_fail_rate']}
    ])
    metrics_C.to_csv(results_dir / 'protocolC_metrics.csv', index=False)
    print(f"\nSaved: {results_dir / 'protocolC_metrics.csv'}")
    
    if len(results_C_GT['results']) > 0:
        create_retrieval_visualization(
            results_C_GT['results'], protocolC_query_dir, main_images_dir,
            figures_dir / 'retrieval_examples_C.png', 'Protocol C (GT)'
        )
    
    print(f"\n{'='*70}")
    print("Step 11.4: Creating Comparison Plot")
    print(f"{'='*70}")
    
    protocolA_path = base_dir / 'results' / f'final_protocolA_5fold_{args.backbone}.csv'
    if protocolA_path.exists():
        df_A = pd.read_csv(protocolA_path)
        yolo_mean = df_A[(df_A['fold'] == 'mean') & (df_A['crop_type'] == 'YOLO')]
        gt_mean = df_A[(df_A['fold'] == 'mean') & (df_A['crop_type'] == 'GT')]
        
        results_A = {
            'yolo_top1': float(yolo_mean['top1'].values[0]) if len(yolo_mean) > 0 else 0,
            'yolo_top5': float(yolo_mean['top5'].values[0]) if len(yolo_mean) > 0 else 0,
            'gt_top1': float(gt_mean['top1'].values[0]) if len(gt_mean) > 0 else 0,
            'gt_top5': float(gt_mean['top5'].values[0]) if len(gt_mean) > 0 else 0
        }
    else:
        results_A = {'yolo_top1': 0, 'yolo_top5': 0, 'gt_top1': 0, 'gt_top5': 0}
    
    results_B_summary = {
        'yolo_top1': results_B_YOLO['top1'], 'yolo_top5': results_B_YOLO['top5'],
        'gt_top1': results_B_GT['top1'], 'gt_top5': results_B_GT['top5']
    }
    
    results_C_summary = {
        'yolo_top1': results_C_YOLO['top1'], 'yolo_top5': results_C_YOLO['top5'],
        'gt_top1': results_C_GT['top1'], 'gt_top5': results_C_GT['top5']
    }
    
    create_comparison_plot(results_A, results_B_summary, results_C_summary,
                          figures_dir / 'protocolA_B_C_comparison.png', args.backbone)
    
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - {args.backbone.upper()}")
    print(f"{'='*70}")
    
    print(f"\nProtocol A (Within-ID Rotation, 5-fold mean):")
    print(f"  GT:   Top-1={results_A['gt_top1']:.2f}%, Top-5={results_A['gt_top5']:.2f}%")
    print(f"  YOLO: Top-1={results_A['yolo_top1']:.2f}%, Top-5={results_A['yolo_top5']:.2f}%")
    
    print(f"\nProtocol B (Cross-Angle, {results_B_GT['num_queries']} queries):")
    print(f"  GT:   Top-1={results_B_GT['top1']:.2f}%, Top-5={results_B_GT['top5']:.2f}%, mAP={results_B_GT['mAP']:.2f}%")
    print(f"  YOLO: Top-1={results_B_YOLO['top1']:.2f}%, Top-5={results_B_YOLO['top5']:.2f}%, mAP={results_B_YOLO['mAP']:.2f}% (fail={results_B_YOLO['yolo_fail_rate']:.1f}%)")
    
    print(f"\nProtocol C (Hard Cases, {results_C_GT['num_queries']} queries):")
    print(f"  GT:   Top-1={results_C_GT['top1']:.2f}%, Top-5={results_C_GT['top5']:.2f}%, mAP={results_C_GT['mAP']:.2f}%")
    print(f"  YOLO: Top-1={results_C_YOLO['top1']:.2f}%, Top-5={results_C_YOLO['top5']:.2f}%, mAP={results_C_YOLO['mAP']:.2f}% (fail={results_C_YOLO['yolo_fail_rate']:.1f}%)")
    
    print(f"\n✓ All deliverables saved to:")
    print(f"  Results: {results_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Prototypes: {prototypes_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
