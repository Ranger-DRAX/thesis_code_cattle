import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.reid_model import ResNet50ReID, ConvNeXtTinyReID
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class YOLOCropper:
    """YOLO-based cow cropper."""
    
    def __init__(self, model_path: str, conf_thresh: float = 0.5, padding: float = 0.15):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.padding = padding
    
    def crop(self, image_path: Path) -> Image.Image:
        """Crop cow from image using YOLO detection."""
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        results = self.model(img_np, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None  # Detection failed
        
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        
        if confs.max() < self.conf_thresh:
            return None  # Confidence too low
        
        best_idx = confs.argmax()
        box = boxes.xyxy[best_idx].cpu().numpy()
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        pad_w, pad_h = w * self.padding, h * self.padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(img.width, x2 + pad_w)
        y2 = min(img.height, y2 + pad_h)
        
        cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
        return cropped

def load_model(backbone: str, checkpoint_path: Path):
    """Load trained ReID model."""
    if backbone == 'resnet50':
        model = ResNet50ReID(embed_dim=512, pretrained=False)
    else:
        model = ConvNeXtTinyReID(embed_dim=512, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def get_embedding(model, image: Image.Image):
    """Get embedding for a PIL image."""
    img_tensor = eval_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy().flatten()

def get_embedding_from_path(model, image_path: Path):
    """Get embedding for an image file."""
    img = Image.open(image_path).convert('RGB')
    return get_embedding(model, img)

def load_gallery_prototypes(backbone: str):
    """Load pre-computed gallery prototypes."""
    proto_dir = Path(f'prototypes/{backbone}')
    
    gallery_B = np.load(proto_dir / 'protocolB_gallery.npy')
    gallery_C = np.load(proto_dir / 'protocolC_gallery.npy')
    
    with open(proto_dir / 'gallery_ids.json', 'r') as f:
        gallery_ids = json.load(f)
    
    return gallery_B, gallery_C, gallery_ids

def evaluate_with_yolo(model, cropper, query_df, gallery_embeddings, gallery_ids, protocol_name, backbone):
    """Evaluate using YOLO crops and return detailed per-query predictions."""
    
    results = []
    yolo_fail_count = 0
    
    gallery_ids_str = [str(gid) for gid in gallery_ids]
    
    for idx, row in query_df.iterrows():
        img_path = Path(row['filepath'])
        true_id = str(row['cow_id'])
        
        if not img_path.exists():
            for base in ['data/test_cross_angle/images', 'data/test_hard/images', 'data/test_unknown/images']:
                alt_path = Path(base) / img_path.name
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        if not img_path.exists():
            results.append({
                'query_image': str(img_path),
                'true_id': true_id,
                'predicted_id': 'FILE_NOT_FOUND',
                'correct': False,
                'top1_similarity': 0.0,
                'top5_ids': '',
                'top5_similarities': '',
                'rank_of_true_id': -1,
                'yolo_status': 'FILE_NOT_FOUND'
            })
            continue
        
        cropped_img = cropper.crop(img_path)
        
        if cropped_img is None:
            yolo_fail_count += 1
            results.append({
                'query_image': img_path.name,
                'true_id': true_id,
                'predicted_id': 'YOLO_FAIL',
                'correct': False,
                'top1_similarity': 0.0,
                'top5_ids': '',
                'top5_similarities': '',
                'rank_of_true_id': -1,
                'yolo_status': 'DETECTION_FAILED'
            })
            continue
        
        query_emb = get_embedding(model, cropped_img)
        
        similarities = np.dot(gallery_embeddings, query_emb)
        
        top5_indices = np.argsort(similarities)[::-1][:5]
        top5_ids = [gallery_ids_str[i] for i in top5_indices]
        top5_sims = [float(similarities[i]) for i in top5_indices]
        
        predicted_id = top5_ids[0]
        correct = (predicted_id == true_id)
        
        if true_id in gallery_ids_str:
            true_idx = gallery_ids_str.index(true_id)
            sorted_indices = np.argsort(similarities)[::-1]
            rank = np.where(sorted_indices == true_idx)[0][0] + 1
        else:
            rank = -1  # True ID not in gallery (unknown)
        
        results.append({
            'query_image': img_path.name,
            'true_id': true_id,
            'predicted_id': predicted_id,
            'correct': correct,
            'top1_similarity': top5_sims[0],
            'top5_ids': '|'.join(map(str, top5_ids)),
            'top5_similarities': '|'.join([f'{s:.4f}' for s in top5_sims]),
            'rank_of_true_id': rank,
            'yolo_status': 'SUCCESS'
        })
    
    df_results = pd.DataFrame(results)
    return df_results, yolo_fail_count

def evaluate_protocolD_with_yolo(model, cropper, known_df, unknown_df, gallery_embeddings, gallery_ids, tau, backbone):
    """Evaluate Protocol D (open-set) using YOLO crops."""
    
    results = []
    yolo_fail_count = 0
    
    gallery_ids_str = [str(gid) for gid in gallery_ids]
    
    for idx, row in known_df.iterrows():
        img_path = Path(row['filepath'])
        true_id = str(row['cow_id'])
        
        if not img_path.exists():
            for base in ['data/test_cross_angle/images', 'data/test_hard/images']:
                alt_path = Path(base) / img_path.name
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        if not img_path.exists():
            results.append({
                'query_image': str(img_path),
                'true_id': true_id,
                'is_unknown': False,
                'predicted_id': 'FILE_NOT_FOUND',
                'final_prediction': 'FILE_NOT_FOUND',
                'max_similarity': 0.0,
                'correct_id': False,
                'correct_decision': False,
                'yolo_status': 'FILE_NOT_FOUND'
            })
            continue
        
        cropped_img = cropper.crop(img_path)
        
        if cropped_img is None:
            yolo_fail_count += 1
            results.append({
                'query_image': img_path.name,
                'true_id': true_id,
                'is_unknown': False,
                'predicted_id': 'YOLO_FAIL',
                'final_prediction': 'YOLO_FAIL',
                'max_similarity': 0.0,
                'correct_id': False,
                'correct_decision': False,
                'yolo_status': 'DETECTION_FAILED'
            })
            continue
        
        query_emb = get_embedding(model, cropped_img)
        similarities = np.dot(gallery_embeddings, query_emb)
        
        max_sim = similarities.max()
        pred_idx = similarities.argmax()
        predicted_id = gallery_ids_str[pred_idx]
        
        if max_sim < tau:
            final_prediction = 'UNKNOWN'
            correct_decision = False  # Known should not be rejected
        else:
            final_prediction = predicted_id
            correct_decision = (predicted_id == true_id)
        
        correct_id = (predicted_id == true_id)
        
        results.append({
            'query_image': img_path.name,
            'true_id': true_id,
            'is_unknown': False,
            'predicted_id': predicted_id,
            'final_prediction': final_prediction,
            'max_similarity': float(max_sim),
            'correct_id': correct_id,
            'correct_decision': correct_decision,
            'yolo_status': 'SUCCESS'
        })
    
    for idx, row in unknown_df.iterrows():
        img_path = Path(row['filepath'])
        
        if not img_path.exists():
            for base in ['data/test_unknown/images']:
                alt_path = Path(base) / img_path.name
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        if not img_path.exists():
            results.append({
                'query_image': str(img_path),
                'true_id': 'unknown',
                'is_unknown': True,
                'predicted_id': 'FILE_NOT_FOUND',
                'final_prediction': 'FILE_NOT_FOUND',
                'max_similarity': 0.0,
                'correct_id': False,
                'correct_decision': False,
                'yolo_status': 'FILE_NOT_FOUND'
            })
            continue
        
        cropped_img = cropper.crop(img_path)
        
        if cropped_img is None:
            yolo_fail_count += 1
            results.append({
                'query_image': img_path.name,
                'true_id': 'unknown',
                'is_unknown': True,
                'predicted_id': 'YOLO_FAIL',
                'final_prediction': 'YOLO_FAIL',
                'max_similarity': 0.0,
                'correct_id': False,
                'correct_decision': False,
                'yolo_status': 'DETECTION_FAILED'
            })
            continue
        
        query_emb = get_embedding(model, cropped_img)
        similarities = np.dot(gallery_embeddings, query_emb)
        
        max_sim = similarities.max()
        pred_idx = similarities.argmax()
        predicted_id = gallery_ids_str[pred_idx]
        
        if max_sim < tau:
            final_prediction = 'UNKNOWN'
            correct_decision = True  # Unknown correctly rejected
        else:
            final_prediction = predicted_id
            correct_decision = False  # Unknown incorrectly accepted
        
        results.append({
            'query_image': img_path.name,
            'true_id': 'unknown',
            'is_unknown': True,
            'predicted_id': predicted_id,
            'final_prediction': final_prediction,
            'max_similarity': float(max_sim),
            'correct_id': False,
            'correct_decision': correct_decision,
            'yolo_status': 'SUCCESS'
        })
    
    df_results = pd.DataFrame(results)
    return df_results, yolo_fail_count

def main():
    base_dir = Path(__file__).parent
    
    print("="*70)
    print("Generating YOLO-based Prediction Logs")
    print("="*70)
    
    yolo_path = base_dir / 'checkpoints' / 'yolo_fold0_best.pt'
    if not yolo_path.exists():
        print(f"YOLO model not found: {yolo_path}")
        return
    
    cropper = YOLOCropper(str(yolo_path), conf_thresh=0.5, padding=0.15)
    print(f"Loaded YOLO model from {yolo_path}")
    
    for backbone in ['resnet50', 'convnext_tiny']:
        print(f"\n{'='*50}")
        print(f"Processing: {backbone.upper()}")
        print(f"{'='*50}")
        
        ckpt_path = base_dir / 'checkpoints' / f'reid_fold0_{backbone}_supcon+arcface_best.pt'
        if not ckpt_path.exists():
            ckpt_path = base_dir / 'checkpoints' / f'reid_fold0_{backbone}_combined_best.pt'
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue
        
        model = load_model(backbone, ckpt_path)
        print(f"  Loaded ReID model from {ckpt_path}")
        
        try:
            gallery_B, gallery_C, gallery_ids = load_gallery_prototypes(backbone)
            print(f"  Loaded gallery: {len(gallery_ids)} IDs")
        except Exception as e:
            print(f"  Error loading gallery: {e}")
            continue
        
        tau_path = base_dir / 'results' / backbone / 'tau_fold0.json'
        if tau_path.exists():
            with open(tau_path) as f:
                tau_data = json.load(f)
            tau = tau_data['tau']
            print(f"  Loaded threshold τ = {tau:.4f}")
        else:
            tau = 0.5
            print(f"  Using default threshold τ = {tau}")
        
        output_dir = base_dir / 'results' / backbone / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Protocol B (Cross-Angle) - YOLO crop...")
        testB_csv = base_dir / 'data' / 'test_cross_angle' / 'testB_index.csv'
        if not testB_csv.exists():
            testB_csv = base_dir / 'test_cross_angle' / 'testB_index.csv'
        
        if testB_csv.exists():
            df_B = pd.read_csv(testB_csv)
            predictions_B, yolo_fails_B = evaluate_with_yolo(
                model, cropper, df_B, gallery_B, gallery_ids, 'B', backbone
            )
            output_path = output_dir / 'protocolB_predictions_YOLO.csv'
            predictions_B.to_csv(output_path, index=False)
            
            valid = predictions_B[predictions_B['yolo_status'] == 'SUCCESS']
            correct = valid['correct'].sum()
            total = len(valid)
            print(f"    Saved: {output_path}")
            print(f"    YOLO Success: {total}/{len(predictions_B)} ({100*total/len(predictions_B):.1f}%)")
            print(f"    Accuracy (on successful): {correct}/{total} ({100*correct/total:.2f}%)")
        else:
            print(f"    testB_index.csv not found")
        
        print(f"\n  Protocol C (Hard Cases) - YOLO crop...")
        testC_csv = base_dir / 'data' / 'test_hard' / 'testC_index.csv'
        if not testC_csv.exists():
            testC_csv = base_dir / 'test_hard' / 'testC_index.csv'
        
        if testC_csv.exists():
            df_C = pd.read_csv(testC_csv)
            predictions_C, yolo_fails_C = evaluate_with_yolo(
                model, cropper, df_C, gallery_C, gallery_ids, 'C', backbone
            )
            output_path = output_dir / 'protocolC_predictions_YOLO.csv'
            predictions_C.to_csv(output_path, index=False)
            
            valid = predictions_C[predictions_C['yolo_status'] == 'SUCCESS']
            correct = valid['correct'].sum()
            total = len(valid)
            print(f"    Saved: {output_path}")
            print(f"    YOLO Success: {total}/{len(predictions_C)} ({100*total/len(predictions_C):.1f}%)")
            print(f"    Accuracy (on successful): {correct}/{total} ({100*correct/total:.2f}%)")
        else:
            print(f"    testC_index.csv not found")
        
        print(f"\n  Protocol D (Open-Set) - YOLO crop...")
        
        known_dfs = []
        if testB_csv.exists():
            known_dfs.append(pd.read_csv(testB_csv))
        if testC_csv.exists():
            known_dfs.append(pd.read_csv(testC_csv))
        
        if len(known_dfs) > 0:
            known_df = pd.concat(known_dfs, ignore_index=True)
        else:
            known_df = pd.DataFrame()
        
        testD_csv = base_dir / 'data' / 'test_unknown' / 'testD_index.csv'
        if not testD_csv.exists():
            testD_csv = base_dir / 'test_unknown' / 'testD_index.csv'
        
        if testD_csv.exists():
            unknown_df = pd.read_csv(testD_csv)
        else:
            unknown_df = pd.DataFrame()
        
        if len(known_df) > 0 or len(unknown_df) > 0:
            predictions_D, yolo_fails_D = evaluate_protocolD_with_yolo(
                model, cropper, known_df, unknown_df, gallery_B, gallery_ids, tau, backbone
            )
            output_path = output_dir / 'protocolD_predictions_YOLO.csv'
            predictions_D.to_csv(output_path, index=False)
            
            valid = predictions_D[predictions_D['yolo_status'] == 'SUCCESS']
            known_valid = valid[~valid['is_unknown']]
            unknown_valid = valid[valid['is_unknown']]
            
            print(f"    Saved: {output_path}")
            print(f"    Total queries: {len(predictions_D)} (Known: {len(known_df)}, Unknown: {len(unknown_df)})")
            print(f"    YOLO failures: {yolo_fails_D}")
            
            if len(known_valid) > 0:
                known_correct = known_valid['correct_decision'].sum()
                print(f"    Known correct decision: {known_correct}/{len(known_valid)} ({100*known_correct/len(known_valid):.1f}%)")
            
            if len(unknown_valid) > 0:
                unknown_rejected = unknown_valid['correct_decision'].sum()
                print(f"    Unknown rejected: {unknown_rejected}/{len(unknown_valid)} ({100*unknown_rejected/len(unknown_valid):.1f}%)")
        else:
            print(f"    No test data found for Protocol D")
    
    print(f"\n{'='*70}")
    print("YOLO-based Prediction Generation Complete!")
    print(f"{'='*70}")
    
    print("\nGenerated files:")
    for backbone in ['resnet50', 'convnext_tiny']:
        print(f"\n  {backbone}:")
        pred_dir = base_dir / 'results' / backbone / 'predictions'
        if pred_dir.exists():
            for f in sorted(pred_dir.glob('*_YOLO.csv')):
                print(f"    ✓ {f.name}")

if __name__ == "__main__":
    main()
