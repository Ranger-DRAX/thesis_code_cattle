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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

def get_embedding(model, image_path: Path):
    """Get embedding for a single image."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy().flatten()

def load_gallery_prototypes(backbone: str):
    """Load pre-computed gallery prototypes."""
    proto_dir = Path(f'prototypes/{backbone}')
    
    gallery_B = np.load(proto_dir / 'protocolB_gallery.npy')
    gallery_C = np.load(proto_dir / 'protocolC_gallery.npy')
    
    with open(proto_dir / 'gallery_ids.json', 'r') as f:
        gallery_ids = json.load(f)
    
    return gallery_B, gallery_C, gallery_ids

def evaluate_with_details(model, query_df, gallery_embeddings, gallery_ids, protocol_name, backbone):
    """Evaluate and return detailed per-query predictions."""
    
    results = []
    
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
                'rank_of_true_id': -1
            })
            continue
        
        query_emb = get_embedding(model, img_path)
        
        similarities = np.dot(gallery_embeddings, query_emb)
        
        top5_indices = np.argsort(similarities)[::-1][:5]
        top5_ids = [gallery_ids[i] for i in top5_indices]
        top5_sims = [float(similarities[i]) for i in top5_indices]
        
        predicted_id = top5_ids[0]
        correct = (str(predicted_id) == str(true_id))
        
        if true_id in gallery_ids:
            true_idx = gallery_ids.index(true_id)
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
            'rank_of_true_id': rank
        })
    
    return pd.DataFrame(results)

def main():
    base_dir = Path(__file__).parent
    
    print("="*70)
    print("Generating Detailed Prediction Logs")
    print("="*70)
    
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
        print(f"  Loaded model from {ckpt_path}")
        
        try:
            gallery_B, gallery_C, gallery_ids = load_gallery_prototypes(backbone)
            print(f"  Loaded gallery: {len(gallery_ids)} IDs")
        except Exception as e:
            print(f"  Error loading gallery: {e}")
            continue
        
        output_dir = base_dir / 'results' / backbone / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Protocol B (Cross-Angle)...")
        testB_csv = base_dir / 'test_cross_angle' / 'testB_index.csv'
        if testB_csv.exists():
            df_B = pd.read_csv(testB_csv)
            predictions_B = evaluate_with_details(model, df_B, gallery_B, gallery_ids, 'B', backbone)
            output_path = output_dir / 'protocolB_predictions.csv'
            predictions_B.to_csv(output_path, index=False)
            
            correct = predictions_B['correct'].sum()
            total = len(predictions_B)
            print(f"    Saved: {output_path}")
            print(f"    Accuracy: {correct}/{total} ({100*correct/total:.2f}%)")
            
            incorrect = predictions_B[~predictions_B['correct']]
            if len(incorrect) > 0:
                print(f"    Sample incorrect predictions:")
                for _, row in incorrect.head(5).iterrows():
                    print(f"      {row['query_image']}: True={row['true_id']}, Pred={row['predicted_id']}, Sim={row['top1_similarity']:.3f}")
        
        print(f"\n  Protocol C (Hard Cases)...")
        testC_csv = base_dir / 'test_hard' / 'testC_index.csv'
        if testC_csv.exists():
            df_C = pd.read_csv(testC_csv)
            predictions_C = evaluate_with_details(model, df_C, gallery_C, gallery_ids, 'C', backbone)
            output_path = output_dir / 'protocolC_predictions.csv'
            predictions_C.to_csv(output_path, index=False)
            
            correct = predictions_C['correct'].sum()
            total = len(predictions_C)
            print(f"    Saved: {output_path}")
            print(f"    Accuracy: {correct}/{total} ({100*correct/total:.2f}%)")
            
            incorrect = predictions_C[~predictions_C['correct']]
            if len(incorrect) > 0:
                print(f"    Sample incorrect predictions:")
                for _, row in incorrect.head(5).iterrows():
                    print(f"      {row['query_image']}: True={row['true_id']}, Pred={row['predicted_id']}, Sim={row['top1_similarity']:.3f}")
        
        print(f"\n  Protocol D (Open-Set)...")
        testD_csv = base_dir / 'test_unknown' / 'testD_index.csv'
        if testD_csv.exists():
            df_D = pd.read_csv(testD_csv)
            
            predictions_D = []
            
            for idx, row in df_D.iterrows():
                img_path = Path(row['filepath'])
                true_id = str(row['cow_id'])
                is_unknown = (row.get('is_unknown', False) or true_id.lower() == 'unknown' or true_id == '-1')
                
                if not img_path.exists():
                    for base in ['data/test_cross_angle/images', 'data/test_hard/images', 'data/test_unknown/images']:
                        alt_path = Path(base) / img_path.name
                        if alt_path.exists():
                            img_path = alt_path
                            break
                
                if not img_path.exists():
                    predictions_D.append({
                        'query_image': str(img_path),
                        'true_id': true_id,
                        'is_unknown': is_unknown,
                        'predicted_id': 'FILE_NOT_FOUND',
                        'max_similarity': 0.0,
                        'decision': 'ERROR',
                        'correct_decision': False
                    })
                    continue
                
                query_emb = get_embedding(model, img_path)
                
                similarities = np.dot(gallery_B, query_emb)
                max_sim = float(np.max(similarities))
                pred_idx = np.argmax(similarities)
                predicted_id = gallery_ids[pred_idx]
                
                tau_path = base_dir / 'results' / backbone / 'tau_fold0.json'
                if tau_path.exists():
                    with open(tau_path, 'r') as f:
                        tau_data = json.load(f)
                    tau = tau_data['tau']
                else:
                    tau = 0.5  # Default
                
                if max_sim < tau:
                    decision = 'REJECT_UNKNOWN'
                    final_pred = 'UNKNOWN'
                else:
                    decision = 'ACCEPT_KNOWN'
                    final_pred = predicted_id
                
                if is_unknown:
                    correct_decision = (decision == 'REJECT_UNKNOWN')
                else:
                    correct_decision = (decision == 'ACCEPT_KNOWN' and str(predicted_id) == str(true_id))
                
                predictions_D.append({
                    'query_image': img_path.name,
                    'true_id': true_id,
                    'is_unknown': is_unknown,
                    'predicted_id': predicted_id,
                    'final_prediction': final_pred,
                    'max_similarity': max_sim,
                    'threshold': tau,
                    'decision': decision,
                    'correct_decision': correct_decision
                })
            
            df_predictions_D = pd.DataFrame(predictions_D)
            output_path = output_dir / 'protocolD_predictions.csv'
            df_predictions_D.to_csv(output_path, index=False)
            
            print(f"    Saved: {output_path}")
            
            known_queries = df_predictions_D[~df_predictions_D['is_unknown']]
            unknown_queries = df_predictions_D[df_predictions_D['is_unknown']]
            
            print(f"    Known queries: {len(known_queries)}, Unknown queries: {len(unknown_queries)}")
            print(f"    Known correct decisions: {known_queries['correct_decision'].sum()}/{len(known_queries)}")
            print(f"    Unknown correctly rejected: {unknown_queries['correct_decision'].sum()}/{len(unknown_queries)}")
    
    print(f"\n{'='*70}")
    print("All prediction details saved!")
    print(f"{'='*70}")
    
    print("\nOutput files:")
    for backbone in ['resnet50', 'convnext_tiny']:
        print(f"\n  {backbone}:")
        print(f"    results/{backbone}/predictions/protocolB_predictions.csv")
        print(f"    results/{backbone}/predictions/protocolC_predictions.csv")
        print(f"    results/{backbone}/predictions/protocolD_predictions.csv")

if __name__ == "__main__":
    main()
