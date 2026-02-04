from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import json
import cv2
from tqdm import tqdm

base_dir = Path(r"d:\identification")
checkpoints_dir = base_dir / "checkpoints"
results_dir = base_dir / "results"
outputs_dir = base_dir / "outputs" / "yolo_preds"
results_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

def evaluate_on_test_set(model, fold=0):
    """Evaluate YOLO on test set and save metrics."""
    print("\n" + "="*70)
    print(f"STEP 2.3: Evaluating YOLO on Fold {fold} Test Set")
    print("="*70)
    
    test_csv = base_dir / "splits" / f"fold{fold}_test.csv"
    df_test = pd.read_csv(test_csv, dtype={'cow_id': str})
    
    print(f"\nTest set: {len(df_test)} images ({df_test['cow_id'].nunique()} cows)")
    
    results_list = []
    detection_count = 0
    no_detection_count = 0
    
    print("\nRunning detections on test images...")
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        img_path = base_dir / row['filepath']
        
        results = model(img_path, verbose=False)
        
        if len(results[0].boxes) > 0:
            detection_count += 1
            boxes = results[0].boxes
            conf = boxes.conf.cpu().numpy()
            best_idx = conf.argmax()
            
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            results_list.append({
                'filename': row['filename'],
                'cow_id': row['cow_id'],
                'detected': True,
                'confidence': float(conf[best_idx]),
                'x_min': float(xyxy[0]),
                'y_min': float(xyxy[1]),
                'x_max': float(xyxy[2]),
                'y_max': float(xyxy[3])
            })
        else:
            no_detection_count += 1
            results_list.append({
                'filename': row['filename'],
                'cow_id': row['cow_id'],
                'detected': False,
                'confidence': 0.0,
                'x_min': 0.0,
                'y_min': 0.0,
                'x_max': 0.0,
                'y_max': 0.0
            })
    
    detection_rate = detection_count / len(df_test)
    failure_rate = no_detection_count / len(df_test)
    
    print(f"\nDetection Results:")
    print(f"  Total images: {len(df_test)}")
    print(f"  Successful detections: {detection_count}")
    print(f"  Failed detections: {no_detection_count}")
    print(f"  Detection rate: {detection_rate:.2%}")
    print(f"  Failure rate: {failure_rate:.2%}")
    
    df_results = pd.DataFrame(results_list)
    results_csv = results_dir / f"detector_metrics_fold{fold}.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\n✓ Saved results to: {results_csv}")
    
    metrics_summary = {
        'fold': fold,
        'total_images': len(df_test),
        'detections': detection_count,
        'failures': no_detection_count,
        'detection_rate': detection_rate,
        'failure_rate': failure_rate,
        'avg_confidence': df_results[df_results['detected']]['confidence'].mean() if detection_count > 0 else 0.0
    }
    
    return metrics_summary, df_results

def export_predictions_json(model, dataset_name, images_dir, output_name):
    """Export YOLO predictions as JSON for a dataset."""
    print(f"\nExporting predictions for {dataset_name}...")
    
    image_files = []
    for ext in ['*.jpg', '*.JPG', '*.png', '*.jpeg', '*.webp', '*.avif']:
        image_files.extend(list(images_dir.glob(ext)))
    
    predictions = []
    detection_count = 0
    
    for img_path in tqdm(image_files):
        results = model(img_path, verbose=False)
        
        entry = {'filename': img_path.name}
        
        if len(results[0].boxes) > 0:
            detection_count += 1
            boxes = results[0].boxes
            conf = boxes.conf.cpu().numpy()
            best_idx = conf.argmax()
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            
            entry.update({
                'detected': True,
                'confidence': float(conf[best_idx]),
                'x_min': float(xyxy[0]),
                'y_min': float(xyxy[1]),
                'x_max': float(xyxy[2]),
                'y_max': float(xyxy[3])
            })
        else:
            entry.update({
                'detected': False,
                'confidence': 0.0,
                'x_min': 0.0,
                'y_min': 0.0,
                'x_max': 0.0,
                'y_max': 0.0
            })
        
        predictions.append(entry)
    
    output_path = outputs_dir / f"{output_name}.json"
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"  ✓ {len(predictions)} images, {detection_count} detections")
    print(f"  ✓ Saved to: {output_path}")
    
    return len(predictions), detection_count

def main():
    print("="*70)
    print("STEP 2.3 & 2.4: YOLO Evaluation and Prediction Export")
    print("="*70)
    
    model_path = checkpoints_dir / 'yolo_fold0_best.pt'
    if not model_path.exists():
        print(f"\n✗ Model not found at: {model_path}")
        print("Please train the model first using step2_2_train_yolo.py")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = YOLO(str(model_path))
    
    metrics, df_results = evaluate_on_test_set(model, fold=0)
    
    print("\n" + "="*70)
    print("STEP 2.4: Exporting Predictions for ReID Pipeline")
    print("="*70)
    
    print("\n[1] Main Dataset Predictions")
    main_images_dir = base_dir / "main" / "images"
    
    all_main_images = []
    for cow_id in range(1, 216):
        cow_dir = main_images_dir / str(cow_id)
        if cow_dir.exists():
            for ext in ['*.jpg', '*.JPG', '*.png', '*.jpeg']:
                all_main_images.extend(list(cow_dir.glob(ext)))
    
    print(f"Processing {len(all_main_images)} main dataset images...")
    main_predictions = []
    for img_path in tqdm(all_main_images):
        results = model(img_path, verbose=False)
        entry = {'filename': img_path.name, 'cow_id': img_path.parent.name}
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            conf = boxes.conf.cpu().numpy()
            best_idx = conf.argmax()
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            entry.update({
                'detected': True,
                'confidence': float(conf[best_idx]),
                'x_min': float(xyxy[0]),
                'y_min': float(xyxy[1]),
                'x_max': float(xyxy[2]),
                'y_max': float(xyxy[3])
            })
        else:
            entry.update({'detected': False, 'confidence': 0.0, 
                         'x_min': 0.0, 'y_min': 0.0, 'x_max': 0.0, 'y_max': 0.0})
        main_predictions.append(entry)
    
    main_json = outputs_dir / "fold0_main.json"
    with open(main_json, 'w') as f:
        json.dump(main_predictions, f, indent=2)
    print(f"  ✓ Saved to: {main_json}")
    
    print("\n[2] Protocol B (test_cross_angle)")
    export_predictions_json(model, "Protocol B", 
                           base_dir / "test_cross_angle" / "images",
                           "test_cross_angle")
    
    print("\n[3] Protocol C (test_hard)")
    export_predictions_json(model, "Protocol C",
                           base_dir / "test_hard" / "images", 
                           "test_hard")
    
    print("\n[4] Protocol D (test_unknown)")
    export_predictions_json(model, "Protocol D",
                           base_dir / "test_unknown" / "images",
                           "test_unknown")
    
    print("\n" + "="*70)
    print("STEP 2 COMPLETE!")
    print("="*70)
    print("\nDeliverables created:")
    print(f"  ✓ checkpoints/yolo_fold0_best.pt")
    print(f"  ✓ results/detector_metrics_fold0.csv")
    print(f"  ✓ outputs/yolo_preds/fold0_main.json")
    print(f"  ✓ outputs/yolo_preds/test_cross_angle.json")
    print(f"  ✓ outputs/yolo_preds/test_hard.json")
    print(f"  ✓ outputs/yolo_preds/test_unknown.json")
    print("\nNext: Step 3 - Build ReID preprocessing pipeline")

if __name__ == "__main__":
    main()
