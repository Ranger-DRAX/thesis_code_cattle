
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def calculate_iou(box1, box2):
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height):
    """Convert YOLO format to xyxy pixel coordinates."""
    x1 = int((x_center - w/2) * img_width)
    y1 = int((y_center - h/2) * img_height)
    x2 = int((x_center + w/2) * img_width)
    y2 = int((y_center + h/2) * img_height)
    return [x1, y1, x2, y2]

def get_bbox_metrics(bbox, img_width, img_height):
    """Calculate area and aspect ratio for a bbox."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = (width * height) / (img_width * img_height) * 100  # Percentage
    aspect_ratio = width / height if height > 0 else 0
    return area, aspect_ratio

def crop_image_with_padding(img, bbox, padding=0.0):
    """Crop image with padding around bbox."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    cropped = img[y1:y2, x1:x2]
    return cropped

def calculate_cosine_similarity(crop1, crop2):
    """Calculate cosine similarity between two image crops."""
    size = (224, 224)
    crop1_resized = cv2.resize(crop1, size)
    crop2_resized = cv2.resize(crop2, size)
    
    vec1 = crop1_resized.flatten().astype(float)
    vec2 = crop2_resized.flatten().astype(float)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0
    return similarity

def create_comparison_figure(image_path, gt_bbox, yolo_bbox, gt_crop, yolo_crop, 
                            metrics, sample_idx, output_path):
    """Create a comparison figure similar to the provided template."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    fig = plt.figure(figsize=(14, 10))
    
    title = f"Protocol B - Sample {sample_idx}\n"
    title += f"Cosine Similarity: {metrics['cosine_similarity']:.4f} | IoU: {metrics['iou']:.3f}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(img)
    rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), 
                             gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                             linewidth=3, edgecolor='lime', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title(f"GT BBox\nArea: {metrics['gt_area']:.1f}%, AR: {metrics['gt_ar']:.2f}", 
                  fontsize=12)
    ax1.axis('off')
    ax1.text(10, 30, 'GT', fontsize=12, color='lime', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(img)
    if yolo_bbox is not None:
        rect = patches.Rectangle((yolo_bbox[0], yolo_bbox[1]), 
                                 yolo_bbox[2]-yolo_bbox[0], yolo_bbox[3]-yolo_bbox[1],
                                 linewidth=3, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(10, 30, 'YOLO', fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.7))
    ax2.set_title(f"YOLO BBox\nArea: {metrics['yolo_area']:.1f}%, AR: {metrics['yolo_ar']:.2f}", 
                  fontsize=12)
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 2, 3)
    gt_crop_rgb = cv2.cvtColor(gt_crop, cv2.COLOR_BGR2RGB)
    ax3.imshow(gt_crop_rgb)
    crop_h, crop_w = gt_crop.shape[:2]
    ax3.set_title(f"GT Crop (0% padding)\nSize: {crop_w}x{crop_h}", fontsize=12)
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 2, 4)
    if yolo_crop is not None:
        yolo_crop_rgb = cv2.cvtColor(yolo_crop, cv2.COLOR_BGR2RGB)
        ax4.imshow(yolo_crop_rgb)
        crop_h, crop_w = yolo_crop.shape[:2]
        ax4.set_title(f"YOLO Crop (0% padding)\nSize: {crop_w}x{crop_h}", fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No Detection', ha='center', va='center', fontsize=14)
        ax4.set_title("YOLO Crop (0% padding)\nSize: N/A", fontsize=12)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {output_path.name}")

def main():
    main_dir = Path(r"d:\identification\main")
    images_dir = main_dir / "images"
    bboxes_dir = main_dir / "bboxes_txt"
    bbox_images_dir = main_dir / "bbox_images"
    comparison_dir = main_dir / "bbox_comparisons"
    
    comparison_dir.mkdir(exist_ok=True)
    
    yolo_model_path = Path(r"d:\identification\yolov8s.pt")
    print(f"Loading YOLO model: {yolo_model_path}")
    yolo = YOLO(str(yolo_model_path))
    print("YOLO model loaded successfully")
    
    sample_images = list(bbox_images_dir.glob("*_bbox.jpg"))
    
    samples_to_process = []
    for img_file in sample_images:
        parts = img_file.stem.split('_')
        if len(parts) >= 2:
            id_str = parts[0]
            view = parts[1]
            samples_to_process.append((id_str, view))
    
    samples_to_process = list(set(samples_to_process))[:10]
    
    print(f"\nProcessing {len(samples_to_process)} samples...")
    
    metrics_list = []
    
    for idx, (id_str, view) in enumerate(samples_to_process, 1):
        id_num = int(id_str)
        img_folder = images_dir / str(id_num)
        
        image_file = None
        for ext in ['.jpg', '.JPG', '.png', '.PNG']:
            potential_file = img_folder / f"{id_num}_{view}{ext}"
            if potential_file.exists():
                image_file = potential_file
                break
        
        if image_file is None:
            continue
        
        img = cv2.imread(str(image_file))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        bbox_file = bboxes_dir / f"{id_str}_{view}.txt"
        if not bbox_file.exists():
            continue
        
        with open(bbox_file, 'r') as f:
            line = f.readline().strip()
            parts = line.split()
            if len(parts) < 5:
                continue
            
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            gt_bbox = yolo_to_xyxy(x_center, y_center, width, height, w, h)
        
        results = yolo(img, verbose=False)
        
        yolo_bbox = None
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)
            yolo_bbox = boxes[best_idx].astype(int).tolist()
        
        gt_area, gt_ar = get_bbox_metrics(gt_bbox, w, h)
        
        if yolo_bbox is not None:
            yolo_area, yolo_ar = get_bbox_metrics(yolo_bbox, w, h)
            iou = calculate_iou(gt_bbox, yolo_bbox)
            
            gt_crop = crop_image_with_padding(img, gt_bbox, padding=0.0)
            yolo_crop = crop_image_with_padding(img, yolo_bbox, padding=0.0)
            
            cosine_sim = calculate_cosine_similarity(gt_crop, yolo_crop)
        else:
            yolo_area, yolo_ar = 0, 0
            iou = 0
            gt_crop = crop_image_with_padding(img, gt_bbox, padding=0.0)
            yolo_crop = None
            cosine_sim = 0
        
        metrics = {
            'id': id_str,
            'view': view,
            'gt_area': gt_area,
            'gt_ar': gt_ar,
            'yolo_area': yolo_area,
            'yolo_ar': yolo_ar,
            'iou': iou,
            'cosine_similarity': cosine_sim
        }
        
        metrics_list.append(metrics)
        
        output_file = comparison_dir / f"{id_str}_{view}_comparison.jpg"
        create_comparison_figure(image_file, gt_bbox, yolo_bbox, gt_crop, yolo_crop,
                                metrics, idx, output_file)
    
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    if metrics_list:
        avg_iou = np.mean([m['iou'] for m in metrics_list])
        avg_cosine = np.mean([m['cosine_similarity'] for m in metrics_list])
        avg_gt_area = np.mean([m['gt_area'] for m in metrics_list])
        avg_yolo_area = np.mean([m['yolo_area'] for m in metrics_list])
        
        print(f"Samples processed: {len(metrics_list)}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Cosine Similarity: {avg_cosine:.4f}")
        print(f"Average GT Area: {avg_gt_area:.2f}%")
        print(f"Average YOLO Area: {avg_yolo_area:.2f}%")
        print(f"\nDetailed Results:")
        print(f"{'ID':<6} {'View':<8} {'IoU':<8} {'Cosine':<10} {'GT Area':<10} {'YOLO Area':<10}")
        print(f"{'-'*60}")
        for m in metrics_list:
            print(f"{m['id']:<6} {m['view']:<8} {m['iou']:<8.3f} "
                  f"{m['cosine_similarity']:<10.4f} {m['gt_area']:<10.2f} {m['yolo_area']:<10.2f}")
    
    print(f"\n{'='*60}")
    print(f"Comparison images saved to: {comparison_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
