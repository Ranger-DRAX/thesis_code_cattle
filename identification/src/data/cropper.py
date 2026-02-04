import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Optional, Dict

class CowCropper:
    def __init__(self, target_size=(224, 224), padding=0.15):
        self.target_size = target_size
        self.padding = padding
    
    def expand_bbox(self, bbox, img_width, img_height):
        x_min, y_min, x_max, y_max = bbox
        w, h = x_max - x_min, y_max - y_min
        
        pad_w, pad_h = w * self.padding, h * self.padding
        
        x_min = max(0, int(x_min - pad_w))
        y_min = max(0, int(y_min - pad_h))
        x_max = min(img_width, int(x_max + pad_w))
        y_max = min(img_height, int(y_max + pad_h))
        
        return x_min, y_min, x_max, y_max
    
    def letterbox_resize(self, img):
        h, w = img.shape[:2]
        tw, th = self.target_size
        
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        
        y_offset = (th - nh) // 2
        x_offset = (tw - nw) // 2
        canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
        
        return canvas
    
    def yolo_to_xyxy(self, yolo_line, img_width, img_height):
        parts = yolo_line.strip().split()
        _, xc, yc, w, h = map(float, parts)
        
        xc_px = xc * img_width
        yc_px = yc * img_height
        w_px = w * img_width
        h_px = h * img_height
        
        x_min = xc_px - w_px / 2
        y_min = yc_px - h_px / 2
        x_max = xc_px + w_px / 2
        y_max = yc_px + h_px / 2
        
        return x_min, y_min, x_max, y_max
    
    def crop_from_gt_bbox(self, img_path, gt_bbox_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        if not gt_bbox_path.exists():
            return None
        
        with open(gt_bbox_path, 'r') as f:
            yolo_line = f.readline().strip()
            if not yolo_line:
                return None
        
        img_h, img_w = img.shape[:2]
        bbox = self.yolo_to_xyxy(yolo_line, img_w, img_h)
        expanded = self.expand_bbox(bbox, img_w, img_h)
        x_min, y_min, x_max, y_max = expanded
        
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None
        
        return self.letterbox_resize(crop)
    
    def crop_from_yolo_prediction(self, img_path, yolo_pred):
        if not yolo_pred.get('detected', False):
            return None
        
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img_h, img_w = img.shape[:2]
        bbox = (yolo_pred['x_min'], yolo_pred['y_min'], 
                yolo_pred['x_max'], yolo_pred['y_max'])
        
        expanded = self.expand_bbox(bbox, img_w, img_h)
        x_min, y_min, x_max, y_max = expanded
        
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None
        
        return self.letterbox_resize(crop)
    
    def crop_batch_oracle(self, image_paths, gt_bbox_dir):
        successful, failed = [], []
        
        for img_path in image_paths:
            bbox_path = gt_bbox_dir / f"{img_path.stem}.txt"
            cropped = self.crop_from_gt_bbox(img_path, bbox_path)
            
            if cropped is not None:
                successful.append({'path': img_path, 'crop': cropped})
            else:
                failed.append(img_path)
        
        return successful, failed
    
    def crop_batch_yolo(self, image_paths, yolo_predictions):
        successful, failed = [], []
        
        for img_path in image_paths:
            pred = yolo_predictions.get(img_path.name)
            if pred is None:
                failed.append(img_path)
                continue
            
            cropped = self.crop_from_yolo_prediction(img_path, pred)
            if cropped is not None:
                successful.append({'path': img_path, 'crop': cropped})
            else:
                failed.append(img_path)
        
        return successful, failed


def load_yolo_predictions(json_path):
    with open(json_path, 'r') as f:
        preds = json.load(f)
    return {pred['filename']: pred for pred in preds}


if __name__ == "__main__":
    base_dir = Path(r"d:\identification")
    cropper = CowCropper(target_size=(224, 224), padding=0.15)
    
    print("Cropper initialized")
    print(f"Target size: {cropper.target_size}")
    print(f"Padding: {cropper.padding}")
