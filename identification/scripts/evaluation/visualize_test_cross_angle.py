"""
Generate a 4x4 grid visualization for test_cross_angle dataset.
Each row contains: 2 original images + 2 GT cropped images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height):
    """Convert YOLO format to xyxy pixel coordinates."""
    x1 = int((x_center - w/2) * img_width)
    y1 = int((y_center - h/2) * img_height)
    x2 = int((x_center + w/2) * img_width)
    y2 = int((y_center + h/2) * img_height)
    return [x1, y1, x2, y2]

def crop_with_bbox(img, bbox):
    """Crop image using bbox coordinates."""
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]

def resize_maintain_aspect(img, target_size=(400, 400)):
    """Resize image while maintaining aspect ratio."""
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create canvas and paste
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def main():
    # Setup paths
    test_dir = Path(r"d:\identification\test_cross_angle")
    images_dir = test_dir / "images"
    bboxes_dir = test_dir / "bboxes_txt"
    output_dir = test_dir / "visualizations"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.jpg"))
    
    # Randomly select 4 images for the 4 rows
    selected_images = random.sample(image_files, 4)
    
    # Create figure with 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Test Cross Angle Dataset - Sample Visualization\n' + 
                 'Each Row: Original Image 1 | Original Image 2 | GT Crop 1 | GT Crop 2',
                 fontsize=16, fontweight='bold')
    
    for row_idx, img_file in enumerate(selected_images):
        # Read original image
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Get cow ID from filename
        cow_id = img_file.stem
        
        # Read bbox
        bbox_file = bboxes_dir / f"{cow_id}.txt"
        bbox = None
        if bbox_file.exists():
            with open(bbox_file, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) >= 5:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    bbox = yolo_to_xyxy(x_center, y_center, width, height, w, h)
        
        # Prepare images for this row
        img_resized = resize_maintain_aspect(img, (400, 400))
        
        if bbox is not None:
            # Draw bbox on a copy for visualization
            img_with_bbox = img.copy()
            cv2.rectangle(img_with_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 3)
            img_with_bbox_resized = resize_maintain_aspect(img_with_bbox, (400, 400))
            
            # Create crop
            crop = crop_with_bbox(img, bbox)
            crop_resized = resize_maintain_aspect(crop, (400, 400))
        else:
            img_with_bbox_resized = img_resized
            crop_resized = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Column 0: Original image
        axes[row_idx, 0].imshow(img_resized)
        axes[row_idx, 0].set_title(f"Cow ID: {cow_id}\nOriginal Image", 
                                    fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Column 1: Image with bbox drawn
        axes[row_idx, 1].imshow(img_with_bbox_resized)
        axes[row_idx, 1].set_title(f"Cow ID: {cow_id}\nWith GT BBox", 
                                    fontsize=10, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # Column 2: GT Crop
        axes[row_idx, 2].imshow(crop_resized)
        axes[row_idx, 2].set_title(f"Cow ID: {cow_id}\nGT Crop", 
                                    fontsize=10, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        # Column 3: GT Crop (duplicate for 4x4 requirement)
        axes[row_idx, 3].imshow(crop_resized)
        crop_h, crop_w = crop.shape[:2] if bbox else (0, 0)
        axes[row_idx, 3].set_title(f"Cow ID: {cow_id}\nGT Crop\nSize: {crop_w}x{crop_h}px", 
                                    fontsize=10, fontweight='bold')
        axes[row_idx, 3].axis('off')
    
    plt.tight_layout()
    output_file = output_dir / "test_cross_angle_4x4_grid.jpg"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated 4x4 grid visualization")
    print(f"✓ Output saved to: {output_file}")
    print(f"\nSamples included:")
    for idx, img_file in enumerate(selected_images, 1):
        print(f"  Row {idx}: Cow ID {img_file.stem}")

if __name__ == "__main__":
    main()
