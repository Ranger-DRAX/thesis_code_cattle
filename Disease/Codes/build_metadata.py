"""
Build metadata.csv for disease classification dataset
Creates one row per image with labels derived from folder path.
"""

import os
import pandas as pd
from pathlib import Path

# Base directory
BASE_DIR = Path(r"e:\Disease Classification")
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "Project"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

def build_metadata():
    """
    Build metadata.csv with columns:
    - filepath: relative path from Dataset folder
    - disease: {healthy, lsd, fmd, ibk}
    - severity: {1, 2, 3} for diseased; NA for healthy
    - label_10: combined label (e.g., healthy, lsd_s1, fmd_s2, etc.)
    """
    
    metadata_records = []
    
    # Disease mapping (folder name -> lowercase disease name)
    disease_mapping = {
        "HEALTHY": "healthy",
        "LSD": "lsd",
        "FMD": "fmd",
        "IBK": "ibk"
    }
    
    # Stage mapping (folder name -> severity number)
    stage_mapping = {
        "Stage1": 1,
        "Stage2": 2,
        "Stage3": 3
    }
    
    # Process each disease folder
    for disease_folder in DATASET_DIR.iterdir():
        if not disease_folder.is_dir():
            continue
            
        disease_name_upper = disease_folder.name
        disease_name = disease_mapping.get(disease_name_upper)
        
        if disease_name is None:
            print(f"Warning: Unknown disease folder: {disease_name_upper}")
            continue
        
        # Handle HEALTHY folder (no stages)
        if disease_name == "healthy":
            for image_file in disease_folder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Relative path from Dataset folder
                    rel_path = image_file.relative_to(DATASET_DIR)
                    
                    metadata_records.append({
                        'filepath': str(rel_path),
                        'disease': 'healthy',
                        'severity': 'NA',
                        'label_10': 'healthy'
                    })
        else:
            # Handle diseased folders with stages
            for stage_folder in disease_folder.iterdir():
                if not stage_folder.is_dir():
                    continue
                
                stage_name = stage_folder.name
                severity = stage_mapping.get(stage_name)
                
                if severity is None:
                    print(f"Warning: Unknown stage folder: {stage_name} in {disease_name_upper}")
                    continue
                
                # Process images in stage folder
                for image_file in stage_folder.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Relative path from Dataset folder
                        rel_path = image_file.relative_to(DATASET_DIR)
                        
                        # Create label_10: disease_s{severity}
                        label_10 = f"{disease_name}_s{severity}"
                        
                        metadata_records.append({
                            'filepath': str(rel_path),
                            'disease': disease_name,
                            'severity': severity,
                            'label_10': label_10
                        })
    
    # Create DataFrame
    df = pd.DataFrame(metadata_records)
    
    # Sort by filepath for consistency
    df = df.sort_values('filepath').reset_index(drop=True)
    
    # Save to CSV
    output_path = OUTPUT_DIR / "metadata.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Metadata CSV created successfully!")
    print(f"  Output: {output_path}")
    print(f"\nDataset Summary:")
    print(f"  Total images: {len(df)}")
    print(f"\nDisease distribution:")
    print(df['disease'].value_counts().sort_index())
    print(f"\nLabel_10 distribution:")
    print(df['label_10'].value_counts().sort_index())
    print(f"\nSeverity distribution (diseased only):")
    diseased_df = df[df['disease'] != 'healthy']
    print(diseased_df['severity'].value_counts().sort_index())
    
    return df

if __name__ == "__main__":
    df = build_metadata()
    print(f"\n✓ Step 1 completed: metadata.csv saved in Project directory")
