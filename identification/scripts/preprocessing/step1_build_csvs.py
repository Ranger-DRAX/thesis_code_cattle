import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

base_dir = Path(r"d:\identification")
main_images_dir = base_dir / "main" / "images"
main_bbox_dir = base_dir / "main" / "bboxes_txt"
splits_dir = base_dir / "splits"
splits_dir.mkdir(exist_ok=True)

print("Building CSV Files...")

views = ["front", "back", "left", "right"]
records = []

for cow_id in range(1, 216):
    img_folder = main_images_dir / str(cow_id)
    
    if not img_folder.exists():
        continue
    
    for view in views:
        img_files = list(img_folder.glob(f"{cow_id}_{view}.*"))
        
        if not img_files:
            continue
        
        img_file = img_files[0]
        bbox_file = main_bbox_dir / f"{cow_id:03d}_{view}.txt"
        
        filepath_rel = f"main/images/{cow_id}/{img_file.name}"
        bbox_rel = f"main/bboxes_txt/{bbox_file.name}"
        
        records.append({
            'filepath': filepath_rel,
            'filename': img_file.name,
            'cow_id': f"{cow_id:03d}",
            'view': view,
            'bbox_txt_path': bbox_rel
        })

df_main = pd.DataFrame(records)
main_csv_path = base_dir / "main" / "main_index.csv"
df_main.to_csv(main_csv_path, index=False)

print(f"Created main_index.csv: {len(df_main)} records, {df_main['cow_id'].nunique()} IDs")

test_ids = list(range(51, 76))
test_ids_str = [f"{i:03d}" for i in test_ids]

remaining_ids = [i for i in range(1, 216) if i not in test_ids]
remaining_ids_str = [f"{i:03d}" for i in remaining_ids]

print(f"Test IDs (fixed): {len(test_ids)} cows")
print(f"Train/Val IDs: {len(remaining_ids)} cows")

np.random.seed(42)
np.random.shuffle(remaining_ids_str)

fold_size = len(remaining_ids_str) // 5

for fold in range(5):
    val_start = fold * fold_size
    val_end = val_start + fold_size
    
    val_ids = remaining_ids_str[val_start:val_end]
    train_ids = [id_ for id_ in remaining_ids_str if id_ not in val_ids]
    
    train_df = df_main[df_main['cow_id'].isin(train_ids)]
    val_df = df_main[df_main['cow_id'].isin(val_ids)]
    test_df = df_main[df_main['cow_id'].isin(test_ids_str)]
    
    train_df.to_csv(splits_dir / f"fold{fold}_train.csv", index=False)
    val_df.to_csv(splits_dir / f"fold{fold}_val.csv", index=False)
    test_df.to_csv(splits_dir / f"fold{fold}_test.csv", index=False)
    
    print(f"Fold {fold}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

test_datasets = {
    'cross_angle': base_dir / 'Test Datasets Identification' / 'cross_angle',
    'hard': base_dir / 'Test Datasets Identification' / 'hard',
    'unknown': base_dir / 'Test Datasets Identification' / 'unknown'
}

for test_name, test_dir in test_datasets.items():
    if not test_dir.exists():
        continue
    
    records = []
    for img_file in test_dir.glob("*.jpg"):
        records.append({
            'filepath': str(img_file.relative_to(base_dir)),
            'filename': img_file.name,
            'test_set': test_name
        })
    
    if records:
        df = pd.DataFrame(records)
        csv_path = base_dir / f"test_{test_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Created test_{test_name}.csv: {len(df)} images")

print("\nCSV generation complete!")
