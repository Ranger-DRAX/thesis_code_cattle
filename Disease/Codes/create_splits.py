"""
Create fixed split + 5-fold CV for disease classification
Step 2: Ensure fair comparison across all training methods
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

# Base directory
BASE_DIR = Path(r"e:\Disease Classification")
PROJECT_DIR = BASE_DIR / "Project"
SPLITS_DIR = PROJECT_DIR / "splits"

# Ensure splits directory exists
SPLITS_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_splits():
    """
    Create fixed test/train split and 5-fold CV splits
    """
    
    # Load metadata
    metadata_path = PROJECT_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)
    
    print("=" * 70)
    print("STEP 2: Creating Fixed Split + 5-Fold Cross-Validation")
    print("=" * 70)
    print(f"\nLoaded metadata: {len(df)} images")
    print(f"\nLabel_10 distribution:")
    print(df['label_10'].value_counts().sort_index())
    
    # ===================================================================
    # 2.1 Hold-out test set (15%, stratified by label_10)
    # ===================================================================
    print("\n" + "=" * 70)
    print("2.1 Creating Hold-out Test Set (15%)")
    print("=" * 70)
    
    # Stratified split: 85% train/val, 15% test
    train_val_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=0.15,
        stratify=df['label_10'],
        random_state=RANDOM_SEED
    )
    
    # Create test split dataframe
    df_test = df.iloc[test_indices].copy()
    df_test = df_test.sort_values('filepath').reset_index(drop=True)
    
    # Save test split
    test_split_path = SPLITS_DIR / "test_split.csv"
    df_test.to_csv(test_split_path, index=False)
    
    print(f"\n✓ Test set created: {len(df_test)} images ({len(df_test)/len(df)*100:.1f}%)")
    print(f"  Saved to: {test_split_path}")
    print(f"\nTest set label_10 distribution:")
    print(df_test['label_10'].value_counts().sort_index())
    
    # ===================================================================
    # 2.2 5-Fold Cross-Validation (on remaining 85%, stratified by label_10)
    # ===================================================================
    print("\n" + "=" * 70)
    print("2.2 Creating 5-Fold Cross-Validation Splits (85%)")
    print("=" * 70)
    
    # Get train/val subset
    df_train_val = df.iloc[train_val_indices].copy().reset_index(drop=True)
    
    print(f"\nTrain/Val set: {len(df_train_val)} images ({len(df_train_val)/len(df)*100:.1f}%)")
    
    # Create stratified 5-fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Add fold column (initialize with -1)
    df_train_val['fold'] = -1
    
    # Assign fold numbers
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val['label_10'])):
        df_train_val.loc[val_idx, 'fold'] = fold_idx
    
    # Verify all samples are assigned
    assert (df_train_val['fold'] != -1).all(), "Some samples not assigned to folds!"
    
    # Sort by filepath for consistency
    df_train_val = df_train_val.sort_values('filepath').reset_index(drop=True)
    
    # Save folds
    folds_path = SPLITS_DIR / "folds.csv"
    df_train_val.to_csv(folds_path, index=False)
    
    print(f"\n✓ 5-fold CV splits created")
    print(f"  Saved to: {folds_path}")
    
    # Print fold statistics
    print(f"\nFold distribution:")
    for fold in range(5):
        fold_df = df_train_val[df_train_val['fold'] == fold]
        print(f"  Fold {fold}: {len(fold_df)} images ({len(fold_df)/len(df_train_val)*100:.1f}%)")
    
    # Verify stratification per fold
    print(f"\nLabel_10 distribution per fold:")
    for fold in range(5):
        fold_df = df_train_val[df_train_val['fold'] == fold]
        print(f"\n  Fold {fold}:")
        label_counts = fold_df['label_10'].value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"    {label}: {count}")
    
    # ===================================================================
    # Summary Statistics
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal dataset: {len(df)} images")
    print(f"  Test set (15%): {len(df_test)} images")
    print(f"  Train/Val set (85%): {len(df_train_val)} images")
    print(f"    → Split into 5 folds for cross-validation")
    print(f"    → Each fold: ~{len(df_train_val)//5} images")
    
    print(f"\nDisease distribution:")
    print(f"  Full dataset:")
    for disease in ['healthy', 'lsd', 'fmd', 'ibk']:
        count = len(df[df['disease'] == disease])
        print(f"    {disease}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n  Test set:")
    for disease in ['healthy', 'lsd', 'fmd', 'ibk']:
        count = len(df_test[df_test['disease'] == disease])
        print(f"    {disease}: {count} ({count/len(df_test)*100:.1f}%)")
    
    print(f"\n  Train/Val set:")
    for disease in ['healthy', 'lsd', 'fmd', 'ibk']:
        count = len(df_train_val[df_train_val['disease'] == disease])
        print(f"    {disease}: {count} ({count/len(df_train_val)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✓ Step 2 Completed Successfully!")
    print("=" * 70)
    print(f"\nCreated files:")
    print(f"  1. {test_split_path}")
    print(f"  2. {folds_path}")
    print(f"\nThese splits will be used for all training methods (Options A/B/C/D/E)")
    print("=" * 70)
    
    return df_test, df_train_val

if __name__ == "__main__":
    # Install required package if needed
    try:
        import sklearn
    except ImportError:
        print("Installing scikit-learn...")
        os.system('pip install scikit-learn')
    
    df_test, df_train_val = create_splits()
