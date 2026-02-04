# Cow Re-Identification Thesis Project

## Project Structure

```
identification/
├── src/                      # Source code
│   ├── models/              # Model architectures (ResNet50, ConvNeXt-Tiny)
│   ├── losses/              # Loss functions (ArcFace, SupCon, Triplet, Combined)
│   ├── data/                # Data processing (cropper, transforms, samplers)
│   ├── training/            # Training utilities
│   ├── evaluation/          # Evaluation utilities
│   └── utils/               # Utility functions
├── scripts/                 # Executable scripts
│   ├── preprocessing/       # Data preparation and YOLO prediction scripts
│   ├── training/            # Training and hyperparameter tuning scripts
│   └── evaluation/          # Evaluation and visualization scripts
├── configs/                 # Configuration files and model weights
├── data/                    # Dataset files
│   ├── raw/                # Raw images and annotations
│   └── processed/          # CSV splits and protocols
├── results/                 # Training outputs
│   ├── checkpoints/        # Saved model checkpoints
│   ├── figures/            # Generated figures
│   ├── metrics/            # Evaluation metrics
│   └── outputs/            # YOLO predictions and other outputs
└── venv/                   # Python virtual environment

## Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate environment:
```bash
# Windows
venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Data Preprocessing
```bash
# Build CSV files and splits
python scripts/preprocessing/step1_build_csvs.py

# Generate YOLO predictions
python scripts/preprocessing/generate_yolo_predictions.py

# Validate data
python scripts/preprocessing/validate_step0.py
python scripts/preprocessing/validate_step1.py
```

### 2. Training

#### Single fold training:
```bash
python scripts/training/train.py --fold 0 --backbone resnet50 --loss combined
```

#### Hyperparameter tuning:
```bash
python scripts/training/tune_fold0_resnet50.py
python scripts/training/tune_fold0_convnext_tiny.py
```

### 3. Evaluation

#### Protocol evaluation:
```bash
python scripts/evaluation/evaluate_protocolA_rotation.py
python scripts/evaluation/evaluate_protocolBC.py
```

#### Results comparison:
```bash
python scripts/evaluation/compare_backbones.py
python scripts/evaluation/compare_all_protocols.py
python scripts/evaluation/aggregate_final_results.py
```

#### Visualization:
```bash
python scripts/evaluation/visualize_results.py
python scripts/evaluation/make_figures.py
```

## Configuration

Model configurations are stored in `configs/`:
- `best_resnet50.yaml` - Best hyperparameters for ResNet50
- `best_convnext_tiny.yaml` - Best hyperparameters for ConvNeXt-Tiny
- `yolov8s.pt` - YOLO detection model weights
- `yolo26n.pt` - Alternative YOLO model

## Key Components

### Models (`src/models/`)
- ResNet-50 based Re-ID model
- ConvNeXt-Tiny based Re-ID model
- 512-dimensional embedding output
- L2 normalized features

### Loss Functions (`src/losses/`)
- **ArcFace**: Additive angular margin loss
- **SupCon**: Supervised contrastive loss
- **Triplet**: Batch-hard triplet loss
- **Combined**: SupCon + ArcFace

### Data Processing (`src/data/`)
- **cropper.py**: Image cropping with YOLO or GT bboxes
- **transforms.py**: Data augmentation pipeline
- **pk_sampler.py**: PK-sampling for metric learning

## Results

Trained models and evaluation results are saved in `results/`:
- Model checkpoints: `results/checkpoints/`
- Learning curves: `results/figures/`
- Evaluation metrics: `results/metrics/`
- YOLO predictions: `results/outputs/yolo_preds/`

## Protocols

The project evaluates 4 different protocols (stored in `data/processed/protocols/`):
- **Protocol A**: Within-ID rotation (4 views per cow)
- **Protocol B**: Cross-angle matching
- **Protocol C**: Gallery set variation (extra)
- **Protocol D**: Known vs unknown queries (extra)

## Hardware Requirements

- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB+ RAM
- 50GB+ storage

## Citation

[Add your thesis citation here]
