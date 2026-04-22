# Thesis Codebase: Cattle Vision Tasks

This repository contains three related computer vision tracks for cattle research:

1. Disease classification (multi-option training and cross-validation)
2. Cow identification / re-identification (detection + metric learning pipeline)
3. Weight estimation (deep-learning regression notebook + model comparison outputs)

## Repository Layout

```text
thesis_code_cattle/
├── Disease/
│   └── Codes/
├── identification/
│   ├── configs/
│   ├── scripts/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
└── weight_estimation/
    ├── cattle-weight-prediction-50-epoch.ipynb
    ├── cattleMetadata.csv
    ├── model_comparison_output.csv
    └── results-epoch-50.txt
```

## Quick Start

Use a separate virtual environment per module (recommended), because `identification` is PyTorch-based while `weight_estimation` is TensorFlow-based.

### 1) Disease Module

Path: `Disease/Codes/`

What it includes:
- Data prep and split creation (`build_metadata.py`, `create_splits.py`)
- Multiple modeling options (`option_a.py` ... `option_e.py`)
- CV and evaluation scripts (`run_5fold_cv.py`, analysis/visualization scripts)

Typical workflow:

```bash
cd Disease/Codes
python check_env.py
python create_splits.py
python option_a.py
python run_5fold_cv.py
python generate_test_predictions.py
python generate_comprehensive_analysis.py
```

Notes:
- Several scripts use hardcoded base paths (for example `e:\...` or `d:\...`). Update those paths to your local dataset/project locations before running.
- Fold and test split CSVs are already present (`folds.csv`, `test_split.csv`) if you want to start from existing splits.

### 2) Identification Module

Path: `identification/`

This module implements a pipeline for cow re-identification, including YOLO-based crop generation and embedding-based ReID training/evaluation.

Setup:

```bash
cd identification
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Typical pipeline:

```bash
python scripts/preprocessing/step1_build_csvs.py
python scripts/preprocessing/generate_yolo_predictions.py
python scripts/training/train.py --fold 0 --backbone resnet50 --loss combined
python scripts/training/step2_3_4_evaluate_export.py
python scripts/evaluation/evaluate_protocolA_rotation.py
python scripts/evaluation/evaluate_protocolBC.py
python scripts/evaluation/aggregate_final_results.py
```

Important files:
- Best config templates: `configs/best_resnet50.yaml`, `configs/best_convnext_tiny.yaml`
- Detector weights: `configs/yolov8s.pt`, `configs/yolo26n.pt`

### 3) Weight Estimation Module

Path: `weight_estimation/`

Main artifact:
- `cattle-weight-prediction-50-epoch.ipynb`

The notebook trains and compares multiple CNN backbones (including fine-tuned variants and ensembles) for weight regression.

Setup (example):

```bash
cd weight_estimation
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn pillow jupyter
jupyter notebook
```

Then open and run:
- `cattle-weight-prediction-50-epoch.ipynb`

Existing result artifacts:
- `model_comparison_output.csv`
- `results-epoch-50.txt`

Best validation score snapshot from existing outputs:
- `Ensemble 3: DenseNet121 v2 + InceptionV3 v2` with Val R2 approx. `0.7791`

## Environment and Reproducibility Notes

1. Use GPU-enabled environments when available (especially for `Disease` and `identification`).
2. Confirm CUDA/GPU before training (for example with scripts like `check_gpu.py` / `verify_gpu_setup.py` where available).
3. Keep dataset paths configurable. If scripts contain fixed drive paths, replace them first.
4. For fair comparison, keep split files fixed once generated.

## Suggested Workflow Order (End-to-End)

1. Finish preprocessing/splits in each module.
2. Train models (single fold first, then full CV).
3. Run evaluation and visualization scripts.
4. Aggregate final metrics into thesis tables/figures.

## License and Citation

Add your preferred license and thesis citation details here.
