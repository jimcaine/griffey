# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Griffey** is a binary frame-level classifier that identifies whether a golf swing frame falls between the Toe-up (TU) and Mid-backswing (MB) phases. It uses EfficientNet-V2-S backbone with a custom classification head trained on GolfDB video data.

## Key Commands

### Training
```bash
# Source environment variables (BLOB_PREFIX, LMDB_PATH)
source .env

# Run training with default settings
python griffey/train.py

# Full training run with custom parameters
python griffey/train.py \
  --batch_size 32 \
  --warmup_epochs 3 \
  --finetune_epochs 20 \
  --backbone efficientnet_v2_s \
  --checkpoint_dir ./checkpoints
```

### Tests
```bash
# Run tests (currently no tests implemented)
make test
```

### Development
```bash
# Check project metadata
make about

# Increment patch version in pyproject.toml
make increment-patch-version

# Create PR (requires git/gh)
make pr          # PR to main
make pr-dev      # PR to dev branch
```

### Setup
```bash
# Install package + dependencies for CUDA (NVIDIA)
pip install -e .

# For ROCm (AMD GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install -e .

# For CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Architecture

### Core Data Pipeline
- **data.py**: Label parsing from CSV, bounding-box cropping, dataset classes (`GolfSwingDataset`, `VideoStreamDataset`), data loaders
- **model.py**: `GolfSwingClassifier` class wrapping EfficientNet-V2-S backbone + custom head
- **train.py**: Two-phase training loop, checkpointing, CLI entry point

### Model Architecture
- **Backbone**: EfficientNet-V2-S (chosen over MobileNetV2 for better ImageNet accuracy: 83.9% vs 71.8%, and fused-MBConv layers)
- **Head**: `BN(in_features) → Dropout(0.3) → Linear(→256) → GELU → Dropout(0.15) → Linear(→1)`
  - Output is a raw logit; apply `sigmoid` at inference time

### Training Strategy
Two-phase training to prevent destroying pretrained backbone knowledge:

1. **Warm-up** (default 3 epochs): Backbone frozen, only head trains at `lr=3e-4`
2. **Fine-tune** (default 20 epochs): Full network unfrozen, lower `lr=5e-5` with cosine annealing

Class imbalance handled via `BCEWithLogitsLoss(pos_weight=neg/pos)` computed automatically from training split (typical `pos_weight ≈ 5–9`).

### Label Convention
Golf swing events are indexed:
```
0=Address, 1=Toe-up (TU), 2=Mid-backswing (MB), 3=Top, 4=Mid-downswing, 5=Impact, 6=Mid-follow-through, 7=Finish
```
- **Label = 1**: Frame is strictly between TU (event 1) and MB (event 2)
- **Label = 0**: Everything else

### Streaming Data Loader
For inference on full videos without building frame indices:
```python
from griffey.data import VideoStreamDataset, video_stream_collate
ds = VideoStreamDataset(df, input_size=224)
loader = DataLoader(ds, batch_size=1, collate_fn=video_stream_collate, num_workers=2)
```

## Environment & Data

- **BLOB_PREFIX**: Root directory containing `GolfDB.csv` and `videos_160/` subdirectory (set in `.env`)
- **LMDB_PATH**: Optional LMDB cache path for frame indexing (set in `.env`)
- Default input size: 224×224 (matches pretrained backbone)
- Default max frames per video: 120 (subsampled during indexing)

## Important Notes

- **Video I/O**: PyAV (CPU-based) handles decoding; no GPU-specific concerns for ROCm or CUDA
- **Mixed precision**: `torch.amp` works on both CUDA and ROCm; disable with `--amp false` if needed
- **Checkpoints**: Saved to `checkpoint_dir`; best model selected by validation F1 (more robust than accuracy for imbalanced data)
- **Backbone swap**: Can swap `--backbone convnext_tiny` for different inductive bias at similar accuracy tier
