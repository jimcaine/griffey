# Griffey: Golf Swing Phase Classifier
**Task:** Binary frame-level classification – is this frame between *Toe-up (TU)* and *Mid-backswing (MB)*?

## File overview

| File | Purpose |
|---|---|
| `data.py` | Label parsing, bbox cropping, `GolfSwingDataset`, `VideoStreamDataset`, `make_dataloaders` |
| `model.py` | `GolfSwingClassifier` (EfficientNet-V2-S backbone + custom head) |
| `train.py` | Two-phase training loop, checkpointing, CLI entry-point |
| `requirements.txt` | Python dependencies |

---

## Architecture

### Backbone – EfficientNet-V2-S
Chosen over MobileNetV2 for the following reasons:

| | MobileNetV2 | EfficientNet-V2-S |
|---|---|---|
| ImageNet top-1 | 71.8 % | 83.9 % |
| Params | 3.4 M | 21.5 M |
| Fused-MBConv | ✗ | ✓ |
| Trained with progressive resizing | ✗ | ✓ |

Still fits comfortably in 12 GB VRAM at `batch_size=32, input_size=224`.  
Swap to `--backbone convnext_tiny` for a slightly different inductive bias (same accuracy tier).

### Head
```
BN(in_features) → Dropout(0.3) → Linear(→256) → GELU → Dropout(0.15) → Linear(→1)
```
Output is a raw logit; use `sigmoid` at inference time.

---

## Labelling logic

```python
events = [0, 47, 65, 68, 82, 87, 90, 93, 106, 137]
#              ↑TU  ↑MB
# Frames 48..64 are labelled 1, all others 0.
```

`TU = events[1]`, `MB = events[2]` (exclusive range: `TU < frame < MB`).

---

## Training

### Install
```bash
# For CUDA (NVIDIA GPU):
pip install -e .

# For ROCm (AMD GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install -e .

# For CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Run
```bash
# Set the root directory that contains GolfDB.csv and videos_160/
export BLOB_PREFIX=/path/to/data

python train.py \
  --batch_size 32 \
  --warmup_epochs 3 \
  --finetune_epochs 20 \
  --checkpoint_dir ./checkpoints
```

### Two-phase training
1. **Warm-up** (`--warmup_epochs`, default 3): backbone frozen, only the head trains at `lr=3e-4`.  
   Avoids destroying pretrained features before the head learns a reasonable representation.
2. **Fine-tune** (`--finetune_epochs`, default 20): full network unfrozen, lower `lr=5e-5` with cosine annealing.

### Handling class imbalance
`BCEWithLogitsLoss(pos_weight=neg/pos)` is computed automatically from the training split.  
For a typical clip, ~10–20 % of frames fall in the TU→MB window, giving `pos_weight ≈ 5–9`.

---

## Streaming data loader

For inference on full videos without building a flat frame index:

```python
from data import VideoStreamDataset, video_stream_collate
from torch.utils.data import DataLoader

ds = VideoStreamDataset(df, input_size=224)
loader = DataLoader(ds, batch_size=1, collate_fn=video_stream_collate,
                    num_workers=2)

for batch in loader:
    (frames, labels, path), = batch   # frames: (N, C, H, W)
    logits = model(frames.to(device))
    probs  = torch.sigmoid(logits)
```

---

## AMD / ROCm note
PyAV decodes on CPU so there are no ROCm-specific concerns for video I/O.  
Mixed-precision (`torch.amp`) works on ROCm via the same `GradScaler` API.  
If ROCm doesn't support `autocast` on your card set `--amp false`.

---

## Checkpoints
Saved to `--checkpoint_dir` (default `./checkpoints`):
- `best_model.pt` – highest validation F1 seen so far  
- `epoch_NNN.pt` – end of every epoch

Load for inference:
```python
from model import build_model
from train import load_checkpoint

model = build_model().to(device)
model = load_checkpoint(model, "checkpoints/best_model.pt", device)
```


## Data & References
* Kaggle : https://www.kaggle.com/datasets/marcmarais/videos-160
* GolfDB : https://github.com/wmcnally/golfdb
