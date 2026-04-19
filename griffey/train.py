"""
train.py – Training loop, evaluation, checkpointing.

Usage (quick start):
    BLOB_PREFIX=/data/golf python train.py

Key design decisions
────────────────────
* Multi-class (8-event) classification with CrossEntropyLoss.
* Per-class weights to handle imbalanced event distribution.
* Two-phase training:
    Phase 1 (warm-up)  – backbone frozen, only head trains for WARMUP_EPOCHS.
    Phase 2 (fine-tune)– entire network unfrozen, lower LR.
* Mixed-precision (torch.amp) for memory efficiency on the 12 GB GPU.
* Cosine annealing LR schedule in fine-tune phase.
* Best checkpoint saved by validation F1 (macro-averaged for multi-class).
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
)

from griffey.data  import LocalBlobfs, read_train_data, make_dataloaders
from griffey.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Defaults (override via CLI or environment) ────────────────────────────────
DEFAULTS = dict(
    backbone          = "efficientnet_v2_s",
    input_size        = 224,
    batch_size        = 32,
    num_workers       = 4,
    warmup_epochs     = 3,
    finetune_epochs   = 20,
    warmup_lr         = 3e-4,
    finetune_lr       = 5e-5,
    weight_decay      = 1e-4,
    dropout           = 0.3,
    freeze_until      = 5,
    val_fraction      = 0.15,
    max_frames        = 120,    # sub-sample frames per video during indexing
    checkpoint_dir    = "checkpoints",
    amp               = True,
)


# ── Metric helpers ────────────────────────────────────────────────────────────

def make_metrics(device, num_classes: int = 8):
    return {
        "acc"  : MulticlassAccuracy(num_classes=num_classes, average="macro").to(device),
        "f1"   : MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "prec" : MulticlassPrecision(num_classes=num_classes, average="macro").to(device),
        "rec"  : MulticlassRecall(num_classes=num_classes, average="macro").to(device),
        "auroc": MulticlassAUROC(num_classes=num_classes, average="macro").to(device),
    }


def reset_metrics(metrics: dict) -> None:
    for m in metrics.values():
        m.reset()


def compute_metrics(metrics: dict) -> dict[str, float]:
    return {k: float(v.compute()) for k, v in metrics.items()}


# ── One epoch ────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, criterion, optimizer,
    device, scaler, metrics,
    is_train: bool,
    epoch: int,
) -> dict[str, float]:
    model.train() if is_train else model.eval()
    reset_metrics(metrics)

    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    amp_device_type = device.type if hasattr(device, "type") else "cuda"

    with ctx:
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            with autocast(device_type=amp_device_type, enabled=scaler is not None):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            probs = torch.softmax(logits, dim=1).detach()
            preds = torch.argmax(probs, dim=1)
            # AUROC needs probabilities; others need class indices
            metrics["auroc"].update(probs, labels)
            for k, m in metrics.items():
                if k != "auroc":
                    m.update(preds, labels)

            if is_train and (batch_idx + 1) % 50 == 0:
                logger.info(
                    "  Epoch %d | batch %d/%d | loss=%.4f",
                    epoch, batch_idx + 1, len(loader), total_loss / n_batches,
                )

    results = compute_metrics(metrics)
    results["loss"] = total_loss / max(n_batches, 1)
    results["time"] = time.time() - t0
    return results


# ── Training entry-point ──────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ────────────────────────────────────────────────────────────────
    blobfs = LocalBlobfs()
    df     = read_train_data(blobfs)
    logger.info("Loaded metadata: %d clips", len(df))

    train_loader, val_loader = make_dataloaders(
        df,
        val_fraction         = cfg["val_fraction"],
        batch_size           = cfg["batch_size"],
        num_workers          = cfg["num_workers"],
        input_size           = cfg["input_size"],
        max_frames_per_video = cfg["max_frames"],
    )

    # Infer number of classes from training dataset
    train_ds = train_loader.dataset
    max_label = max((s[3] for s in train_ds.samples), default=7)
    num_classes = max(8, max_label + 1)
    logger.info("Number of event classes detected: %d", num_classes)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes  = num_classes,
        backbone     = cfg["backbone"],
        dropout      = cfg["dropout"],
        freeze_until = cfg["freeze_until"],
    ).to(device)

    class_weights = train_loader.dataset.class_weights(num_classes=num_classes).to(device)
    logger.info("class_weights for CELoss: %s", class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Phase 1: warm-up (head only) ─────────────────────────────────────────
    scaler  = GradScaler() if (cfg["amp"] and device.type == "cuda") else None
    metrics = make_metrics(device, num_classes=num_classes)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1  = -1.0
    best_ckpt    = ckpt_dir / "best_model.pt"

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(head_params, lr=cfg["warmup_lr"],
                        weight_decay=cfg["weight_decay"])

    logger.info("=== Phase 1: warm-up (%d epochs, head only) ===",
                cfg["warmup_epochs"])

    for epoch in range(1, cfg["warmup_epochs"] + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer,
                       device, scaler, metrics, is_train=True, epoch=epoch)
        va = run_epoch(model, val_loader, criterion, None,
                       device, scaler, metrics, is_train=False, epoch=epoch)
        _log_epoch(epoch, tr, va, phase="warm-up")

        if va["f1"] > best_val_f1:
            best_val_f1 = va["f1"]
            _save_checkpoint(model, optimizer, epoch, va, best_ckpt)

    # ── Phase 2: fine-tune (full network) ────────────────────────────────────
    model.unfreeze_all()
    logger.info("=== Phase 2: fine-tune (%d epochs, full net) ===",
                cfg["finetune_epochs"])

    optimizer = AdamW(model.parameters(), lr=cfg["finetune_lr"],
                      weight_decay=cfg["weight_decay"])

    T_max   = cfg["finetune_epochs"]
    warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=max(1, T_max // 5))
    cosine_sched  = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-7)
    scheduler     = SequentialLR(optimizer,
                                 schedulers=[warmup_sched, cosine_sched],
                                 milestones=[max(1, T_max // 5)])

    for epoch in range(cfg["warmup_epochs"] + 1,
                       cfg["warmup_epochs"] + cfg["finetune_epochs"] + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer,
                       device, scaler, metrics, is_train=True, epoch=epoch)
        va = run_epoch(model, val_loader, criterion, None,
                       device, scaler, metrics, is_train=False, epoch=epoch)
        scheduler.step()
        _log_epoch(epoch, tr, va, phase="fine-tune",
                   lr=scheduler.get_last_lr()[0])

        if va["f1"] > best_val_f1:
            best_val_f1 = va["f1"]
            _save_checkpoint(model, optimizer, epoch, va, best_ckpt)

        # Per-epoch checkpoint
        _save_checkpoint(model, optimizer, epoch, va,
                         ckpt_dir / f"epoch_{epoch:03d}.pt")

    logger.info("Training complete.  Best val F1=%.4f  checkpoint=%s",
                best_val_f1, best_ckpt)


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_video(
    model: nn.Module,
    video_path: str,
    bbox: list[float],
    device: torch.device,
    input_size: int = 224,
) -> list[tuple[int, int, list[float]]]:
    """
    Run inference on every frame of a video.

    Returns list of (frame_idx, predicted_class, class_probs).
    class_probs is a list of 8 softmax scores.
    """
    from torchvision import transforms
    from data import crop_frame, IMAGENET_MEAN, IMAGENET_STD
    import av

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    model.eval()
    results = []

    container = av.open(video_path)
    stream    = container.streams.video[0]

    with torch.no_grad():
        for fi, frame in enumerate(container.decode(stream)):
            frame_np = frame.to_ndarray(format="rgb24")
            cropped  = crop_frame(frame_np, bbox)
            tensor   = tf(cropped).unsqueeze(0).to(device)

            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]  # (8,)
            pred   = int(torch.argmax(probs).item())
            results.append((fi, pred, probs.cpu().tolist()))

    container.close()
    return results


# ── Utilities ─────────────────────────────────────────────────────────────────

def _log_epoch(epoch, tr, va, phase="", lr=None):
    lr_str = f"  lr={lr:.2e}" if lr else ""
    logger.info(
        "Epoch %3d [%s]%s\n"
        "  train | loss=%.4f  acc=%.3f  f1=%.3f  prec=%.3f  rec=%.3f  auroc=%.3f\n"
        "  val   | loss=%.4f  acc=%.3f  f1=%.3f  prec=%.3f  rec=%.3f  auroc=%.3f  (%.1fs)",
        epoch, phase, lr_str,
        tr["loss"], tr["acc"], tr["f1"], tr["prec"], tr["rec"], tr["auroc"],
        va["loss"], va["acc"], va["f1"], va["prec"], va["rec"], va["auroc"],
        va["time"],
    )


def _save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch"     : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics"   : metrics,
    }, path)
    logger.debug("Saved checkpoint → %s", path)


def load_checkpoint(model, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info("Loaded checkpoint epoch=%d  val_f1=%.4f",
                ckpt["epoch"], ckpt["metrics"].get("f1", -1))
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Train 8-class golf swing event phase classifier")
    for key, val in DEFAULTS.items():
        t = type(val) if val is not None else str
        if t == bool:
            p.add_argument(f"--{key}", default=val,
                           type=lambda x: x.lower() != "false")
        else:
            p.add_argument(f"--{key}", default=val, type=t)
    args = p.parse_args()
    return vars(args)


if __name__ == "__main__":
    cfg = parse_args()
    logger.info("Config: %s", cfg)
    train(cfg)
