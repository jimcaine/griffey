"""
data.py – Golf swing dataset & streaming data loader.

Label convention:
    0–7 → frame phase label based on which swing event segment it falls in.
          Each frame is assigned the index of the last event it has passed.
          Example: if events = [0, 47, 65, ...], then:
            frames [0, 47) → label 0 (Address)
            frames [47, 65) → label 1 (Toe-up)
            frames [65, ...) → label 2 (Mid-backswing), etc.

Bounding box convention (normalised, relative to frame W×H):
    [x_min, y_min, x_max, y_max]
"""

import ast
import logging
import os
import re
from pathlib import Path
from typing import Optional

import av                          # PyAV – GPU-free video decoding
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Event indices ────────────────────────────────────────────────────────────
EVENTS = {
    0: "Address (A)",
    1: "Toe-up (TU)",
    2: "Mid-backswing (MB)",
    3: "Top (T)",
    4: "Mid-downswing (MD)",
    5: "Impact (I)",
    6: "Mid-follow-through (MFT)",
    7: "Finish (F)",
}
TU_IDX = 1   # index inside the events list
MB_IDX = 2   # index inside the events list

# ── Image normalisation (ImageNet stats, matches pretrained backbones) ────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Blob-FS helper ───────────────────────────────────────────────────────────

class LocalBlobfs:
    """Thin wrapper around a local directory that mirrors a blob-store API."""

    def __init__(self):
        self.root_dir = Path(os.environ.get("BLOB_PREFIX", os.getcwd()))
        logger.info("LocalBlobfs root: %s", self.root_dir)

    def ls(self) -> list[str]:
        return os.listdir(self.root_dir)

    def read_csv(self, key: str) -> pd.DataFrame:
        return pd.read_csv(self.root_dir / key)

    def write_csv(self, df: pd.DataFrame, key: str) -> None:
        df.to_csv(self.root_dir / key, index=False)

    def video_path(self, video_id: int) -> Path:
        return self.root_dir / "videos_160" / f"{video_id}.mp4"


# ── Label loading ────────────────────────────────────────────────────────────

def _parse_bbox(s: str) -> list[float]:
    s_clean = s.strip()[1:-1]
    parts = re.split(r"[,\s]+", s_clean)
    return [float(p) for p in parts if p]


def read_train_data(blobfs: LocalBlobfs) -> pd.DataFrame:
    df = blobfs.read_csv("GolfDB.csv")

    for col in ["Unnamed: 0"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df["events"] = df["events"].apply(ast.literal_eval)
    df["bbox"]   = df["bbox"].apply(_parse_bbox)

    # Attach resolved video paths so the dataset never has to re-derive them
    df["video_path"] = df["id"].apply(lambda id: str(blobfs.video_path(id)))
    return df


# ── Frame-level label helper ─────────────────────────────────────────────────

def build_frame_labels(events: list[int]) -> dict[int, int]:
    """
    Return {frame_index: label} mapping frames to their event phase (0–7).

    Each frame is assigned the index of the last event it has passed.
    Frames before the first event get label 0; frames are labeled greedily
    by the most recent event boundary they cross.
    If events list is empty or malformed, returns empty dict.
    """
    if not events:
        return {}

    labels: dict[int, int] = {}
    # For each possible frame, find which event phase it's in by checking
    # which is the latest event it has passed. We don't know total frame count
    # here, so we only label frames that are referenced by events.
    max_event_frame = max(events) if events else 0
    for f in range(max_event_frame + 1):
        label = 0
        for i, evt_frame in enumerate(events):
            if f >= evt_frame:
                label = i
        labels[f] = label
    return labels


# ── Crop helper ──────────────────────────────────────────────────────────────

def crop_frame(frame_np: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Crop a HxWxC uint8 numpy array using a normalised [x0,y0,x1,y1] bbox.
    Returns the cropped region (may be empty if bbox is degenerate).
    """
    H, W = frame_np.shape[:2]
    x0 = int(bbox[0] * W)
    y0 = int(bbox[1] * H)
    x1 = int(bbox[2] * W)
    y1 = int(bbox[3] * H)

    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)

    if x1 <= x0 or y1 <= y0:
        return frame_np          # fallback: return full frame

    return frame_np[y0:y1, x0:x1]


# ── Core dataset ─────────────────────────────────────────────────────────────

class GolfSwingDataset(Dataset):
    """
    Frame-level dataset for the 8-class swing event phase classification task.

    Each item is (tensor_CHW_float32, label_int) where label ∈ {0..7}.

    Args:
        df            – DataFrame produced by read_train_data().
        input_size    – Spatial size fed to the backbone (default 224).
        augment       – Whether to apply random augmentation (train split).
        max_frames    – If set, randomly sub-sample this many frames per clip
                        during __init__ so the index stays manageable.
        video_indices – Subset of df row indices to use (for train/val split).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_size: int = 224,
        augment: bool = True,
        max_frames: Optional[int] = None,
        video_indices: Optional[list[int]] = None,
    ):
        self.input_size = input_size
        self.augment    = augment

        if video_indices is not None:
            df = df.iloc[video_indices].reset_index(drop=True)

        # Build flat index: list of (video_path, bbox, frame_idx, label)
        self.samples: list[tuple[str, list[float], int, int]] = []

        for _, row in df.iterrows():
            vpath  = row["video_path"]
            bbox   = row["bbox"]
            events = row["events"]

            # Count frames without decoding via PyAV container probe
            total_frames = _probe_frame_count(vpath)
            if total_frames == 0:
                logger.warning("Skipping %s – could not probe frame count", vpath)
                continue

            label_map = build_frame_labels(events)

            all_frames = list(range(total_frames))
            if max_frames and len(all_frames) > max_frames:
                rng = np.random.default_rng(42)
                all_frames = sorted(rng.choice(all_frames, max_frames, replace=False).tolist())

            for fidx in all_frames:
                label = label_map.get(fidx, 0)  # default to label 0 if out of events range
                self.samples.append((vpath, bbox, fidx, label))

        class_counts = [0] * 8
        for s in self.samples:
            class_counts[s[3]] += 1
        logger.info(
            "Dataset built: %d samples | class dist: %s",
            len(self.samples), class_counts,
        )

        # Transforms
        crop_and_resize = [
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
        ]
        aug_ops = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ] if augment else []

        self.transform = transforms.Compose([
            *crop_and_resize,
            *aug_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        vpath, bbox, frame_idx, label = self.samples[idx]

        frame_np = _decode_single_frame(vpath, frame_idx)   # HxWxC uint8
        cropped  = crop_frame(frame_np, bbox)
        tensor   = self.transform(cropped)

        return tensor, label

    # ── Class-weight helper for imbalanced data ──────────────────────────────
    def class_weights(self, num_classes: int = 8) -> torch.Tensor:
        """Returns per-class inverse frequency weights for CrossEntropyLoss."""
        class_counts = [0] * num_classes
        for s in self.samples:
            class_counts[s[3]] += 1
        total = sum(class_counts)
        if total == 0:
            return torch.ones(num_classes)
        weights = [total / (num_classes * (c or 1)) for c in class_counts]
        return torch.tensor(weights, dtype=torch.float32)


# ── Streaming video-level DataLoader ─────────────────────────────────────────

class VideoStreamDataset(Dataset):
    """
    Streams videos one at a time, yielding ALL frames of a single clip
    as a batch.  Use with batch_size=1 in DataLoader and collate below.

    This is memory-efficient for long videos: only one video is decoded
    at a time and frames are yielded as a (N, C, H, W) tensor.

    Returns: (frames_tensor, labels_tensor, video_path)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_size: int = 224,
        augment: bool = False,
    ):
        self.records: list[tuple[str, list[float], dict[int, int]]] = []
        for _, row in df.iterrows():
            label_map = build_frame_labels(row["events"])
            self.records.append((row["video_path"], row["bbox"], label_map))

        resize_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform = resize_norm

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        vpath, bbox, label_map = self.records[idx]

        frames_t: list[torch.Tensor] = []
        labels:   list[int]          = []

        try:
            container = av.open(vpath)
            stream    = container.streams.video[0]
            stream.thread_type = "AUTO"

            for fi, packet in enumerate(container.decode(stream)):
                frame_np = packet.to_ndarray(format="rgb24")
                cropped  = crop_frame(frame_np, bbox)
                frames_t.append(self.transform(cropped))
                labels.append(label_map.get(fi, 0))

            container.close()
        except Exception as exc:
            logger.error("Error decoding %s: %s", vpath, exc)
            # Return empty tensors so the collate fn can skip
            return torch.zeros(0), torch.zeros(0, dtype=torch.long), vpath

        return (
            torch.stack(frames_t),              # (N, C, H, W)
            torch.tensor(labels, dtype=torch.long),
            vpath,
        )


def video_stream_collate(batch):
    """Collate for VideoStreamDataset – keeps each video as its own item."""
    return batch   # list of (frames, labels, path) – do NOT stack across videos


# ── Low-level video utilities ─────────────────────────────────────────────────

def _probe_frame_count(video_path: str) -> int:
    """Return approximate frame count from container metadata (fast, no decode)."""
    try:
        container = av.open(video_path)
        stream    = container.streams.video[0]
        count     = stream.frames
        if count and count > 0:
            container.close()
            return int(count)
        # Fallback: iterate (slow but reliable)
        count = sum(1 for _ in container.decode(stream))
        container.close()
        return count
    except Exception as exc:
        logger.error("Could not probe %s: %s", video_path, exc)
        return 0


def _decode_single_frame(video_path: str, target_frame: int) -> np.ndarray:
    """
    Decode and return a single frame by index as a (H, W, 3) uint8 ndarray.

    Strategy: seek to the nearest keyframe then step forward.  This is much
    faster than decoding from the beginning for large frame indices.
    """
    try:
        container = av.open(video_path)
        stream    = container.streams.video[0]
        stream.thread_type = "AUTO"

        fps      = float(stream.average_rate or 30)
        duration = stream.duration  # in stream time_base units
        tb       = float(stream.time_base)

        if duration:
            # Seek to ~1 sec before the target frame
            seek_ts = max(0, int((target_frame / fps - 1.0) / tb))
            container.seek(seek_ts, stream=stream, backward=True, any_frame=False)

        frame_np = None
        for fi, frame in enumerate(container.decode(stream)):
            # After seeking we may start a few frames before the target
            current = frame.pts or 0
            current_idx = int(current * tb * fps)
            if current_idx >= target_frame or fi >= target_frame + 5:
                frame_np = frame.to_ndarray(format="rgb24")
                break
            # Always keep the last decoded frame as fallback
            frame_np = frame.to_ndarray(format="rgb24")

        container.close()

        if frame_np is None:
            raise ValueError(f"Frame {target_frame} not found in {video_path}")
        return frame_np

    except Exception as exc:
        logger.error("Decode error %s frame %d: %s", video_path, target_frame, exc)
        # Return a blank frame so training doesn't crash
        return np.zeros((160, 160, 3), dtype=np.uint8)


# ── Factory helpers ───────────────────────────────────────────────────────────

def make_dataloaders(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
    max_frames_per_video: Optional[int] = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Split df by *video* (not frame) to avoid leakage, then build
    frame-level DataLoaders.

    Returns (train_loader, val_loader).
    """
    rng = np.random.default_rng(seed)
    n   = len(df)
    idx = rng.permutation(n)
    n_val   = max(1, int(n * val_fraction))
    val_idx = sorted(idx[:n_val].tolist())
    trn_idx = sorted(idx[n_val:].tolist())

    logger.info("Train videos: %d | Val videos: %d", len(trn_idx), len(val_idx))

    train_ds = GolfSwingDataset(df, input_size=input_size, augment=True,
                                max_frames=max_frames_per_video,
                                video_indices=trn_idx)
    val_ds   = GolfSwingDataset(df, input_size=input_size, augment=False,
                                max_frames=max_frames_per_video,
                                video_indices=val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test of dataset loading
    logging.basicConfig(level=logging.INFO)
    blobfs = LocalBlobfs()
    df = read_train_data(blobfs)
    print(df)
    # train_loader, val_loader = make_dataloaders(df, batch_size=4, num_workers=0)

    # for batch in train_loader:
    #     images, labels = batch
    #     print("Batch images shape:", images.shape)
    #     print("Batch labels:", labels)
    #     break
