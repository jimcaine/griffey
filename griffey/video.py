import logging

import numpy as np
import av

logger = logging.getLogger(__name__)


def crop_frame(frame_np: np.ndarray, bbox: list[float]) -> np.ndarray:
    """ Crop a HxWxC uint8 numpy array using a normalised [x0,y0,x1,y1] bbox.
    Returns the cropped region; if the bbox is degenerate after clipping,
    returns the full frame as a fallback.
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


def probe_frame_count(video_path: str) -> int:
    """ Return approximate frame count from container metadata. """
    try:
        container = av.open(video_path)
    except Exception as exc:
        logger.error("Could not open %s: %s", video_path, exc)
        return 0

    try:
        stream    = container.streams.video[0]
        count     = stream.frames
        if count and count > 0:
            container.close()
            return int(count)
        # Fallback: iterate (slow but reliable)
        count = sum(1 for _ in container.decode(stream))
        return count
    except Exception as exc:
        logger.error("Could not probe %s: %s", video_path, exc)
        return 0
    finally:
        container.close()


def decode_single_frame(video_path: str, target_frame: int) -> np.ndarray:
    """
    Decode and return a single frame by index as a (H, W, 3) uint8 ndarray.

    Strategy: seek to the nearest keyframe then step forward.  This is much
    faster than decoding from the beginning for large frame indices.
    """
    try:
        container = av.open(video_path)
    except Exception as e:
        logger.error("Could not open %s: %s", video_path, e)
        # Return a blank frame so training doesn't crash
        return np.zeros((160, 160, 3), dtype=np.uint8)

    try:
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


        if frame_np is None:
            raise ValueError("Frame %d not found in %s", target_frame, video_path)
        return frame_np

    except Exception as exc:
        logger.error("Decode error %s frame %d: %s", video_path, target_frame, exc)
        # Return a blank frame so training doesn't crash
        return np.zeros((160, 160, 3), dtype=np.uint8)

    finally:
        container.close()
