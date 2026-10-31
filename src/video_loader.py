import os
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np


class VideoLoader:
    def __init__(self):
        pass

    def get_metadata(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return {"valid": False, "error": "Cannot open video"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return {
            "valid": True,
            "filename": os.path.basename(video_path),
            "path": video_path,
            "fps": float(fps),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": float(duration),
            "resolution": f"{width}x{height}",
        }

    def sample_frames(self, video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
        meta = self.get_metadata(video_path)
        if not meta.get("valid"):
            return {"valid": False, "frames": [], "metadata": meta}
        cap = cv2.VideoCapture(video_path)
        total = meta["frame_count"]
        if total <= 0:
            cap.release()
            return {"valid": False, "frames": [], "metadata": meta, "error": "No frames"}
        if total < num_frames:
            # sample all and pad by repeating last
            indices = list(range(total))
            while len(indices) < num_frames:
                indices.append(total - 1 if total > 0 else 0)
        else:
            # uniform sampling
            step = total / num_frames
            indices = [int(i * step) for i in range(num_frames)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # try to read next if failed
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            return {"valid": False, "frames": [], "metadata": meta, "error": "Failed to read frames"}
        # pad if still short
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        frames = frames[:num_frames]
        return {"valid": True, "frames": np.array(frames), "metadata": meta, "indices": indices}
