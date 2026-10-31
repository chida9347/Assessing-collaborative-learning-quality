import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import ensure_dirs, save_json, load_json


class DatasetManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cfg = config.get("dataset", {})
        self.splits_dir = self.data_cfg.get("splits_dir", "data/splits")
        self.raw_dir = self.data_cfg.get("raw_dir", "data/raw")
        self.class_names = self.data_cfg.get("class_names", ["Peer Discussion", "Shared Task Activity", "Presentation"])
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}

    def discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        # Try to find videos in raw_dir organized by class or flat
        raw = Path(self.raw_dir)
        if not raw.exists():
            return samples
        # Check class subdirs
        subdirs = [d for d in raw.iterdir() if d.is_dir()]
        if len(subdirs) > 0:
            for sd in subdirs:
                cls = sd.name
                cls_mapped = None
                # direct match
                if cls in self.label_to_idx:
                    cls_mapped = cls
                else:
                    # case-insensitive direct
                    for c in self.class_names:
                        if c.lower() == cls.lower():
                            cls_mapped = c
                            break
                    # common aliases
                    if cls_mapped is None:
                        cl = cls.lower()
                        if cl in ["mid", "medium", "med"]:
                            for c in self.class_names:
                                if "shared" in c.lower():
                                    cls_mapped = c
                                    break
                        elif cl in ["low"]:
                            for c in self.class_names:
                                if "discussion" in c.lower():
                                    cls_mapped = c
                                    break
                        elif cl in ["high"]:
                            for c in self.class_names:
                                if "presentation" in c.lower():
                                    cls_mapped = c
                                    break
                for vf in sd.glob("**/*"):
                    if vf.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
                        label = cls_mapped if cls_mapped and cls_mapped in self.label_to_idx else None
                        samples.append({
                            "id": vf.stem,
                            "path": str(vf),
                            "label": label,
                            "label_idx": self.label_to_idx.get(label) if label else None,
                            "source_dir": str(sd),
                        })
            return samples
        # Flat
        for vf in raw.glob("**/*"):
            if vf.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
                samples.append({
                    "id": vf.stem,
                    "path": str(vf),
                    "label": None,
                    "label_idx": None,
                    "source_dir": str(raw),
                })
        return samples

    def create_splits(self, samples: List[Dict[str, Any]], test_size: float = 0.1, val_size: float = 0.1, seed: int = 42) -> Dict[str, Any]:
        # Filter labeled
        labeled = [s for s in samples if s.get("label_idx") is not None]
        if len(labeled) == 0:
            return {"train": [], "val": [], "test": [], "note": "No labeled samples found"}
        labels = [s["label_idx"] for s in labeled]
        # Stratified split
        train_val, test = train_test_split(labeled, test_size=test_size, stratify=labels, random_state=seed)
        val_ratio = val_size / (1 - test_size) if (1 - test_size) > 0 else 0.1
        train_val_labels = [s["label_idx"] for s in train_val]
        train, val = train_test_split(train_val, test_size=val_ratio, stratify=train_val_labels, random_state=seed)
        manifest = {
            "seed": seed,
            "class_names": self.class_names,
            "label_to_idx": self.label_to_idx,
            "splits": {
                "train": [{"id": s["id"], "path": s["path"], "label": s["label"], "label_idx": s["label_idx"]} for s in train],
                "val": [{"id": s["id"], "path": s["path"], "label": s["label"], "label_idx": s["label_idx"]} for s in val],
                "test": [{"id": s["id"], "path": s["path"], "label": s["label"], "label_idx": s["label_idx"]} for s in test],
            },
            "counts": {"train": len(train), "val": len(val), "test": len(test), "total_labeled": len(labeled)},
        }
        ensure_dirs([self.splits_dir])
        save_json(manifest, os.path.join(self.splits_dir, "splits.json"))
        return manifest

    def load_splits(self) -> Dict[str, Any]:
        path = os.path.join(self.splits_dir, "splits.json")
        if os.path.exists(path):
            return load_json(path)
        return {}
