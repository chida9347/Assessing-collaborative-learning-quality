import os
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from src.utils import ensure_dirs, save_json


class FeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cnn_cfg = config.get("cnn", {})
        self.backbone_name = self.cnn_cfg.get("backbone", "MobileNetV2")
        self.weights = self.cnn_cfg.get("weights", "imagenet")
        self.include_top = self.cnn_cfg.get("include_top", False)
        self.pooling = self.cnn_cfg.get("pooling", "avg")
        self.target_size = tuple(config.get("video", {}).get("target_size", [224, 224]))
        self.model = self._build_model()

    def _build_model(self):
        if self.backbone_name == "MobileNetV2":
            base = MobileNetV2(weights=self.weights, include_top=self.include_top, pooling=self.pooling, input_shape=(self.target_size[0], self.target_size[1], 3))
            model = Model(inputs=base.input, outputs=base.output)
            return model
        else:
            # Fallback
            base = MobileNetV2(weights=self.weights, include_top=False, pooling="avg", input_shape=(224, 224, 3))
            model = Model(inputs=base.input, outputs=base.output)
            return model

    def extract_features(self, preprocessed_frames: np.ndarray) -> np.ndarray:
        if preprocessed_frames is None or len(preprocessed_frames) == 0:
            return np.array([])
        feats = self.model.predict(preprocessed_frames, verbose=0)
        return feats  # shape (T, D)

    def extract_and_cache(self, sample_id: str, preprocessed_frames: np.ndarray, cache_dir: str = "data/processed/features_cache") -> str:
        ensure_dirs([cache_dir])
        feats = self.extract_features(preprocessed_frames)
        out_path = os.path.join(cache_dir, f"{sample_id}.npy")
        meta_path = os.path.join(cache_dir, f"{sample_id}.json")
        np.save(out_path, feats)
        save_json({"sample_id": sample_id, "shape": list(feats.shape), "feature_dim": int(feats.shape[1]) if len(feats.shape) == 2 else None}, meta_path)
        return out_path
