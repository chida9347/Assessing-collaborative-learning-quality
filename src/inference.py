import os
from typing import Dict, Any

import numpy as np
from tensorflow.keras.models import load_model

from src.video_loader import VideoLoader
from src.preprocessing import Preprocessor
from src.feature_extraction import FeatureExtractor
from src.utils import load_config


class VideoAnalyzer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.loader = VideoLoader()
        self.preprocessor = Preprocessor(tuple(self.config.get("video", {}).get("target_size", [224, 224])))
        self.fe = FeatureExtractor(self.config)
        self.models = {}
        self._load_models()

    def _load_models(self):
        paths = self.config.get("paths", {})
        # Try to load available models
        for name, p in [("baseline", paths.get("cnn_baseline_model")), ("rnn", paths.get("rnn_model")), ("lstm", paths.get("lstm_model")), ("gru", paths.get("gru_model"))]:
            if p and os.path.exists(p):
                try:
                    self.models[name] = load_model(p)
                except Exception as e:
                    print(f"Failed to load {name} from {p}: {e}")

    def analyze(self, video_path: str, model_name: str = "lstm") -> Dict[str, Any]:
        if model_name not in self.models:
            return {"valid": False, "error": f"Model {model_name} not available or not loaded"}
        meta_res = self.loader.sample_frames(video_path, num_frames=self.config.get("dataset", {}).get("frame_sample_count", 16), target_size=tuple(self.config.get("video", {}).get("target_size", [224, 224])))
        if not meta_res.get("valid"):
            return {"valid": False, "error": meta_res.get("error", "Failed to load video")}
        frames = meta_res["frames"]
        preproc = self.preprocessor.preprocess_frames(frames)
        feats = self.fe.extract_features(preproc)  # (T, D)
        model = self.models[model_name]
        # Determine input shape
        inp_shape = model.input_shape
        if len(inp_shape) == 3:
            # temporal model expects (batch, T, D)
            x = np.expand_dims(feats, axis=0)
        elif len(inp_shape) == 4:
            # baseline might expect frames; but our baseline is trained on features? We'll handle
            x = np.expand_dims(preproc, axis=0) if False else np.expand_dims(feats, axis=0)  # default to features path if needed
        else:
            x = np.expand_dims(feats, axis=0)
        try:
            probs = model.predict(x, verbose=0)[0]
        except Exception as e:
            return {"valid": False, "error": str(e)}
        class_names = self.config.get("dataset", {}).get("class_names", ["Peer Discussion", "Shared Task Activity", "Presentation"])
        pred_idx = int(np.argmax(probs))
        return {
            "valid": True,
            "predicted_class": class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx),
            "predicted_idx": pred_idx,
            "probabilities": {class_names[i] if i < len(class_names) else str(i): float(probs[i]) for i in range(len(probs))},
            "metadata": meta_res["metadata"],
            "model_name": model_name,
            "note": "Prediction confidence is not the same as overall model accuracy.",
        }
