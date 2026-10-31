#!/usr/bin/env python3
import os
import argparse
import time

import numpy as np
from tensorflow.keras.utils import to_categorical

from src.utils import load_config, set_seeds, ensure_dirs, save_json
from src.dataset import DatasetManager
from src.feature_extraction import FeatureExtractor
from src.models import build_cnn_baseline_feature_model
from src.training import get_callbacks, compile_model
from src.evaluation import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seeds(config["project"]["seed"])

    ensure_dirs(["models/cnn", "results/metrics", "results/logs", "results/plots"])

    dm = DatasetManager(config)
    splits = dm.load_splits()
    if not splits or "splits" not in splits:
        print("No splits found. Run dataset discovery/split creation first.")
        print("You may need to prepare OUC-CGE data in data/raw/ following class subdirs or flat structure.")
        return

    fe = FeatureExtractor(config)
    class_names = config["dataset"]["class_names"]
    num_classes = len(class_names)

    # Load features for splits if cached
    cache_dir = config["paths"]["features_cache"]
    # For training, we'd need to extract; for now, check if features exist
    # This is a scaffold - will need actual feature loading
    print("Baseline training scaffold ready. Requires extracted features per split.")
    print(f"Splits: {splits.get('counts')}")

    # Try to build and save untrained model as scaffold if nothing else
    model = build_cnn_baseline_feature_model(feature_dim=config["cnn"]["feature_dim"], num_classes=num_classes)
    model = compile_model(model, config["training"]["learning_rate"])
    model.summary()
    out_path = config["paths"]["cnn_baseline_model"]
    # Don't save untrained as "trained" - but save architecture? Better to skip if no data
    print(f"Model architecture created. To train, provide features. Target save: {out_path}")


if __name__ == "__main__":
    main()
