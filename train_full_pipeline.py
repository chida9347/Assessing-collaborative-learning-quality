#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path

from src.utils import load_config, set_seeds, ensure_dirs, save_json
from src.dataset import DatasetManager
from src.video_loader import VideoLoader
from src.preprocessing import Preprocessor
from src.feature_extraction import FeatureExtractor
from src.models import build_cnn_baseline_feature_model, build_temporal_rnn, build_temporal_lstm, build_temporal_gru
from src.training import compile_model
from src.evaluation import Evaluator


def extract_features_for_split(config, split_name, samples_split, fe, vl, pp):
    X_list = []
    y_list = []
    for s in samples_split:
        vid = s["path"]
        if not os.path.exists(vid):
            print(f"Missing: {vid}")
            continue
        res = vl.sample_frames(vid, num_frames=config["dataset"]["frame_sample_count"], 
                               target_size=tuple(config["video"]["target_size"]))
        if not res.get("valid"):
            print(f"Failed to load frames for {vid}: {res.get('error')}")
            continue
        frames = res["frames"]
        preproc = pp.preprocess_frames(frames)
        feats = fe.extract_features(preproc)
        X_list.append(feats)
        y_list.append(s["label_idx"])
    if len(X_list) == 0:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seeds(config["project"]["seed"])
    
    ensure_dirs(["models/cnn", "models/temporal", "results/metrics", "results/logs", "results/plots", "data/processed"])
    
    dm = DatasetManager(config)
    splits = dm.load_splits()
    if not splits or "splits" not in splits:
        print("No splits found.")
        return
    
    vl = VideoLoader()
    pp = Preprocessor(tuple(config["video"]["target_size"]))
    fe = FeatureExtractor(config)
    
    class_names = config["dataset"]["class_names"]
    num_classes = len(class_names)
    feat_dim = config["cnn"]["feature_dim"]
    seq_len = config["dataset"]["frame_sample_count"]
    
    results_all = {}
    
    print("Extracting features...")
    X_train, y_train = extract_features_for_split(config, "train", splits["splits"]["train"], fe, vl, pp)
    X_val, y_val = extract_features_for_split(config, "val", splits["splits"]["val"], fe, vl, pp)
    X_test, y_test = extract_features_for_split(config, "test", splits["splits"]["test"], fe, vl, pp)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes) if len(y_val) > 0 else None
    y_test_cat = to_categorical(y_test, num_classes=num_classes) if len(y_test) > 0 else None
    
    evalr = Evaluator(class_names)
    
    if len(X_train) > 0:
        X_train_flat = X_train.mean(axis=1)
        X_val_flat = X_val.mean(axis=1) if len(X_val) > 0 else None
        X_test_flat = X_test.mean(axis=1) if len(X_test) > 0 else None
        
        baseline = build_cnn_baseline_feature_model(feature_dim=feat_dim, num_classes=num_classes)
        baseline = compile_model(baseline, float(config["training"]["learning_rate"]))
        os.makedirs(os.path.dirname(config["paths"]["cnn_baseline_model"]), exist_ok=True)
        cbs = [
            EarlyStopping(monitor='val_loss', patience=config["training"]["early_stopping_patience"], restore_best_weights=True),
            ModelCheckpoint(config["paths"]["cnn_baseline_model"], monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=config["training"]["reduce_lr_patience"], factor=0.5, verbose=1),
        ]
        print("Training baseline...")
        baseline.fit(X_train_flat, y_train_cat, validation_data=(X_val_flat, y_val_cat) if X_val_flat is not None and y_val_cat is not None else None,
                    epochs=config["training"]["epochs"], batch_size=config["training"]["batch_size"], callbacks=cbs, verbose=1)
        if X_test_flat is not None and len(X_test_flat) > 0:
            probs = baseline.predict(X_test_flat, verbose=0)
            res = evalr.evaluate(y_test, probs)
            results_all["baseline"] = res
            evalr.save(res, os.path.join("results/metrics", "baseline_metrics.json"))
            print("Baseline:", res["accuracy"], res["macro_f1"])
    
    for model_type in ["rnn", "lstm", "gru"]:
        if model_type == "rnn":
            model = build_temporal_rnn(feature_dim=feat_dim, sequence_length=seq_len, hidden_units=config["temporal"]["hidden_units_rnn"], num_classes=num_classes, dropout=config["temporal"]["dropout"])
            out = config["paths"]["rnn_model"]
        elif model_type == "gru":
            model = build_temporal_gru(feature_dim=feat_dim, sequence_length=seq_len, hidden_units=config["temporal"]["hidden_units_gru"], num_classes=num_classes, dropout=config["temporal"]["dropout"])
            out = config["paths"]["gru_model"]
        else:
            model = build_temporal_lstm(feature_dim=feat_dim, sequence_length=seq_len, hidden_units=config["temporal"]["hidden_units_lstm"], num_classes=num_classes, dropout=config["temporal"]["dropout"])
            out = config["paths"]["lstm_model"]
        model = compile_model(model, float(config["training"]["learning_rate"]))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cbs = [
            EarlyStopping(monitor='val_loss', patience=config["training"]["early_stopping_patience"], restore_best_weights=True),
            ModelCheckpoint(out, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=config["training"]["reduce_lr_patience"], factor=0.5, verbose=1),
        ]
        print(f"Training {model_type}...")
        model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat) if len(X_val) > 0 else None,
                  epochs=config["training"]["epochs"], batch_size=config["training"]["batch_size"], callbacks=cbs, verbose=1)
        if len(X_test) > 0:
            probs = model.predict(X_test, verbose=0)
            res = evalr.evaluate(y_test, probs)
            results_all[model_type] = res
            evalr.save(res, os.path.join("results/metrics", f"{model_type}_metrics.json"))
            print(f"{model_type}:", res["accuracy"], res["macro_f1"])
    
    save_json(results_all, os.path.join("results/metrics", "all_results.json"))
    print("Done. Results saved.")


if __name__ == "__main__":
    main()
