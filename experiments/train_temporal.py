#!/usr/bin/env python3
import os
import argparse

from src.utils import load_config, set_seeds, ensure_dirs
from src.models import build_temporal_rnn, build_temporal_lstm, build_temporal_gru
from src.training import compile_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["rnn", "lstm", "gru"], default="lstm")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seeds(config["project"]["seed"])
    ensure_dirs(["models/temporal", "results/metrics"])

    class_names = config["dataset"]["class_names"]
    num_classes = len(class_names)
    feat_dim = config["cnn"]["feature_dim"]
    seq_len = config["temporal"]["sequence_length"]

    if args.model == "rnn":
        model = build_temporal_rnn(feature_dim=feat_dim, sequence_length=seq_len, hidden_units=config["temporal"]["hidden_units_rnn"], num_classes=num_classes, dropout=config["temporal"]["dropout"])
        out = config["paths"]["rnn_model"]
    elif args.model == "gru":
        model = build_temporal_gru(feature_dim=feat_dim, sequence_length=seq_len, hidden_units=config["temporal"]["hidden_units_gru"], num_classes=num_classes, dropout=config["temporal"]["dropout"])
        out = config["paths"]["gru_model"]
    else:
        model = build_temporal_lstm(feature_dim=feat_dim, sequence_length=seq_len, hidden_units=config["temporal"]["hidden_units_lstm"], num_classes=num_classes, dropout=config["temporal"]["dropout"])
        out = config["paths"]["lstm_model"]

    model = compile_model(model, config["training"]["learning_rate"])
    model.summary()
    print(f"Temporal {args.model} scaffold ready. To train, provide feature sequences. Target save: {out}")


if __name__ == "__main__":
    main()
