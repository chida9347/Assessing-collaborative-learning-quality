#!/usr/bin/env python3
import os
import json
from pathlib import Path

from src.utils import load_config, save_json


def main():
    config = load_config("configs/config.yaml")
    metrics_dir = Path("results/metrics")
    # Collect all metric jsons
    results = []
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                results.append({"file": f.name, **data})
            except Exception:
                continue
    out = {"models_compared": results, "note": "Only actual results from trained/evaluated models are reported."}
    save_json(out, "results/metrics/model_comparison.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
