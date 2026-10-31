#!/usr/bin/env python3
import json
from src.utils import load_config, save_json, ensure_dirs


def main():
    config = load_config("configs/config.yaml")
    ensure_dirs(["results/metrics"])
    # Ablation scaffold - not executed until data/training done
    ablation = {
        "experiment": "frame_count_ablation",
        "status": "not_executed",
        "description": "Compare performance with different numbers of sampled frames (e.g., 4, 8, 16) keeping split and protocol constant.",
        "planned_settings": [4, 8, 16],
        "note": "Will only report results if actually executed with real data."
    }
    save_json(ablation, "results/metrics/ablation_frame_count.json")
    print(json.dumps(ablation, indent=2))


if __name__ == "__main__":
    main()
