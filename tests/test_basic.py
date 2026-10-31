import os
import numpy as np
from pathlib import Path

from src.video_loader import VideoLoader
from src.preprocessing import Preprocessor
from src.dataset import DatasetManager
from src.utils import load_config, set_seeds


def test_config_load():
    cfg = load_config("configs/config.yaml")
    assert cfg["project"]["title"] == "Video-Based Deep Learning for Assessing Collaborative Learning Quality"
    assert "dataset" in cfg


def test_dataset_discovery_empty():
    cfg = load_config("configs/config.yaml")
    dm = DatasetManager(cfg)
    samples = dm.discover_samples()
    assert isinstance(samples, list)


def test_preprocessor():
    p = Preprocessor()
    frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    out = p.preprocess_frames(frames)
    assert out.shape == (2, 224, 224, 3)


def test_video_loader_meta_missing():
    vl = VideoLoader()
    meta = vl.get_metadata("nonexistent_video.mp4")
    assert meta["valid"] is False


def test_seeds():
    set_seeds(123)
    a = np.random.rand(3)
    set_seeds(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)
