import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class Preprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        # frames assumed RGB uint8
        if frames is None or len(frames) == 0:
            return np.array([])
        x = frames.astype(np.float32)
        x = preprocess_input(x)
        return x
