import os
from pathlib import Path
from typing import Dict, Any

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def get_callbacks(config: Dict[str, Any], checkpoint_path: str):
    train_cfg = config.get("training", {})
    Path(os.path.dirname(checkpoint_path)).mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=train_cfg.get("early_stopping_patience", 10), restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=train_cfg.get("save_best_only", True), verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=train_cfg.get("reduce_lr_patience", 5), factor=0.5, verbose=1),
    ]
    return callbacks


def compile_model(model, lr: float = 1e-3):
    try:
        model.compile(optimizer=Adam(learning_rate=float(lr)), loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception:
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
