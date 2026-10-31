import os
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from src.utils import save_json, ensure_dirs


class Evaluator:
    def __init__(self, class_names=None):
        self.class_names = class_names or ["Peer Discussion", "Shared Task Activity", "Presentation"]

    def evaluate(self, y_true, y_pred_probs) -> Dict[str, Any]:
        y_true = np.array(y_true)
        if y_pred_probs is None:
            raise ValueError("y_pred_probs is None")
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_true, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        prec_per, rec_per, f1_per, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        res = {
            "accuracy": float(acc),
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "macro_f1": float(f1_macro),
            "per_class": {
                self.class_names[i]: {
                    "precision": float(prec_per[i]) if i < len(prec_per) else 0.0,
                    "recall": float(rec_per[i]) if i < len(rec_per) else 0.0,
                    "f1": float(f1_per[i]) if i < len(f1_per) else 0.0,
                    "support": int(support[i]) if i < len(support) else 0,
                } for i in range(len(self.class_names))
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        return res

    def save(self, results: Dict[str, Any], path: str):
        ensure_dirs([os.path.dirname(path)])
        save_json(results, path)
