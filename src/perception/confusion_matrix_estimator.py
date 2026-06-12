from __future__ import annotations

import numpy as np


def normalize_confusion_matrix(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)


def empirical_fnr_fpr(confusion: np.ndarray, disaster_indices=(0, 1, 3), normal_index: int = 2) -> tuple[float, float]:
    cm = np.asarray(confusion, dtype=float)
    fnr = 1.0 - np.diag(cm)[list(disaster_indices)].mean()
    fpr = cm[normal_index, list(disaster_indices)].sum()
    return float(fnr), float(fpr)
