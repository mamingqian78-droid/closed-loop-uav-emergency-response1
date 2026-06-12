from __future__ import annotations

import numpy as np


def sample_confusion(true_class: int, confusion: np.ndarray, rng: np.random.Generator | None = None) -> int:
    rng = rng or np.random.default_rng()
    row = np.asarray(confusion[true_class], dtype=float)
    row = row / max(row.sum(), 1e-12)
    return int(rng.choice(len(row), p=row))
