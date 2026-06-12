from __future__ import annotations

import numpy as np


def apply_false_negative(active: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    return active & (rng.random(active.shape) > level)


def apply_false_positive(active: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    return active | (rng.random(active.shape) < level)


def apply_localization_noise(positions: np.ndarray, sigma: float, grid_size: int, rng: np.random.Generator) -> np.ndarray:
    return np.clip(positions + rng.normal(0, sigma, size=positions.shape), 0, grid_size - 1)
