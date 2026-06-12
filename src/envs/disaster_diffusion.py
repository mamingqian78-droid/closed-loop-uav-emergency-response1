from __future__ import annotations

import numpy as np


def logistic_growth(state: np.ndarray, rate: float = 0.02, capacity: float = 1.0) -> np.ndarray:
    return np.clip(state + rate * state * (1.0 - state / capacity), 0.0, capacity)


def spatial_diffusion(state: np.ndarray, diffusion: float = 0.05) -> np.ndarray:
    padded = np.pad(state, 1, mode="edge")
    neighborhood = (
        padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) / 5.0
    return np.clip((1.0 - diffusion) * state + diffusion * neighborhood, 0.0, 1.0)


def update_disaster_field(state: np.ndarray, growth_rate: float = 0.02, diffusion: float = 0.05) -> np.ndarray:
    return spatial_diffusion(logistic_growth(state, growth_rate), diffusion)
