from __future__ import annotations

import numpy as np


def generate_demand_points(seed: int, grid_size: int, n_points: int, min_manhattan_gap: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    while len(points) < n_points:
        point = rng.integers(1, grid_size - 1, size=2)
        if all(np.linalg.norm(point - q, ord=1) >= min_manhattan_gap for q in points):
            points.append(point)
    return np.vstack(points).astype(int)


def mark_served(served: np.ndarray, index: int) -> np.ndarray:
    updated = served.copy()
    updated[int(index)] = True
    return updated
