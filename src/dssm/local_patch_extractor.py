from __future__ import annotations

import numpy as np


def extract_patch(tensor: np.ndarray, center: tuple[int, int], patch_size: int = 5) -> np.ndarray:
    radius = patch_size // 2
    padded = np.pad(tensor, ((0, 0), (radius, radius), (radius, radius)), mode="constant")
    x, y = center
    x += radius
    y += radius
    return padded[:, x - radius : x + radius + 1, y - radius : y + radius + 1]
