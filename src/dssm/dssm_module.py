from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DSSM:
    grid_size: int = 32
    channels: int = 4

    def __post_init__(self) -> None:
        self.state = np.zeros((self.channels, self.grid_size, self.grid_size), dtype=np.float32)

    def update_cell(self, x: int, y: int, severity: float, confidence: float, age: float = 0.0) -> None:
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))
        self.state[0, x, y] = max(self.state[0, x, y], float(severity))
        self.state[1, x, y] = max(self.state[1, x, y], float(confidence))
        self.state[2, x, y] = float(age)
        self.state[3, x, y] = 1.0

    def decay(self, factor: float = 0.98) -> None:
        self.state[:3] *= factor

    def as_array(self) -> np.ndarray:
        return self.state.copy()
