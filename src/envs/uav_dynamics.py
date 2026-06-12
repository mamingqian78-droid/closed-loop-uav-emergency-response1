from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class UAVState:
    x: int
    y: int
    battery: float
    distance: float = 0.0


class UAVDynamics:
    """Grid UAV helper with four moves, hover, and return-to-base actions."""

    ACTIONS = {
        0: np.array([0, 1]),
        1: np.array([0, -1]),
        2: np.array([-1, 0]),
        3: np.array([1, 0]),
        4: np.array([0, 0]),
        5: "return_to_base",
    }

    def __init__(self, grid_size: int = 32, base: tuple[int, int] | None = None):
        self.grid_size = int(grid_size)
        self.base = np.array(base if base is not None else (grid_size // 2, grid_size // 2), dtype=int)

    def step(self, state: UAVState, action: int) -> UAVState:
        pos = np.array([state.x, state.y], dtype=int)
        move = self.ACTIONS[int(action)]
        if isinstance(move, str):
            new_pos = self.base.copy()
        else:
            new_pos = np.clip(pos + move, 0, self.grid_size - 1)
        travel = float(np.abs(new_pos - pos).sum())
        return UAVState(int(new_pos[0]), int(new_pos[1]), state.battery - travel, state.distance + travel)
