from __future__ import annotations

import numpy as np


class GreedyPolicy:
    def act(self, severity: np.ndarray, positions: np.ndarray, current_position: np.ndarray, valid_actions: np.ndarray) -> int:
        candidates = np.flatnonzero(valid_actions)
        if len(candidates) == 0:
            return 0
        distances = np.abs(positions[candidates] - current_position).sum(axis=1)
        scores = severity[candidates] - 0.018 * distances
        return int(candidates[np.argmax(scores)])
