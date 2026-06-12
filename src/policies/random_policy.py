from __future__ import annotations

import numpy as np


class RandomPolicy:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, valid_actions: np.ndarray) -> int:
        actions = np.flatnonzero(valid_actions)
        return int(self.rng.choice(actions)) if len(actions) else 0
