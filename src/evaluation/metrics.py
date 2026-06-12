from __future__ import annotations

import numpy as np


def task_completion(served: np.ndarray) -> float:
    return float(np.asarray(served, dtype=bool).mean())


def response_latency(completion_times: list[float], horizon: float, n_targets: int) -> float:
    missing = n_targets - len(completion_times)
    return float(np.mean(list(completion_times) + [horizon] * missing))


def flight_distance(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(abs(x1 - x0) + abs(y1 - y0) for (x0, y0), (x1, y1) in zip(path[:-1], path[1:])))
