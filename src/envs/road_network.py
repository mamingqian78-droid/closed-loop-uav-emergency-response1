from __future__ import annotations

from collections import defaultdict


def grid_adjacency(grid_size: int = 32) -> dict[tuple[int, int], list[tuple[int, int]]]:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for x in range(grid_size):
        for y in range(grid_size):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    graph[(x, y)].append((nx, ny))
    return dict(graph)
