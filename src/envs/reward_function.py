from __future__ import annotations


def weighted_reward(
    severity: float,
    latency_cost: float,
    energy_cost: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    coverage_reward = alpha * (2.6 * severity + 1.02) + 0.08
    latency_penalty = 0.55 * beta * latency_cost
    energy_penalty = 0.35 * gamma * energy_cost
    return float(coverage_reward - latency_penalty - energy_penalty)
