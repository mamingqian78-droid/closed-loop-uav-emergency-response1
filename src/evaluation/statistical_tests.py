from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_signed_rank(a, b) -> float:
    return float(wilcoxon(np.asarray(a), np.asarray(b), zero_method="zsplit").pvalue)


def bonferroni(p_values, n_tests: int | None = None):
    n = n_tests or len(p_values)
    return [min(float(p) * n, 1.0) for p in p_values]


def rank_biserial(a, b) -> float:
    diff = np.asarray(a) - np.asarray(b)
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / max(len(diff), 1))
