"""
Incumbent methods, so "our model is good" is a comparison and not an assertion.

The fab's actual screening ladder is PAT -> DPAT -> NNR -> GDBN. Those methods assume
univariate Gaussian distributions, which is exactly what breaks at advanced nodes -- and
recon Q5 measured a median Cohen's d of 0.14 across 500 parametrics, so we expect the
univariate methods to find almost nothing. Showing that is the setup for Model A.

Wave 1 ships ALL-PASS only. DPAT / NNR / GDBN land in wave 2 (one hour, hard cap).
"""
from __future__ import annotations

import numpy as np

from .features_spatial import bad_neighbour_ratio, wafer_grid
from .io_data import Split


def all_pass(split: Split) -> np.ndarray:
    """
    The trivial model the PS warns about: predict every die passes.

    Scores ~96% accuracy and 0.000 fail-F1. We report it in every run as the standing
    proof that accuracy is the wrong metric for this problem.
    """
    return np.zeros(len(split.dies), dtype=float)


def gdbn(split: Split, window: int = 5) -> np.ndarray:
    """
    Classic Good Die in a Bad Neighborhood: score a die by its neighbours' failure rate.

    Twenty years old, still the incumbent, and a single scalar. This is the bar Model A
    has to clear -- and note it uses no parametric information whatsoever.
    """
    scores = np.zeros(len(split.dies), dtype=float)
    dies = split.dies.reset_index(drop=True)
    for _, w in dies.groupby("wafer_id", sort=False, observed=True):
        grid, r, c = wafer_grid(w)
        scores[w.index.to_numpy()] = bad_neighbour_ratio(grid, window)[r, c]
    return scores
