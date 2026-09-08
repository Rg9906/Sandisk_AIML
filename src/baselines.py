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


def dpat(split: Split, sigma: float = 3.0) -> np.ndarray:
    """
    Dynamic Part Average Testing: flag a die whose parametrics fall outside +/- k sigma
    of THAT WAFER's robust distribution (median +/- k * 1.4826 * MAD).

    DPAT is the fab's actual incumbent and it assumes each parameter is univariate
    Gaussian. Recon Q5 measured a median Cohen's d of 0.140 across the 500 parametrics,
    so we expect this to find very little -- which is exactly the published critique of
    DPAT at advanced nodes, reproduced on our own data rather than asserted from a paper.

    Score = the number of parameters outside the limits, which is what a fab would sum.
    """
    dies = split.dies.reset_index(drop=True)
    fcols = split.feature_cols
    scores = np.zeros(len(dies), dtype=float)

    for _, w in dies.groupby("wafer_id", sort=False, observed=True):
        F = w[fcols].to_numpy(dtype=np.float32)
        med = np.median(F, axis=0)
        mad = np.median(np.abs(F - med), axis=0) * 1.4826 + 1e-9
        outside = np.abs(F - med) > (sigma * mad)
        scores[w.index.to_numpy()] = outside.sum(axis=1)
    return scores


def nnr(split: Split, k: int = 8) -> np.ndarray:
    """
    Nearest Neighbour Residual: how far a die's parametrics sit from its k nearest
    spatial neighbours on the same wafer.

    The spatially-aware successor to DPAT -- it removes the wafer's process gradient
    before judging a die, so a die is compared against its own locality rather than the
    whole wafer. Our `die_anom_score` is the multivariate, learned generalisation of
    exactly this idea.
    """
    from scipy.spatial import cKDTree

    dies = split.dies.reset_index(drop=True)
    fcols = split.feature_cols
    scores = np.zeros(len(dies), dtype=float)

    for _, w in dies.groupby("wafer_id", sort=False, observed=True):
        xy = w[["die_row", "die_col"]].to_numpy(dtype=np.float32)
        F = w[fcols].to_numpy(dtype=np.float32)
        kk = min(k + 1, len(w))
        if kk < 3:
            continue
        _, nb = cKDTree(xy).query(xy, k=kk)
        # Column 0 is the die itself; compare it against the median of its neighbours.
        resid = F - np.median(F[nb[:, 1:]], axis=1)
        scale = np.median(np.abs(resid), axis=0) * 1.4826 + 1e-9
        scores[w.index.to_numpy()] = np.abs(resid / scale).mean(axis=1)
    return scores


ALL_BASELINES = {
    "all-pass": all_pass,
    "DPAT (3 sigma)": dpat,
    "NNR (k=8)": nnr,
    "GDBN (5x5)": gdbn,
}
