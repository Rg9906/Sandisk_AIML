"""
Block-level features -- collapsing 2000 sub-die readings per die into a usable vector.

Recon Q2 established the readings are ORDERED (lag-1 autocorrelation ~0.25 against a
0.00 shuffled control), and the failure signal arrives as a *contiguous cluster* of
shifted blocks rather than scattered outliers. Order-aware statistics -- run lengths,
cluster centroid and spread -- are therefore meaningful, not noise. That is the whole
justification for the v2 feature set, and it is measured rather than assumed.

Blocks are read from a float16 memmap in row batches; the full array is never
materialised (this has to run on a 16 GB laptop with no GPU).

Versions:
    v1 (wave 1) -- mean, std, max. Exactly three. Deliberately trivial.
    v2 (wave 2) -- ~25 pooled statistics including the hot-cluster shape features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BATCH = 8192

V2_NAMES = [
    "bl_mean", "bl_std", "bl_median", "bl_skew", "bl_kurtosis",
    "bl_p10", "bl_p50", "bl_p90", "bl_p95", "bl_p99",
    "bl_min", "bl_max", "bl_range", "bl_iqr",
    "bl_hot_count", "bl_hot_frac", "bl_hot_longest_run",
    "bl_hot_centroid", "bl_hot_spread", "bl_argmax_pos",
    "bl_top10_mean", "bl_top50_mean", "bl_excess_mass",
    "bl_entropy", "bl_diff_absmean",
]


def _longest_true_run(H: np.ndarray) -> np.ndarray:
    """
    Longest run of consecutive True per row of a (n, k) boolean matrix.

    THE discriminative block feature: the kit clusters a failing die's anomalous blocks
    around a single seed index, so a failing die shows one long hot stretch while a
    passing die shows only short chance runs. Loops over the 2000 block positions with
    vectorised row operations -- never over dies.
    """
    n = H.shape[0]
    run = np.zeros(n, dtype=np.float32)
    best = np.zeros(n, dtype=np.float32)
    for j in range(H.shape[1]):
        run = np.where(H[:, j], run + 1.0, 0.0)
        np.maximum(best, run, out=best)
    return best


class BlockFeaturizer:
    """Pools the (n_dies, 2000) block matrix down to a handful of columns per die."""

    def __init__(self, version: str = "v1"):
        assert version in ("v1", "v2"), f"unknown block feature version {version!r}"
        self.version = version
        self.hot_threshold_: float | None = None
        self.baseline_mean_: float | None = None
        self.fitted_ = False

    def fit(self, split, y: np.ndarray | None = None) -> "BlockFeaturizer":
        """Learn the 'hot block' threshold from TRAIN ONLY (CLAUDE.md rule 7)."""
        if self.version == "v1":
            self.fitted_ = True
            return self

        # Sample rows rather than reading the whole memmap just to get two scalars.
        rng = np.random.default_rng(42)
        n = len(split.dies)
        idx = np.sort(rng.choice(n, size=min(4000, n), replace=False))
        X = np.asarray(split.blocks[idx], dtype=np.float32)

        self.baseline_mean_ = float(X.mean())
        # A block counts as "hot" if it sits in the top 2% of the train distribution.
        # Derived from train, applied unchanged to val/test.
        self.hot_threshold_ = float(np.percentile(X, 98.0))
        self.fitted_ = True
        return self

    def transform(self, split) -> pd.DataFrame:
        assert self.fitted_, "call fit() on the TRAIN split before transform()"
        assert split.blocks is not None, f"split '{split.name}' has no block readings"
        return self._v1(split.blocks) if self.version == "v1" else self._v2(split.blocks)

    def fit_transform(self, split, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(split, y).transform(split)

    # ------------------------------------------------------------------ v1
    @staticmethod
    def _v1(blocks: np.ndarray) -> pd.DataFrame:
        """Wave 1: mean, std, max. Streamed in batches so the memmap stays lazy."""
        n = len(blocks)
        out = np.empty((n, 3), dtype=np.float32)
        for i in range(0, n, BATCH):
            X = np.asarray(blocks[i:i + BATCH], dtype=np.float32)
            out[i:i + len(X), 0] = X.mean(axis=1)
            out[i:i + len(X), 1] = X.std(axis=1)
            out[i:i + len(X), 2] = X.max(axis=1)
        return pd.DataFrame(out, columns=["bl_mean", "bl_std", "bl_max"])

    # ------------------------------------------------------------------ v2
    def _v2(self, blocks: np.ndarray) -> pd.DataFrame:
        n, k = blocks.shape
        out = np.empty((n, len(V2_NAMES)), dtype=np.float32)
        thr = self.hot_threshold_
        pos = np.arange(k, dtype=np.float32)

        for i in range(0, n, BATCH):
            X = np.asarray(blocks[i:i + BATCH], dtype=np.float32)
            m = len(X)
            sl = slice(i, i + m)

            Xs = np.sort(X, axis=1)
            q = np.percentile(X, [10, 25, 50, 75, 90, 95, 99], axis=1).T

            hot = X > thr
            hot_n = hot.sum(axis=1).astype(np.float32)
            wsum = (hot * pos).sum(axis=1)
            centroid = np.where(hot_n > 0, wsum / np.maximum(hot_n, 1), k / 2.0)
            # Positional spread of the hot set: small = one tight cluster (a real defect),
            # large = scattered chance exceedances (a healthy die).
            var = np.where(
                hot_n > 1,
                (hot * (pos[None, :] - centroid[:, None]) ** 2).sum(axis=1)
                / np.maximum(hot_n - 1, 1),
                0.0,
            )

            # Probability-like normalisation for a distribution entropy.
            Xp = X - X.min(axis=1, keepdims=True) + 1e-6
            Xp /= Xp.sum(axis=1, keepdims=True)
            entropy = -(Xp * np.log(Xp + 1e-12)).sum(axis=1)

            cols = [
                X.mean(axis=1), X.std(axis=1), q[:, 2],
                stats.skew(X, axis=1), stats.kurtosis(X, axis=1),
                q[:, 0], q[:, 2], q[:, 4], q[:, 5], q[:, 6],
                X.min(axis=1), X.max(axis=1),
                X.max(axis=1) - X.min(axis=1), q[:, 3] - q[:, 1],
                hot_n, hot_n / k, _longest_true_run(hot),
                centroid, np.sqrt(var), np.argmax(X, axis=1).astype(np.float32),
                Xs[:, -10:].mean(axis=1), Xs[:, -50:].mean(axis=1),
                # Total signal above threshold: count x severity, the sparse-defect load.
                np.maximum(X - thr, 0).sum(axis=1),
                entropy,
                np.abs(np.diff(X, axis=1)).mean(axis=1),
            ]
            for j, c in enumerate(cols):
                out[sl, j] = c

        return pd.DataFrame(out, columns=V2_NAMES)


def feature_names(version: str = "v1") -> list[str]:
    return ["bl_mean", "bl_std", "bl_max"] if version == "v1" else list(V2_NAMES)
