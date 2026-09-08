"""
Block-level features -- collapsing 2000 sub-die readings per die into a usable vector.

Recon Q2 established the readings are ORDERED (lag-1 autocorrelation ~0.25 against a
0.00 shuffled control), and the kit injects failure signal as a *contiguous cluster* of
roughly 100 shifted blocks out of 2000. Order-aware statistics -- run lengths, cluster
centroid and spread -- are therefore meaningful, not noise.

Blocks are read from a float16 memmap in row batches; the full array is never
materialised (CLAUDE.md: this must fit on a laptop).

Versions:
    v1 (wave 1) -- mean, std, max. Exactly three. Deliberately trivial.
    v2 (wave 2) -- ~25 pooled statistics including the hot-cluster shape features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BATCH = 8192


class BlockFeaturizer:
    """Pools the (n_dies, 2000) block matrix down to a handful of columns per die."""

    def __init__(self, version: str = "v1"):
        assert version in ("v1", "v2"), f"unknown block feature version {version!r}"
        self.version = version
        self.hot_threshold_: float | None = None
        self.fitted_ = False

    def fit(self, split, y: np.ndarray | None = None) -> "BlockFeaturizer":
        """Learn the 'hot block' threshold from TRAIN ONLY (CLAUDE.md rule 7)."""
        if self.version == "v1":
            self.fitted_ = True
            return self
        raise NotImplementedError("block features v2 lands in wave 2")

    def transform(self, split) -> pd.DataFrame:
        assert self.fitted_, "call fit() on the TRAIN split before transform()"
        assert split.blocks is not None, f"split '{split.name}' has no block readings"

        if self.version == "v1":
            return self._v1(split.blocks)
        raise NotImplementedError("block features v2 lands in wave 2")

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


def feature_names(version: str = "v1") -> list[str]:
    if version == "v1":
        return ["bl_mean", "bl_std", "bl_max"]
    raise NotImplementedError("block features v2 lands in wave 2")
