"""
Die-level parametric features.

Fit/transform, not a bare function: several of these learn statistics (feature means,
the anomaly-score direction) and CLAUDE.md rule 7 says those are fit on TRAIN ONLY and
applied everywhere else. The class is what makes that impossible to get wrong.

Versions:
    v1 (wave 1) -- raw parametrics, nothing else. Deliberately trivial.
    v2 (wave 2) -- per-wafer spatial detrending + a pooled multivariate anomaly score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


class DieFeaturizer:
    """Turns the raw parametric matrix into model features. Stateful by design."""

    def __init__(self, version: str = "v1", n_raw_passthrough: int = 40):
        assert version in ("v1", "v2"), f"unknown die feature version {version!r}"
        self.version = version
        self.n_raw_passthrough = n_raw_passthrough
        self.feature_cols_: list[str] = []
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, split, y: np.ndarray | None = None) -> "DieFeaturizer":
        """
        Learn everything this featurizer needs. TRAIN SPLIT ONLY.

        `y` must already be restricted to eligible dies if supplied.
        """
        self.feature_cols_ = list(split.feature_cols)

        if self.version == "v1":
            self.fitted_ = True
            return self

        raise NotImplementedError("die features v2 lands in wave 2")

    # ------------------------------------------------------------ transform
    def transform(self, split) -> pd.DataFrame:
        assert self.fitted_, "call fit() on the TRAIN split before transform()"
        if self.version == "v1":
            # Wave 1: hand the raw parametrics straight through.
            return split.dies[self.feature_cols_].astype(np.float32).reset_index(drop=True)
        raise NotImplementedError("die features v2 lands in wave 2")

    def fit_transform(self, split, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(split, y).transform(split)
