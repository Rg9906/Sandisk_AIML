"""
Die-level parametric features.

Fit/transform, not a bare function: several of these learn statistics (residual scales,
the anomaly-score direction) and CLAUDE.md rule 7 says those are fit on TRAIN ONLY and
applied everywhere else. The class is what makes that impossible to get wrong.

Versions:
    v1 (wave 1) -- raw parametrics, nothing else. Deliberately trivial.
    v2 (wave 2) -- per-wafer spatial detrending + a pooled multivariate anomaly score.

Why v2 looks like this (measured in recon, not assumed):

  * Q5 found a median Cohen's d of 0.140 across the 500 parametrics, with the best single
    feature reaching only 0.328. No tree will find a needle here, because there is no
    needle -- there are 500 pieces of straw. But they are near-independent, so they add in
    quadrature to an aggregate d of ~3.5. The win comes from POOLING them into one
    explicit multivariate score, not from feature selection.

  * The generator adds a random per-wafer radial and linear gradient to every feature.
    That nuisance variance is larger than the failure shift itself, so we regress it out
    per wafer, per feature, before scoring. This is NNR / spatial-trend detrending, the
    most-validated replacement for DPAT in the literature.

Collapsing 500 columns into ~35 also fixes an attribution artefact: with 500 die columns
against 19 spatial ones, the die family won resolution attribution on column COUNT alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SEED = 42
N_TOP_RAW = 30      # strongest individual parametrics, kept alongside the pooled score
N_FOLDS = 5         # for cross-fitted training scores


def _design_matrix(w: pd.DataFrame) -> np.ndarray:
    """[1, radial, x, y] for one wafer -- the nuisance trend the generator injects."""
    r = w["die_row"].to_numpy(dtype=np.float32)
    c = w["die_col"].to_numpy(dtype=np.float32)
    rows, cols = r.max() + 1, c.max() + 1
    yn = (r - rows / 2) / max(rows / 2, 1)
    xn = (c - cols / 2) / max(cols / 2, 1)
    rad = np.sqrt(xn ** 2 + yn ** 2)
    return np.column_stack([np.ones_like(xn), rad, xn, yn]).astype(np.float32)


def detrend_per_wafer(dies: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Regress every parametric on [1, radial, x, y] WITHIN each wafer and keep the residual.

    Purely within-wafer, so it is safe to apply to any split: it uses no labels and no
    cross-wafer statistics. Vectorised -- one small least-squares solve per wafer covers
    all 500 features at once, never a loop over dies.
    """
    out = np.empty((len(dies), len(feature_cols)), dtype=np.float32)
    for _, w in dies.groupby("wafer_id", sort=False, observed=True):
        D = _design_matrix(w)
        F = w[feature_cols].to_numpy(dtype=np.float32)
        beta, *_ = np.linalg.lstsq(D, F, rcond=None)
        out[w.index.to_numpy()] = F - D @ beta
    return out


class DieFeaturizer:
    """Turns the raw parametric matrix into model features. Stateful by design."""

    def __init__(self, version: str = "v1", n_top_raw: int = N_TOP_RAW):
        assert version in ("v1", "v2"), f"unknown die feature version {version!r}"
        self.version = version
        self.n_top_raw = n_top_raw
        self.feature_cols_: list[str] = []
        self.scale_: np.ndarray | None = None
        self.clf_: LogisticRegression | None = None
        self.top_idx_: np.ndarray | None = None
        self._train_oof_: dict = {}
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, split, y: np.ndarray | None = None) -> "DieFeaturizer":
        """Learn residual scales and the anomaly direction. TRAIN SPLIT ONLY."""
        self.feature_cols_ = list(split.feature_cols)
        if self.version == "v1":
            self.fitted_ = True
            return self

        assert y is not None, "die features v2 need labels (eligible dies only) to fit"
        dies = split.dies.reset_index(drop=True)
        R = detrend_per_wafer(dies, self.feature_cols_)

        el = split.eligible
        self.scale_ = R[el].std(axis=0) + 1e-6      # train-only scaling (rule 7)
        Z = R[el] / self.scale_

        self.clf_ = LogisticRegression(
            C=0.05, max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=SEED,
        ).fit(Z, y)

        coef = np.abs(self.clf_.coef_.ravel())
        self.top_idx_ = np.argsort(-coef)[:self.n_top_raw]

        # Cross-fitted scores for the TRAIN rows only. An in-sample anomaly score would
        # be optimistically sharp, and the downstream LightGBM would learn to over-trust
        # it -- the classic stacking leak. Folds are grouped BY WAFER, like every other
        # split in this project.
        wafers = dies.loc[el, "wafer_id"].to_numpy()
        uniq = np.unique(wafers)
        fold_of = {w: i % N_FOLDS for i, w in enumerate(uniq)}
        folds = np.array([fold_of[w] for w in wafers])

        oof = np.zeros(len(Z), dtype=np.float32)
        for f in range(N_FOLDS):
            tr, te = folds != f, folds == f
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                oof[te] = self.clf_.decision_function(Z[te])
                continue
            m = LogisticRegression(
                C=0.05, max_iter=2000, class_weight="balanced",
                solver="lbfgs", random_state=SEED,
            ).fit(Z[tr], y[tr])
            oof[te] = m.decision_function(Z[te])

        full = np.zeros(len(dies), dtype=np.float32)
        full[np.flatnonzero(el)] = oof
        # Non-eligible rows are never scored, but must hold a defined value.
        full[~el] = oof.mean()
        self._train_oof_ = {"key": id(split), "values": full}
        self.fitted_ = True
        return self

    # ------------------------------------------------------------ transform
    def transform(self, split) -> pd.DataFrame:
        assert self.fitted_, "call fit() on the TRAIN split before transform()"
        if self.version == "v1":
            # Wave 1: hand the raw parametrics straight through.
            return split.dies[self.feature_cols_].astype(np.float32).reset_index(drop=True)

        dies = split.dies.reset_index(drop=True)
        R = detrend_per_wafer(dies, self.feature_cols_)
        Z = R / self.scale_

        # Cross-fitted score for train; a clean full-fit score for every other split.
        if self._train_oof_.get("key") == id(split):
            score = self._train_oof_["values"]
        else:
            score = self.clf_.decision_function(Z).astype(np.float32)

        absz = np.abs(Z)
        out = {
            # THE feature: 500 weak signals pooled into one strong one.
            "die_anom_score": score,
            "die_absz_mean": absz.mean(axis=1),
            "die_absz_max": absz.max(axis=1),
            "die_absz_p99": np.percentile(absz, 99, axis=1),
            "die_n_gt2": (absz > 2).sum(axis=1).astype(np.float32),
            "die_n_gt3": (absz > 3).sum(axis=1).astype(np.float32),
            # Signed mean picks up a coherent whole-die drift; |mean| would hide it.
            "die_z_mean": Z.mean(axis=1),
            "die_z_std": Z.std(axis=1),
        }
        for rank, j in enumerate(self.top_idx_):
            out[f"die_top{rank:02d}_{self.feature_cols_[j]}"] = Z[:, j]

        return pd.DataFrame(out).astype(np.float32)

    def fit_transform(self, split, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(split, y).transform(split)
