"""
Model A -- die parametrics + spatial context -> per-die failure probability.

Model B reuses this trainer wholesale and only adds block columns to the matrix, so the
A-vs-B delta measures the *data*, not two different pipelines. That is the whole point
of deliverable 3: any architectural difference between A and B would confound the
comparison the PS actually asks for.

Feature families are tracked by column prefix so the explainer can decompose attribution
by resolution without a second bookkeeping system:
    sp_*  -> spatial      bl_*  -> block      everything else -> die
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .contract import DiePrediction
from .features_block import BlockFeaturizer
from .features_die import DieFeaturizer
from .features_spatial import spatial_features
from .io_data import Split, assert_eligible_only
from .metrics import tune_threshold

SEED = 42


def family_of(col: str) -> str:
    """Map a column name to its resolution. The one source of truth for attribution."""
    if col.startswith("sp_"):
        return "spatial"
    if col.startswith("bl_"):
        return "block"
    return "die"


@dataclass
class FeatureBundle:
    """A design matrix plus the metadata every downstream consumer needs."""

    X: pd.DataFrame
    families: dict = field(default_factory=dict)   # column -> "die"|"spatial"|"block"

    def __post_init__(self):
        self.families = {c: family_of(c) for c in self.X.columns}


class WaferLensModel:
    """LightGBM over die + spatial (+ optionally block) features."""

    def __init__(self, *, use_blocks: bool, wave: int = 1,
                 die_version: str = "v1", spatial_version: str = "v1",
                 block_version: str = "v1", params: dict | None = None):
        self.use_blocks = use_blocks
        self.wave = wave
        self.die_version = die_version
        self.spatial_version = spatial_version
        self.block_version = block_version
        self.name = "B" if use_blocks else "A"

        self.die_fx = DieFeaturizer(die_version)
        self.block_fx = BlockFeaturizer(block_version) if use_blocks else None
        self.clf: LGBMClassifier | None = None
        self.threshold_: float = float("nan")
        self.columns_: list[str] = []

        self.params = params or dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            class_weight="balanced",   # imbalance ladder rung 1
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

    # ----------------------------------------------------------------- features
    def build_features(self, split: Split, *, fit: bool = False,
                       y: np.ndarray | None = None) -> FeatureBundle:
        """
        Assemble the design matrix for a split.

        Spatial features are built from ALL dies (pre-test failures are the neighbourhood
        signal); the eligibility filter is applied afterwards, by the caller.
        """
        parts = []
        if fit:
            self.die_fx.fit(split, y)
        parts.append(self.die_fx.transform(split))
        parts.append(spatial_features(split.dies, self.spatial_version))

        if self.use_blocks:
            if fit:
                self.block_fx.fit(split, y)
            parts.append(self.block_fx.transform(split))

        X = pd.concat(parts, axis=1)
        assert len(X) == len(split.dies), "feature/row misalignment"
        return FeatureBundle(X)

    # ---------------------------------------------------------------------- fit
    def fit(self, train: Split, val: Split) -> "WaferLensModel":
        """Fit on train's eligible dies; tune the threshold on val's PR curve."""
        el_tr = train.eligible
        y_tr = train.y()[el_tr]
        assert_eligible_only(y_tr, train.dies["old_label"].to_numpy()[el_tr], "model.fit/train")

        bundle = self.build_features(train, fit=True, y=y_tr)
        X_tr = bundle.X.loc[el_tr].reset_index(drop=True)
        self.columns_ = list(X_tr.columns)

        self.clf = LGBMClassifier(**self.params)
        self.clf.fit(X_tr, y_tr)

        # Threshold comes from validation only -- never from train, never left at 0.5.
        el_va = val.eligible
        y_va = val.y()[el_va]
        assert_eligible_only(y_va, val.dies["old_label"].to_numpy()[el_va], "model.fit/val")
        p_va = self.predict_proba(val)[el_va]
        self.threshold_, _ = tune_threshold(y_va, p_va)
        return self

    # ------------------------------------------------------------------ predict
    def predict_proba(self, split: Split) -> np.ndarray:
        """Failure probability for EVERY die in the split (eligible or not)."""
        assert self.clf is not None, "model is not fitted"
        X = self.build_features(split).X[self.columns_]
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, split: Split, explainer=None) -> list[DiePrediction]:
        """
        Contract-conforming predictions for the split's ELIGIBLE dies.

        `explainer` is optional: without it, attribution falls back to a global
        family-importance split so wave 1 still satisfies the contract.
        """
        from .explain import attribute  # local import avoids a cycle

        p_all = self.predict_proba(split)
        el = np.flatnonzero(split.eligible)
        dies = split.dies
        has_y = "label" in dies.columns

        X = self.build_features(split).X[self.columns_]
        attribs, tops = attribute(self, X, el, explainer=explainer)

        preds = []
        for k, i in enumerate(el):
            preds.append(DiePrediction(
                wafer_id=str(dies["wafer_id"].iloc[i]),
                die_x=int(dies["die_col"].iloc[i]),
                die_y=int(dies["die_row"].iloc[i]),
                y_true=int(dies["label"].iloc[i]) if has_y else -1,
                old_label=int(dies["old_label"].iloc[i]),
                p_fail=float(p_all[i]),
                y_pred=int(p_all[i] >= self.threshold_),
                attribution=attribs[k],
                top_features=tops[k],
            ))
        return preds


def build_model_a(wave: int = 1, **kw) -> WaferLensModel:
    """Model A: die features + spatial context. Deliverable 1."""
    return WaferLensModel(use_blocks=False, wave=wave, **kw)
