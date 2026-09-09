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


# ------------------------------------------------------- transductive wafer prior
WAFER_AGG_COLS = ["sp_wprior_mean", "sp_wprior_p90", "sp_wprior_max",
                  "sp_wprior_std", "sp_wprior_frac_hi", "sp_wprior_n",
                  "sp_die_vs_wafer_gap"]


def wafer_aggregates(dies: pd.DataFrame, scores: np.ndarray,
                     eligible: np.ndarray,
                     die_anom: np.ndarray | None = None) -> pd.DataFrame:
    """
    Summarise a wafer's stage-1 risk scores and broadcast them back to its dies.

    WHY THIS EXISTS. The generator draws each wafer's new-failure rate from an
    exponential distribution -- a per-wafer latent that no single die's features reveal,
    and which dominates the label variance. It looks irreducible.

    It is not, because prediction is transductive: at inference we hold the WHOLE wafer,
    not one die at a time. A wafer with a high latent rate has many dies scoring high, so
    the wafer's own score distribution estimates the latent that generated it. Stage 2
    then asks a different question of each die: "is this die risky *relative to the
    wafer it sits on*", which is the question a fab actually asks.

    Uses no labels and no cross-wafer statistics -- only test-time inputs. Aggregates are
    computed over ELIGIBLE dies only, since pre-test failures are not scored.
    """
    out = pd.DataFrame(np.zeros((len(dies), len(WAFER_AGG_COLS)), dtype=np.float32),
                       columns=WAFER_AGG_COLS)
    hi = float(np.quantile(scores[eligible], 0.9)) if eligible.any() else 0.0

    for _, w in dies.groupby("wafer_id", sort=False, observed=True):
        rows = w.index.to_numpy()
        el_rows = rows[eligible[rows]]
        s = scores[el_rows] if len(el_rows) else np.zeros(1, dtype=np.float32)
        out.iloc[rows, 0:6] = np.array([
            s.mean(), np.percentile(s, 90), s.max(), s.std(),
            float((s > hi).mean()), float(len(el_rows)),
        ], dtype=np.float32)

    if die_anom is not None:
        out["sp_die_vs_wafer_gap"] = (die_anom - out["sp_wprior_mean"].to_numpy()).astype(np.float32)
    else:
        out["sp_die_vs_wafer_gap"] = 0.0
    return out


class WaferLensModel:
    """LightGBM over die + spatial (+ optionally block) features."""

    def __init__(self, *, use_blocks: bool, wave: int = 1,
                 die_version: str = "v1", spatial_version: str = "v1",
                 block_version: str = "v1", params: dict | None = None,
                 two_stage: bool = False):
        self.use_blocks = use_blocks
        self.wave = wave
        self.die_version = die_version
        self.spatial_version = spatial_version
        self.block_version = block_version
        self.two_stage = two_stage
        self.name = "B" if use_blocks else "A"

        self.die_fx = DieFeaturizer(die_version)
        self.block_fx = BlockFeaturizer(block_version) if use_blocks else None
        self.clf: LGBMClassifier | None = None
        self.stage1_: LGBMClassifier | None = None
        self.stage1_columns_: list[str] = []
        self.threshold_: float = float("nan")
        self.columns_: list[str] = []
        self._train_key_: int | None = None
        self._train_X_: pd.DataFrame | None = None

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
        X_all = bundle.X

        if self.two_stage:
            X_all = self._fit_stage1(train, X_all, y_tr, el_tr)

        # Remember the EXACT matrix we trained on. The wafer-prior columns here are built
        # from cross-fitted stage-1 scores, whereas `design_matrix()` rebuilds them from
        # the full-fit stage-1 model. Those are different matrices, and anything that
        # re-trains on `design_matrix(train)` -- the tuner, the imbalance ladder -- would
        # otherwise be fitting on in-sample-sharpened features that no longer match what
        # the shipped model saw.
        self._train_key_ = id(train)
        self._train_X_ = X_all

        X_tr = X_all.loc[el_tr].reset_index(drop=True)
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

    # ----------------------------------------------------------------- stage 1
    def _fit_stage1(self, train: Split, X_all: pd.DataFrame,
                    y_tr: np.ndarray, el_tr: np.ndarray) -> pd.DataFrame:
        """
        Fit the stage-1 die scorer and append this wafer's transductive prior features.

        Train rows get CROSS-FITTED scores, folded BY WAFER. Using in-sample stage-1
        scores would make the wafer aggregate look far sharper on train than it can ever
        be at test time, and stage 2 would learn to trust it accordingly -- the classic
        stacking leak.
        """
        self.stage1_columns_ = list(X_all.columns)
        X_el = X_all.loc[el_tr].reset_index(drop=True)

        wafers = train.dies.loc[el_tr, "wafer_id"].to_numpy()
        uniq = np.unique(wafers)
        fold_of = {w: i % 5 for i, w in enumerate(uniq)}
        folds = np.array([fold_of[w] for w in wafers])

        oof = np.zeros(len(train.dies), dtype=np.float32)
        el_idx = np.flatnonzero(el_tr)
        for f in range(5):
            tr_m, te_m = folds != f, folds == f
            if te_m.sum() == 0 or len(np.unique(y_tr[tr_m])) < 2:
                continue
            m = LGBMClassifier(**self.params).fit(X_el[tr_m], y_tr[tr_m])
            oof[el_idx[te_m]] = m.predict_proba(X_el[te_m])[:, 1]

        # Full fit for inference on unseen wafers.
        self.stage1_ = LGBMClassifier(**self.params).fit(X_el, y_tr)

        die_anom = X_all["die_anom_score"].to_numpy() if "die_anom_score" in X_all.columns else None
        W = wafer_aggregates(train.dies.reset_index(drop=True), oof, el_tr, die_anom=die_anom)
        return pd.concat([X_all.reset_index(drop=True), W], axis=1)

    def _apply_stage1(self, split: Split, X_all: pd.DataFrame) -> pd.DataFrame:
        """Score a split with stage 1 and append its wafer-prior features."""
        p1 = self.stage1_.predict_proba(X_all[self.stage1_columns_])[:, 1]
        die_anom = X_all["die_anom_score"].to_numpy() if "die_anom_score" in X_all.columns else None
        W = wafer_aggregates(split.dies.reset_index(drop=True), p1, split.eligible, die_anom=die_anom)
        return pd.concat([X_all.reset_index(drop=True), W], axis=1)

    # ------------------------------------------------------------------ predict
    def predict_proba(self, split: Split) -> np.ndarray:
        """Failure probability for EVERY die in the split (eligible or not)."""
        assert self.clf is not None, "model is not fitted"
        X = self.build_features(split).X
        if self.two_stage:
            X = self._apply_stage1(split, X)
        return self.clf.predict_proba(X[self.columns_])[:, 1]

    def design_matrix(self, split: Split) -> pd.DataFrame:
        """
        The exact matrix the final model consumes -- used by the explainer and by any
        code that re-trains on these features.

        For the split this model was FITTED on, returns the stored training matrix
        (cross-fitted wafer prior) rather than rebuilding it. Rebuilding would produce
        in-sample stage-1 scores, so a re-trained model would learn to trust the wafer
        prior far more than it can be trusted at inference.
        """
        if self.two_stage and getattr(self, "_train_key_", None) == id(split):
            return self._train_X_[self.columns_]
        X = self.build_features(split).X
        if self.two_stage:
            X = self._apply_stage1(split, X)
        return X[self.columns_]

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

        X = self.design_matrix(split)
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
