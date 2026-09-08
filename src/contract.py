"""
THE FROZEN CONTRACT (PROJECT.md §4).

Agreed at hour 1, never changed. Every model returns `list[DiePrediction]`.
The explainer, the dashboard and the metrics module consume only this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Feature families used by the resolution-decomposed attribution (our signature output).
RESOLUTIONS = ("die", "spatial", "block")

# results.csv schema -- one row per run, appended, never rewritten.
RESULTS_COLUMNS = [
    "run_id", "timestamp", "wave", "model", "features_version",
    "f1", "auc_pr", "auc_roc", "accuracy", "precision", "recall",
    "threshold", "brier", "n_train", "n_test", "fail_rate", "notes",
]


@dataclass
class DiePrediction:
    """One die's prediction, explanation and uncertainty. The only inter-module currency."""

    wafer_id: str
    die_x: int
    die_y: int
    y_true: int                      # 0 pass, 1 fail (post-test)
    old_label: int                   # 1 = pre-test failure; EXCLUDED from train/eval
    p_fail: float                    # calibrated probability
    y_pred: int                      # after tuned threshold
    attribution: dict                # {"die": f, "spatial": f, "block": f}, sums to 1.0
    top_features: list               # [(feature_name, signed_shap_value), ...] top 5
    block_attention: np.ndarray | None = None   # (n_chunks,), Model B only
    conformal_lo: float | None = None
    conformal_hi: float | None = None
    root_cause: str | None = None    # from the spatial-signature lookup


def validate_predictions(preds: list[DiePrediction], *, require_block: bool = False) -> None:
    """Fail loudly if a model breaks the contract. Called at the end of every model run."""
    assert len(preds) > 0, "empty prediction list"

    for i, p in enumerate(preds[:1000]):  # spot-check the head; full scan is wasteful
        assert 0.0 <= p.p_fail <= 1.0, f"pred[{i}] p_fail out of range: {p.p_fail}"
        assert p.y_pred in (0, 1), f"pred[{i}] y_pred not binary: {p.y_pred}"
        assert p.old_label == 0, f"pred[{i}] old_label==1 leaked into predictions"
        assert set(p.attribution) == set(RESOLUTIONS), (
            f"pred[{i}] attribution keys {set(p.attribution)} != {set(RESOLUTIONS)}"
        )
        total = sum(p.attribution.values())
        assert abs(total - 1.0) < 1e-6, f"pred[{i}] attribution sums to {total}, not 1.0"
        assert len(p.top_features) <= 5, f"pred[{i}] top_features longer than 5"
        if require_block:
            assert p.block_attention is not None, f"pred[{i}] missing block_attention"
