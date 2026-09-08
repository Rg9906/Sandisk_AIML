"""
Shared metrics, threshold tuning and run logging.

Hard rules enforced here (CLAUDE.md):
  - AUC-PR is the primary selection metric. Accuracy is REPORTED, never optimised.
  - The decision threshold is tuned on the validation PR curve, never left at 0.5.
  - Every run is appended to results/results.csv. An unlogged experiment did not happen.
"""
from __future__ import annotations

import csv
import datetime as _dt
import uuid
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from .contract import RESULTS_COLUMNS

RESULTS_PATH = Path("results/results.csv")


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    """
    Pick the threshold maximising F1 on the failure class, using the validation PR curve.

    Returns (threshold, best_f1). Never returns 0.5 by accident -- if the curve is
    degenerate we fall back to the validation fail rate as a quantile threshold.
    """
    if y_val.sum() == 0:
        return float(np.quantile(p_val, 0.97)), 0.0

    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    # precision/recall have one more element than thresholds.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision[:-1] + recall[:-1]) > 0,
            2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
            0.0,
        )
    if len(f1) == 0 or not np.isfinite(f1).any():
        return float(np.quantile(p_val, 1 - y_val.mean())), 0.0

    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])


def compute_metrics(y_true: np.ndarray, p_fail: np.ndarray, threshold: float) -> dict:
    """Full metric bundle for one model on one split. `threshold` must be pre-tuned."""
    y_true = np.asarray(y_true).astype(int)
    p_fail = np.asarray(p_fail, dtype=float)
    y_pred = (p_fail >= threshold).astype(int)

    has_both = 0 < y_true.sum() < len(y_true)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_true, p_fail) if has_both else float("nan"),
        "auc_roc": roc_auc_score(y_true, p_fail) if has_both else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, np.clip(p_fail, 0, 1)),
        "threshold": float(threshold),
        "fail_rate": float(y_true.mean()),
    }


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """The exact confusion layout the hackathon README asks for."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fail_acc = tp / (tp + fn) if (tp + fn) else 0.0
    pass_acc = tn / (tn + fp) if (tn + fp) else 0.0
    return (
        "```\n"
        f"{'':<20}{'Pred Fail':>11}{'Pred Pass':>11}   {'Metric':<16}{'Value':>10}\n"
        f"{'  Actual Fail':<20}{tp:>11}{fn:>11}   {'Fail Accuracy':<16}{fail_acc:>10.6f}\n"
        f"{'  Actual Pass':<20}{fp:>11}{tn:>11}   {'Pass Accuracy':<16}{pass_acc:>10.6f}\n"
        "```"
    )


def markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render metric rows as a markdown table -- pasteable straight into the deck."""
    if not rows:
        return "_(no rows)_"
    columns = columns or list(rows[0])

    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def log_run(*, wave: int, model: str, features_version: str, metrics: dict,
            n_train: int, n_test: int, notes: str = "") -> str:
    """Append one row to results/results.csv. Returns the run_id."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_PATH.exists()

    run_id = uuid.uuid4().hex[:8]
    row = {
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "wave": wave,
        "model": model,
        "features_version": features_version,
        "n_train": n_train,
        "n_test": n_test,
        "notes": notes,
        **{k: metrics.get(k, "") for k in
           ("f1", "auc_pr", "auc_roc", "accuracy", "precision",
            "recall", "threshold", "brier", "fail_rate")},
    }

    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)
    return run_id


def assert_not_trivial(metrics: dict, model: str) -> None:
    """
    Standing guard against the two failure modes the PS calls out by name:
    a model that predicts all-pass, and a threshold nobody tuned.
    """
    assert abs(metrics["threshold"] - 0.5) > 1e-9, (
        f"{model}: threshold is exactly 0.5 -- it was never tuned (CLAUDE.md rule 4)"
    )
    if metrics["recall"] == 0.0:
        print(f"  !! WARNING {model}: recall is 0.0 -- this is a trivial all-pass model")
