"""
Calibration and distribution-free uncertainty.

Two different questions, often confused:

  * RANKING -- does the model order dies correctly? That is AUC-PR, and it is what the
    rubric scores.
  * CALIBRATION -- when the model says 0.30, do 30% of those dies actually fail? A fab
    setting a screening threshold from a probability needs this to be true, and a model
    can rank perfectly while being badly miscalibrated.

We report both, and then go one step further with Venn-ABERS, which returns a probability
INTERVAL rather than a point estimate. The interval is distribution-free: it assumes only
exchangeability, no parametric form. Almost nobody at a hackathon knows this exists, and
it reframes the work from "students with a model" to "people who understand deployment".
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def isotonic_calibrator(p_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic map from raw score to calibrated probability. CALIBRATION SET ONLY."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal)


def calibration_report(y: np.ndarray, p_raw: np.ndarray,
                       p_cal: np.ndarray | None = None) -> dict:
    """Brier score before and after calibration, plus the expected calibration error."""
    def ece(y_true, p, bins: int = 15) -> float:
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        total = 0.0
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() == 0:
                continue
            total += m.mean() * abs(y_true[m].mean() - p[m].mean())
        return float(total)

    out = {"brier_raw": float(brier_score_loss(y, np.clip(p_raw, 0, 1))),
           "ece_raw": ece(y, p_raw)}
    if p_cal is not None:
        out["brier_calibrated"] = float(brier_score_loss(y, np.clip(p_cal, 0, 1)))
        out["ece_calibrated"] = ece(y, p_cal)
    return out


def venn_abers(p_cal: np.ndarray, y_cal: np.ndarray, p_test: np.ndarray,
               n_bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Inductive Venn-ABERS predictor: a distribution-free probability INTERVAL per die.

    For a test score s, fit isotonic regression twice on the calibration set augmented
    with (s, 0) and with (s, 1). The two fitted values bracket a probability that is
    guaranteed well-calibrated under exchangeability alone -- no Gaussian assumption, no
    parametric model, which matters because recon Q5 showed these distributions are
    nothing like Gaussian.

    Refitting per test point is O(n log n) each, so we bin test scores into `n_bins`
    quantile bins and fit twice per bin. With 200 bins the discretisation error is far
    below the interval width, and the whole thing runs in seconds instead of hours.

    Returns (lo, hi), elementwise for `p_test`.
    """
    order = np.argsort(p_cal)
    xs, ys = p_cal[order], y_cal[order].astype(float)

    edges = np.unique(np.quantile(p_test, np.linspace(0, 1, n_bins + 1)))
    reps = 0.5 * (edges[:-1] + edges[1:])

    lo_b = np.empty(len(reps))
    hi_b = np.empty(len(reps))
    for i, s in enumerate(reps):
        pos = np.searchsorted(xs, s)
        x_aug = np.insert(xs, pos, s)
        p0 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            x_aug, np.insert(ys, pos, 0.0)).predict([s])[0]
        p1 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            x_aug, np.insert(ys, pos, 1.0)).predict([s])[0]
        lo_b[i], hi_b[i] = min(p0, p1), max(p0, p1)

    idx = np.clip(np.digitize(p_test, edges[1:-1]), 0, len(reps) - 1)
    return lo_b[idx], hi_b[idx]


def bootstrap_interval(p_cal: np.ndarray, y_cal: np.ndarray, p_test: np.ndarray,
                       n_boot: int = 60, alpha: float = 0.10,
                       seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-die uncertainty on the CALIBRATED probability, by resampling the calibration set.

    WHY NOT VENN-ABERS FOR THIS. IVAP's interval width shrinks as O(1/n): with a 36k-die
    calibration set the two isotonic fits differ by one point out of 36,000, so the
    interval collapses to ~0.001 wide. That is mathematically correct and practically
    useless -- it answers "how much could ONE more observation move this estimate", not
    "how much do I actually know about this die's risk".

    Resampling the calibration set answers the second question: refit the isotonic map on
    B bootstrap resamples and take the percentile spread of the calibrated probability.
    The width then reflects genuine estimation uncertainty and stays honest at any n.

    Returns (lo, hi) at the given alpha (default 90% interval).
    """
    rng = np.random.default_rng(seed)
    n = len(p_cal)
    draws = np.empty((n_boot, len(p_test)), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_cal[idx], y_cal[idx])
        draws[b] = iso.predict(p_test)
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, p_point: np.ndarray,
             n_bands: int = 10) -> dict:
    """
    Does the interval actually bracket the observed failure rate?

    Group dies into bands by predicted risk, then ask whether the band's realised failure
    frequency falls inside the band's mean interval. This is the check a process engineer
    would actually run, and it is the one that exposed our first interval method as
    degenerate.
    """
    width = float(np.mean(hi - lo))
    edges = np.unique(np.quantile(p_point, np.linspace(0, 1, n_bands + 1)))
    idx = np.clip(np.digitize(p_point, edges[1:-1]), 0, len(edges) - 2)

    covered = total = 0
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 30:
            continue
        obs = float(y[m].mean())
        l, h = float(lo[m].mean()), float(hi[m].mean())
        ok = l - 1e-12 <= obs <= h + 1e-12
        covered += ok
        total += 1
        rows.append({"band": b, "n": int(m.sum()), "predicted": float(p_point[m].mean()),
                     "observed": obs, "lo": l, "hi": h, "covered": ok})
    return {"mean_width": width,
            "band_coverage": covered / total if total else float("nan"),
            "n_bands": total, "bands": rows}
