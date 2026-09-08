"""
The economics of the operating point.

A threshold is not a statistical choice, it is a business one. The two errors cost
wildly different amounts:

  * FALSE ACCEPT  -- a failing die ships, becomes a field failure, and costs an RMA plus
    reputational damage. In automotive or datacentre NAND this dominates.
  * FALSE REJECT  -- a good die is scrapped. That is pure yield loss, bounded by the cost
    of one die.

With a cost ratio R = (field failure cost) / (die cost), the expected cost of a screening
policy is directly computable, and the optimal threshold falls out. That turns "we picked
0.42 because it maximised F1" into "we picked 0.42 because at your cost ratio it minimises
expected cost, and here is the curve".

F1 implicitly assumes R = 1, which is never true in a fab.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sweep(y: np.ndarray, p: np.ndarray, n_points: int = 300) -> pd.DataFrame:
    """Yield loss, escapes and confusion counts across the full threshold range."""
    ths = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_points)))
    rows = []
    n = len(y)
    n_fail = int(y.sum())
    for t in ths:
        flag = p >= t
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        fn = n_fail - tp
        rows.append({
            "threshold": float(t),
            "screened": int(flag.sum()),
            "yield_loss_pct": 100.0 * flag.sum() / n,
            "good_dies_scrapped": fp,
            "escapes": fn,
            "escape_rate_pct": 100.0 * fn / max(n_fail, 1),
            "dppm_escaped": 1e6 * fn / n,
            "recall": tp / max(n_fail, 1),
            "precision": tp / max(tp + fp, 1),
        })
    return pd.DataFrame(rows)


def optimal_threshold(y: np.ndarray, p: np.ndarray, cost_ratio: float,
                      max_yield_loss_pct: float = 25.0) -> dict:
    """
    Threshold minimising expected cost, in units of one die.

        cost(t) = (good dies scrapped) * 1  +  (escaped failures) * cost_ratio

    `cost_ratio` is how many dies a single field failure is worth. Automotive quotes
    reach the thousands; consumer parts are far lower. It is the one number a customer
    actually has an opinion about, which is why the demo exposes it as a slider.

    `max_yield_loss_pct` caps how much of the wafer a policy may scrap. Without it, a
    weak ranker at a high cost ratio "optimally" scraps 100% of dies -- technically the
    cost minimum, but not a screening policy any fab would run, and it makes the
    comparison meaningless (every method that scraps everything achieves zero escapes).
    The cap is what forces methods to compete on RANKING rather than on caution.
    """
    df = sweep(y, p)
    df["expected_cost"] = df["good_dies_scrapped"] + cost_ratio * df["escapes"]
    feasible = df[df["yield_loss_pct"] <= max_yield_loss_pct]
    if feasible.empty:                      # cap tighter than the sparsest threshold
        feasible = df.nsmallest(1, "yield_loss_pct")
    best = feasible.loc[feasible["expected_cost"].idxmin()]

    # The do-nothing policy: ship everything, eat every field failure.
    baseline = cost_ratio * int(y.sum())
    return {
        "cost_ratio": cost_ratio,
        "threshold": float(best["threshold"]),
        "yield_loss_pct": float(best["yield_loss_pct"]),
        "escape_rate_pct": float(best["escape_rate_pct"]),
        "recall": float(best["recall"]),
        "precision": float(best["precision"]),
        "expected_cost": float(best["expected_cost"]),
        "cost_vs_ship_all": float(best["expected_cost"]) / max(baseline, 1e-9),
        "savings_pct": 100.0 * (1 - float(best["expected_cost"]) / max(baseline, 1e-9)),
    }


def cost_table(y: np.ndarray, p: np.ndarray,
               ratios: tuple = (10, 50, 100, 500, 1000),
               max_yield_loss_pct: float = 25.0) -> pd.DataFrame:
    """How the optimal operating point moves as the cost of an escape rises."""
    return pd.DataFrame([optimal_threshold(y, p, r, max_yield_loss_pct) for r in ratios])


def compare_to_gdbn(y: np.ndarray, p_model: np.ndarray, p_gdbn: np.ndarray,
                    cost_ratio: float = 100.0,
                    max_yield_loss_pct: float = 25.0) -> pd.DataFrame:
    """
    Head-to-head at each method's OWN cost-optimal threshold.

    This is the fair comparison: giving the incumbent its best possible operating point
    rather than a threshold chosen to flatter us.
    """
    rows = []
    for name, p in (("GDBN (incumbent)", p_gdbn), ("WaferLens", p_model)):
        r = optimal_threshold(y, p, cost_ratio, max_yield_loss_pct)
        rows.append({"method": name, **{k: r[k] for k in
                     ("threshold", "yield_loss_pct", "escape_rate_pct",
                      "recall", "precision", "expected_cost", "savings_pct")}})
    return pd.DataFrame(rows)
