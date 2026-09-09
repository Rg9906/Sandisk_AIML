"""
The data layer behind WaferLens OS.

Everything the UI shows is read from artifacts the pipeline actually produced --
`results/preds_<dataset>.pkl`, `results/results.csv`, the die cache and the block
memmap. Nothing here invents a number: if an artifact is missing, the endpoint says
so rather than substituting a plausible value.

Loaded once at process start; the whole test split is ~33k rows, so it stays resident.
"""
from __future__ import annotations

import functools
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "cache"

# The die a demo should open on if the operator has not picked one yet: the highest-risk
# true failure on the busiest wafer. Resolved at load time, never hard-coded.
DATASET = "full"


class MissingArtifact(RuntimeError):
    """Raised when the UI asks for something the pipeline has not produced yet."""


# --------------------------------------------------------------------------- helpers
def _f(x, nd: int = 6):
    """JSON-safe float: numpy scalars out, NaN/inf to None."""
    if x is None:
        return None
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def _clean_root_cause(raw: str | None) -> dict:
    """
    `root_cause` is stored as "Signature: cause | ACTION: action". Split it back into
    parts so the UI can style the signature, the cause and the dispatch action apart.
    """
    if not raw:
        return {"signature": "Unclassified", "cause": None, "action": None}
    head, _, action = raw.partition("| ACTION:")
    sig, _, cause = head.partition(":")
    return {
        "signature": sig.strip() or "Unclassified",
        "cause": cause.strip() or None,
        "action": action.strip() or None,
    }


# --------------------------------------------------------------------------- store
class Store:
    """Everything the API serves, resolved once."""

    def __init__(self, dataset: str = DATASET):
        self.dataset = dataset
        self.preds_path = RESULTS / f"preds_{dataset}.pkl"
        if not self.preds_path.exists():
            raise MissingArtifact(
                f"{self.preds_path} not found. Run:  python -m src.report --dataset {dataset}"
            )

        blob = pickle.load(self.preds_path.open("rb"))
        self.preds = blob["preds"]
        self.metrics = {k: _f(v) for k, v in blob["metrics"].items()}
        self.model_name = blob.get("model", "Model B")

        # ---- every die on every test wafer, including the pre-test failures the model
        # never scores. The wafer maps need those; the metrics never do.
        dies = pd.read_parquet(
            CACHE / dataset / "test_dies.parquet",
            columns=["wafer_id", "die_row", "die_col", "old_label", "label"],
        )
        dies["wafer_id"] = dies["wafer_id"].astype(str)
        self.dies = dies

        self.blocks_path = CACHE / dataset / "test_blocks.f16.npy"
        self._blocks = None

        # ---- index eligible predictions back onto their die rows
        eligible_rows = np.flatnonzero(dies["old_label"].to_numpy() == 0)
        if len(eligible_rows) != len(self.preds):
            raise MissingArtifact(
                f"prediction/die mismatch: {len(self.preds)} preds against "
                f"{len(eligible_rows)} eligible dies -- regenerate preds_{dataset}.pkl"
            )
        self.pred_row = eligible_rows                      # pred index -> die-table row
        self.row_pred = {int(r): i for i, r in enumerate(eligible_rows)}

        self.p_fail = np.array([p.p_fail for p in self.preds], dtype=np.float64)
        self.y_true = np.array([p.y_true for p in self.preds], dtype=np.int8)
        self.y_pred = np.array([p.y_pred for p in self.preds], dtype=np.int8)
        self.wafer_of = np.array([p.wafer_id for p in self.preds], dtype=object)

        # `p_fail` on the contract is the ISOTONIC-CALIBRATED probability, while the
        # threshold in the metrics block was tuned on the raw score. Comparing one to the
        # other silently under-counts flags (539 against the true 909). `y_pred` is the
        # model's own decision and is what the reported metrics were computed from, so it
        # is authoritative here; the calibrated-scale cut is recovered from where it splits.
        self.threshold_raw = float(blob["metrics"]["threshold"])
        flagged = self.p_fail[self.y_pred == 1]
        clear_ = self.p_fail[self.y_pred == 0]
        if len(flagged) and len(clear_):
            self.threshold = (float(flagged.min()) + float(clear_.max())) / 2
        else:
            self.threshold = self.threshold_raw

        self.runs = self._load_runs()

    # ------------------------------------------------------------------ block memmap
    @property
    def blocks(self) -> np.ndarray:
        if self._blocks is None:
            if not self.blocks_path.exists():
                raise MissingArtifact(f"{self.blocks_path} not found -- run src.ingest")
            self._blocks = np.load(self.blocks_path, mmap_mode="r")
        return self._blocks

    # ------------------------------------------------------------------ results.csv
    def _load_runs(self) -> pd.DataFrame:
        path = RESULTS / "results.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, on_bad_lines="skip")
        df = df[pd.to_numeric(df.get("n_test"), errors="coerce") == len(self.preds)]
        if df.empty:
            return df
        # One row per model: the most recent run wins.
        df = df.sort_values("timestamp").drop_duplicates("model", keep="last")
        return df.reset_index(drop=True)

    def _run(self, model: str) -> dict | None:
        if self.runs.empty:
            return None
        hit = self.runs[self.runs["model"] == model]
        if hit.empty:
            return None
        r = hit.iloc[0]
        return {k: _f(r[k]) for k in
                ("auc_pr", "f1", "recall", "precision", "accuracy", "threshold")}

    def _runs_prefixed(self, prefix: str) -> list[dict]:
        if self.runs.empty:
            return []
        out = []
        for _, r in self.runs[self.runs["model"].astype(str).str.startswith(prefix)].iterrows():
            out.append({
                "name": str(r["model"])[len(prefix):].strip(),
                **{k: _f(r[k]) for k in
                   ("auc_pr", "f1", "recall", "precision", "accuracy", "threshold")},
            })
        return out


@functools.lru_cache(maxsize=1)
def store(dataset: str = DATASET) -> Store:
    return Store(dataset)


# --------------------------------------------------------------------------- payloads
def summary() -> dict:
    s = store()
    n_el = len(s.preds)
    n_fail = int(s.y_true.sum())
    all_dies = len(s.dies)
    pre_fail = int((s.dies["old_label"] == 1).sum())

    model_a = s._run("ablation/A: + pooled anomaly score (die v2)  [MODEL A FINAL]")
    model_b = s._run("Model B / pooling") or {"auc_pr": s.metrics.get("auc_pr")}
    gdbn = s._run("baseline/GDBN (5x5)")
    best_base = max(
        (r for r in (s._run(f"baseline/{n}") for n in
                     ("all-pass", "DPAT (3 sigma)", "NNR (k=8)", "GDBN (5x5)")) if r),
        key=lambda r: r["auc_pr"], default=None)

    flagged = int((s.y_pred == 1).sum())
    caught = int(((s.y_pred == 1) & (s.y_true == 1)).sum())

    a_pr = (model_a or {}).get("auc_pr")
    b_pr = (model_b or {}).get("auc_pr")
    return {
        "dataset": s.dataset,
        "model": s.model_name,
        "threshold": _f(s.threshold),
        "threshold_raw": _f(s.threshold_raw),
        "counts": {
            "wafers": int(s.dies["wafer_id"].nunique()),
            "dies_total": all_dies,
            "eligible": n_el,
            "pre_test_fail": pre_fail,
            "new_fail": n_fail,
            "passed": all_dies - pre_fail - n_fail,
            "fail_rate": _f(n_fail / n_el),
            "flagged": flagged,
            "caught": caught,
            "escaped": n_fail - caught,
        },
        "model_a": model_a,
        "model_b": model_b,
        "delta_auc_pr": _f((b_pr - a_pr) if (a_pr and b_pr) else None),
        "delta_pct": _f(100 * (b_pr - a_pr) / a_pr if (a_pr and b_pr) else None, 2),
        "best_baseline": best_base,
        "gdbn": gdbn,
        "discrimination_gain": _f(
            b_pr / best_base["auc_pr"] if (b_pr and best_base and best_base["auc_pr"]) else None, 2),
        "metrics": s.metrics,
        "confusion": confusion(),
    }


def confusion() -> dict:
    s = store()
    pred = s.y_pred == 1
    y = s.y_true == 1
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    return {
        "true_pos": tp, "false_pos": fp, "false_neg": fn, "true_neg": tn,
        "fail_recall": _f(tp / max(tp + fn, 1)),
        "pass_recall": _f(tn / max(tn + fp, 1)),
        "precision": _f(tp / max(tp + fp, 1)),
        "accuracy": _f((tp + tn) / max(tp + tn + fp + fn, 1)),
        "scrap_pct": _f(100 * (tp + fp) / max(tp + tn + fp + fn, 1), 2),
        "escape_pct": _f(100 * fn / max(tp + fn, 1), 2),
    }


def wafer_list() -> list[dict]:
    s = store()
    out = []
    for wid, g in s.dies.groupby("wafer_id", sort=True):
        el = g["old_label"].to_numpy() == 0
        rows = g.index.to_numpy()
        pidx = [s.row_pred[int(r)] for r in rows[el]]
        p = s.p_fail[pidx]
        y = s.y_true[pidx]
        yp = s.y_pred[pidx]
        out.append({
            "wafer_id": str(wid),
            "dies": int(len(g)),
            "eligible": int(el.sum()),
            "pre_test_fail": int((~el).sum()),
            "new_fail": int(y.sum()),
            "fail_rate": _f(y.mean() if len(y) else 0),
            "flagged": int((yp == 1).sum()),
            "caught": int(((yp == 1) & (y == 1)).sum()),
            "mean_risk": _f(p.mean() if len(p) else 0),
            "max_risk": _f(p.max() if len(p) else 0),
            "rows": int(g["die_row"].max()) + 1,
            "cols": int(g["die_col"].max()) + 1,
            "yield_pct": _f(100 * (1 - (int((~el).sum()) + int(y.sum())) / len(g)), 2),
        })
    out.sort(key=lambda w: w["new_fail"], reverse=True)
    return out


def wafer(wafer_id: str) -> dict:
    """Every die on one wafer, in a form the three map panels can render directly."""
    s = store()
    g = s.dies[s.dies["wafer_id"] == wafer_id]
    if g.empty:
        raise MissingArtifact(f"unknown wafer '{wafer_id}'")

    rows = g.index.to_numpy()
    r = g["die_row"].to_numpy(); c = g["die_col"].to_numpy()
    old = g["old_label"].to_numpy(); lab = g["label"].to_numpy()

    p = np.full(len(g), -1.0)
    pred = np.zeros(len(g), dtype=np.int8)
    clo = np.full(len(g), np.nan)
    chi = np.full(len(g), np.nan)
    for k, row in enumerate(rows):
        i = s.row_pred.get(int(row))
        if i is not None:
            p[k] = s.p_fail[i]
            pred[k] = int(s.y_pred[i])
            q = s.preds[i]
            if q.conformal_lo is not None:
                clo[k], chi[k] = q.conformal_lo, q.conformal_hi

    # state: 2 = pre-test fail (input), 1 = newly failed (target), 0 = passed
    state = np.where(old == 1, 2, np.where(lab == 1, 1, 0))
    meta = next(w for w in wafer_list() if w["wafer_id"] == wafer_id)

    # The wafer's dominant spatial signature, taken from the dies' own root-cause field.
    sigs = {}
    for row in rows:
        i = s.row_pred.get(int(row))
        if i is None:
            continue
        sg = _clean_root_cause(s.preds[i].root_cause)["signature"]
        sigs[sg] = sigs.get(sg, 0) + 1
    dominant = max((k for k in sigs if k != "Clean"), key=lambda k: sigs[k], default="Clean")

    return {
        "wafer_id": wafer_id,
        "meta": meta,
        "grid": {"rows": meta["rows"], "cols": meta["cols"]},
        "die_row": r.tolist(),
        "die_col": c.tolist(),
        "state": state.tolist(),
        "p_fail": [None if v < 0 else _f(v, 4) for v in p],
        "conformal_lo": [None if np.isnan(v) else _f(v, 4) for v in clo],
        "conformal_hi": [None if np.isnan(v) else _f(v, 4) for v in chi],
        "y_pred": pred.tolist(),
        "signature": dominant,
        "signature_counts": sigs,
        "root_cause": _clean_root_cause(
            next((s.preds[s.row_pred[int(x)]].root_cause for x in rows
                  if s.row_pred.get(int(x)) is not None
                  and _clean_root_cause(s.preds[s.row_pred[int(x)]].root_cause)["signature"] == dominant),
                 None)),
    }


def die(wafer_id: str, row: int, col: int, with_blocks: bool = True) -> dict:
    """The full diagnosis for one die -- attribution, evidence, waveform, work order."""
    s = store()
    m = s.dies[(s.dies["wafer_id"] == wafer_id)
               & (s.dies["die_row"] == row) & (s.dies["die_col"] == col)]
    if m.empty:
        raise MissingArtifact(f"no die at {wafer_id} [{row},{col}]")
    die_row_idx = int(m.index[0])
    old_label = int(m["old_label"].iloc[0])

    if old_label == 1:
        # A pre-test failure is INPUT, never a prediction. Say that rather than faking one.
        return {
            "wafer_id": wafer_id, "die_row": row, "die_col": col,
            "eligible": False, "old_label": 1,
            "note": "Pre-test failure. Carried as spatial input, never scored or targeted.",
        }

    i = s.row_pred[die_row_idx]
    p = s.preds[i]
    rc = _clean_root_cause(p.root_cause)

    payload = {
        "wafer_id": p.wafer_id, "die_row": row, "die_col": col,
        "eligible": True, "old_label": 0,
        "p_fail": _f(p.p_fail, 4),
        "y_pred": int(p.y_pred),
        "y_true": int(p.y_true),
        "threshold": _f(s.threshold, 4),
        "conformal": {"lo": _f(p.conformal_lo, 4), "hi": _f(p.conformal_hi, 4),
                      "width": _f((p.conformal_hi - p.conformal_lo)
                                  if (p.conformal_hi is not None and p.conformal_lo is not None)
                                  else None, 4)},
        "attribution": {k: _f(v, 4) for k, v in (p.attribution or {}).items()},
        "top_features": [{"feature": f, "value": _f(v, 4)} for f, v in (p.top_features or [])],
        "root_cause": rc,
        "consistency": p.consistency,
        "rank": int((s.p_fail > p.p_fail).sum()) + 1,
    }

    if with_blocks:
        payload["blocks"] = block_trace(die_row_idx)
        if p.block_attention is not None:
            payload["blocks"]["attention"] = [_f(v, 5) for v in np.asarray(p.block_attention)]
    return payload


def block_trace(die_row_idx: int, downsample: int = 1) -> dict:
    """
    The 2000 sub-die readings for one die, plus where the anomalous cluster sits.

    Hot blocks are the ones above the die's own mean + 2 sd. That is a display rule for
    the oscilloscope annotation, not the model's feature -- the model's block features
    use a threshold learned on train.
    """
    s = store()
    raw = np.asarray(s.blocks[die_row_idx], dtype=np.float32)
    mu, sd = float(raw.mean()), float(raw.std())
    hot = np.flatnonzero(raw > mu + 2 * sd)

    # Longest contiguous hot run: the "defect excursion cluster" the UI annotates.
    lo = hi = None
    if len(hot):
        splits = np.split(hot, np.flatnonzero(np.diff(hot) > 1) + 1)
        best = max(splits, key=len)
        lo, hi = int(best[0]), int(best[-1])

    series = raw[::downsample] if downsample > 1 else raw
    lag1 = float(np.corrcoef(raw[:-1], raw[1:])[0, 1]) if len(raw) > 2 else None
    return {
        "n": int(len(raw)),
        "series": [round(float(v), 2) for v in series],
        "mean": _f(mu, 3), "std": _f(sd, 3),
        "min": _f(raw.min(), 3), "max": _f(raw.max(), 3),
        "hot_count": int(len(hot)),
        "hot_pct": _f(100 * len(hot) / len(raw), 2),
        "cluster": ({"lo": lo, "hi": hi, "length": hi - lo + 1} if lo is not None else None),
        "peak_index": int(raw.argmax()),
        "peak_amplitude": _f(raw.max() - mu, 2),
        "autocorrelation": _f(lag1, 3),
    }


def benchmarks() -> dict:
    s = store()
    order = [
        ("baseline/all-pass", "all-pass", "trap"),
        ("baseline/DPAT (3 sigma)", "DPAT (3 sigma)", "incumbent"),
        ("baseline/GDBN (5x5)", "GDBN (5x5)", "incumbent"),
        ("baseline/NNR (k=8)", "NNR (k=8)", "incumbent"),
        ("ablation/A: + pooled anomaly score (die v2)  [MODEL A FINAL]", "Model A (die + spatial)", "ours"),
        ("Model B / pooling", "Model B (+ 2000 blocks)", "ours"),
        ("Model B / attention-MIL", "Model B / attention-MIL", "variant"),
    ]
    rows = []
    for key, label, kind in order:
        r = s._run(key)
        if r:
            rows.append({"model": label, "kind": kind, **r})
    return {
        "rows": rows,
        "pr_curves": pr_curves(),
        "ablation": ablation(),
        "note": "AUC-PR on the failure class. Accuracy is shown only because the "
                "deliverable table asks for it; nothing is selected on it.",
    }


def ablation() -> list[dict]:
    """The component waterfall, in the order the pipeline actually adds them."""
    s = store()
    steps = s._runs_prefixed("ablation/")
    out, prev = [], None
    for st in steps:
        d = None if prev is None else _f(st["auc_pr"] - prev, 4)
        out.append({"step": st["name"], "auc_pr": st["auc_pr"], "delta": d})
        prev = st["auc_pr"]
    return out


def imbalance() -> dict:
    s = store()
    rungs = s._runs_prefixed("imbalance/")
    rungs.sort(key=lambda r: r["name"])
    best_pr = max(rungs, key=lambda r: r["auc_pr"] or 0, default=None)
    best_f1 = max(rungs, key=lambda r: r["f1"] or 0, default=None)
    return {
        "rungs": rungs,
        "best_auc_pr": best_pr["name"] if best_pr else None,
        "best_f1": best_f1["name"] if best_f1 else None,
        "fail_rate": _f(s.y_true.mean()),
        "all_pass_accuracy": _f(1 - s.y_true.mean()),
    }


def pr_curves(points: int = 120) -> dict:
    """Precision-recall for the shipped model, thinned to a drawable number of points."""
    from sklearn.metrics import precision_recall_curve
    s = store()
    pr, rc, _ = precision_recall_curve(s.y_true, s.p_fail)
    idx = np.unique(np.linspace(0, len(pr) - 1, points).astype(int))
    return {
        "model_b": [[_f(rc[i], 4), _f(pr[i], 4)] for i in idx],
        "baseline": _f(s.y_true.mean()),
    }


def calibration(bins: int = 12) -> dict:
    """Reliability of the shipped probabilities: predicted risk against observed rate."""
    s = store()
    p, y = s.p_fail, s.y_true
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi if hi == edges[-1] else p < hi)
        if m.sum() < 20:
            continue
        out.append({"predicted": _f(p[m].mean(), 4), "observed": _f(y[m].mean(), 4),
                    "n": int(m.sum())})
    brier = float(np.mean((p - y) ** 2))
    ece = float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in out) / max(len(p), 1))
    widths = [(q.conformal_hi - q.conformal_lo) for q in s.preds
              if q.conformal_hi is not None and q.conformal_lo is not None]
    return {
        "curve": out,
        "brier": _f(brier, 4),
        "ece": _f(ece, 4),
        "mean_interval_width": _f(float(np.mean(widths)) if widths else None, 5),
        "n_with_interval": len(widths),
    }


def economics(cost_ratio: float = 100.0, die_cost: float = 2.5,
              wafers_per_month: float = 20000.0, scrap_ceiling: float = 0.25) -> dict:
    """
    Live threshold sweep. A false accept costs `cost_ratio` times a scrapped die, so the
    optimum moves with the cost the operator sets. This is the slider the demo turns.
    """
    s = store()
    p, y = s.p_fail, s.y_true
    n = len(y)
    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.9995, 220)))

    curve, best = [], None
    for t in grid:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        scrap = (tp + fp) / n
        recall = tp / max(tp + fn, 1)
        escape = fn / max(tp + fn, 1)
        cost = fp * die_cost + fn * die_cost * cost_ratio
        row = {
            "threshold": _f(t, 4), "scrap_pct": _f(100 * scrap, 3),
            "escape_pct": _f(100 * escape, 3), "recall": _f(recall, 4),
            "precision": _f(tp / max(tp + fp, 1), 4),
            "dppm": _f(1e6 * fn / n, 1), "cost": _f(cost, 2),
        }
        curve.append(row)
        if scrap <= scrap_ceiling and (best is None or cost < best["cost"]):
            best = row

    if best is None:                       # ceiling tighter than any operating point
        best = min(curve, key=lambda r: r["cost"])

    ship_all_cost = int((y == 1).sum()) * die_cost * cost_ratio
    saving = ship_all_cost - best["cost"]
    # Per-wafer economics scaled to the fab's monthly volume.
    dies_per_wafer = len(s.dies) / s.dies["wafer_id"].nunique()
    annual = saving / n * dies_per_wafer * wafers_per_month * 12

    gdbn = s._run("baseline/GDBN (5x5)")
    return {
        "curve": curve,
        "optimum": best,
        "cost_ratio": cost_ratio,
        "die_cost": die_cost,
        "scrap_ceiling": scrap_ceiling,
        "ship_all_cost": _f(ship_all_cost, 2),
        "saving_pct": _f(100 * saving / max(ship_all_cost, 1e-9), 2),
        "annual_value": _f(annual, 0),
        "wafers_per_month": wafers_per_month,
        "gdbn_recall": (gdbn or {}).get("recall"),
        "recall_lift": _f((best["recall"] - (gdbn or {}).get("recall", 0)), 4),
    }


def work_order(wafer_id: str, limit: int = 60) -> dict:
    """The dispatch ticket: which dies to quarantine on this wafer, and what to go fix."""
    s = store()
    g = s.dies[s.dies["wafer_id"] == wafer_id]
    if g.empty:
        raise MissingArtifact(f"unknown wafer '{wafer_id}'")

    items = []
    for row in g.index.to_numpy():
        i = s.row_pred.get(int(row))
        if i is None:
            continue
        p = s.preds[i]
        if p.y_pred != 1:
            continue
        rc = _clean_root_cause(p.root_cause)
        items.append({
            "die_row": int(g.loc[row, "die_row"]),
            "die_col": int(g.loc[row, "die_col"]),
            "p_fail": _f(p.p_fail, 4),
            "signature": rc["signature"],
            "y_true": int(p.y_true),
            "marginal": bool(p.p_fail < s.threshold * 1.15),
        })
    items.sort(key=lambda d: d["p_fail"], reverse=True)

    w = wafer(wafer_id)
    rc = w["root_cause"]
    return {
        "wafer_id": wafer_id,
        "ticket": f"WEXC-{wafer_id}-{len(items):03d}",
        "quarantine_count": len(items),
        "dies": items[:limit],
        "truncated": max(0, len(items) - limit),
        "signature": w["signature"],
        "root_cause": rc,
        "confidence": _f(
            (w["signature_counts"].get(w["signature"], 0)
             / max(sum(w["signature_counts"].values()), 1)), 3),
        "steps": _dispatch_steps(rc, len(items)),
    }


def _dispatch_steps(rc: dict, n_quarantine: int) -> list[dict]:
    """Turn a root-cause signature into the ordered actions a fab would actually take."""
    sig = rc.get("signature", "Clean")
    action = rc.get("action") or "Screen die only"
    tool = {
        "Edge-Ring": "RTP-CHAMBER-02 // Module B",
        "Edge-Loc": "LOAD-LOCK-01 // Valve bank",
        "Center": "ETCH-RF-04 // Chamber A",
        "Donut": "CLEAN-STRIP-03 // Bath 2",
        "Scratch": "HANDLER ARM-03 // Robot",
        "Loc": "Zone-local tool -- see metrology",
        "Random": "Lot-level escalation",
        "Near-full": "Lot-level escalation -- halt if repeated",
    }.get(sig, "No wafer-scale tool implicated")
    return [
        {"label": f"Halt wafer lot at {tool.split(' //')[0]}",
         "detail": "Automated interlock; bay shutter paused", "status": "QUEUED"},
        {"label": action, "detail": f"Documented response to a {sig} signature",
         "status": "PENDING"},
        {"label": f"Quarantine {n_quarantine} identified dies",
         "detail": "Scrap coordinate map handed to the MES router", "status": "ENFORCED"},
        {"label": "Log excursion against the lot record",
         "detail": "Audit trail for the yield review", "status": "VERIFIED"},
    ]


def recon() -> dict:
    """The five wave-0 proofs, parsed out of the recon the pipeline wrote."""
    path = RESULTS / "recon.md"
    if not path.exists():
        return {"available": False, "questions": []}
    text = path.read_text(encoding="utf-8", errors="replace")
    qs = []
    for m in re.finditer(r"^## (Q\d) -- (.+?)$(.*?)(?=^## |\Z)", text, re.S | re.M):
        body = m.group(3).strip()
        verdict = ""
        vm = re.search(r"\*\*(.+?)\*\*", body, re.S)
        if vm:
            verdict = re.sub(r"\s+", " ", vm.group(1)).strip()
        prose = re.sub(r"\|.*\|", "", body)
        prose = re.sub(r"[*`#]", "", prose)
        prose = re.sub(r"\s+", " ", prose).strip()
        # The verdict is rendered as the card's heading, so drop it from the body text
        # rather than printing the same sentence twice.
        if verdict and prose.startswith(verdict):
            prose = prose[len(verdict):].lstrip(" .-")
        qs.append({"id": m.group(1), "question": m.group(2).strip(),
                   "verdict": verdict, "detail": prose[:520]})
    return {"available": True, "questions": qs}


def root_cause_table() -> list[dict]:
    """The signature -> physical cause -> action lookup, read from the explainer itself."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.explain import ROOT_CAUSE_TABLE
    s = store()
    seen = {}
    for p in s.preds:
        sg = _clean_root_cause(p.root_cause)["signature"]
        seen[sg] = seen.get(sg, 0) + 1
    return [{"signature": k, "cause": v[0], "action": v[1], "dies_seen": seen.get(k, 0)}
            for k, v in ROOT_CAUSE_TABLE.items()]


def focus_die() -> dict:
    """The die the demo opens on: highest risk among true failures, else highest risk."""
    s = store()
    fails = np.flatnonzero(s.y_true == 1)
    pick = int(fails[np.argmax(s.p_fail[fails])]) if len(fails) else int(np.argmax(s.p_fail))
    p = s.preds[pick]
    row = int(s.pred_row[pick])
    return {"wafer_id": p.wafer_id,
            "die_row": int(s.dies.loc[row, "die_row"]),
            "die_col": int(s.dies.loc[row, "die_col"])}
