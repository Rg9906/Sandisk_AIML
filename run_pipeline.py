"""
WaferLens -- one command runs everything end to end.

    python run_pipeline.py --wave 1                     # everything at wave-1 quality
    python run_pipeline.py --wave 1 --dataset full      # ...on the 200-wafer set
    python run_pipeline.py --model a --wave 2
    python run_pipeline.py --all --report               # regenerate every table + figure

Every wave must end with this script completing, metrics in results/results.csv, and a
git tag. If a wave overruns, we ship what works and move on.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines, figures
from src.contract import validate_predictions
from src.explain import enrich_with_root_cause, explain_die, make_explainer
from src.io_data import load_all, load_split
from src.metrics import (assert_not_trivial, compute_metrics, confusion_table,
                         log_run, markdown_table, tune_threshold)
from src.model_a import build_model_a
from src.model_b import build_model_b

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """CLAUDE.md rule 9 -- judges may ask us to rerun."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def versions_for(wave: int) -> dict:
    """Feature-set versions per wave. Wave 1 is deliberately trivial."""
    if wave <= 1:
        return {"die_version": "v1", "spatial_version": "v1", "block_version": "v1",
                "two_stage": False}
    return {"die_version": "v2", "spatial_version": "v2", "block_version": "v2",
            "two_stage": True}


def run_baselines(train, val, test) -> tuple[list[dict], dict]:
    """Score the incumbent methods on the same eligible dies, same metrics."""
    rows, curves = [], {}
    el_te, y_te = test.eligible, test.y()[test.eligible]
    el_va, y_va = val.eligible, val.y()[val.eligible]

    for name, fn in baselines.ALL_BASELINES.items():
        p_va = fn(val)[el_va]
        p_te = fn(test)[el_te]
        # All-pass has no usable curve; give it a threshold above its constant score.
        thr = 0.5 if name == "all-pass" else tune_threshold(y_va, p_va)[0]
        if name == "all-pass":
            thr = 1.1
        m = compute_metrics(y_te, p_te, thr)
        rows.append({"model": name, **{k: m[k] for k in
                     ("f1", "auc_pr", "auc_roc", "accuracy", "precision", "recall")}})
        curves[name] = (y_te, p_te)
    return rows, curves


def run_model(kind: str, wave: int, train, val, test, *, explain: bool) -> dict:
    """Fit, evaluate, explain and log one model. Returns its metric row."""
    t0 = time.time()
    build = build_model_a if kind == "a" else build_model_b
    model = build(wave=wave, **versions_for(wave))
    print(f"\n--- Model {model.name} (wave {wave}) ---")

    model.fit(train, val)
    print(f"  fitted on {int(train.eligible.sum()):,} eligible dies, "
          f"{len(model.columns_)} features, threshold={model.threshold_:.4f}")

    el_te = test.eligible
    y_te = test.y()[el_te]
    p_te = model.predict_proba(test)[el_te]
    m = compute_metrics(y_te, p_te, model.threshold_)
    assert_not_trivial(m, f"Model {model.name}")

    explainer = make_explainer(model) if explain else None
    preds = model.predict(test, explainer=explainer)
    X_te = model.design_matrix(test)
    enrich_with_root_cause(preds, X_te, np.flatnonzero(el_te))
    validate_predictions(preds)

    run_id = log_run(
        wave=wave, model=f"Model {model.name}",
        features_version=f"die={model.die_version},sp={model.spatial_version}"
                         + (f",bl={model.block_version}" if model.use_blocks else ""),
        metrics=m, n_train=int(train.eligible.sum()), n_test=int(el_te.sum()),
        notes=f"lgbm balanced; {len(model.columns_)} feats; {time.time() - t0:.0f}s",
    )
    print(f"  AUC-PR={m['auc_pr']:.4f}  F1={m['f1']:.4f}  "
          f"recall={m['recall']:.3f}  precision={m['precision']:.3f}  (run {run_id})")

    return {"model": f"Model {model.name}",
            **{k: m[k] for k in ("f1", "auc_pr", "auc_roc", "accuracy",
                                 "precision", "recall")},
            "_obj": model, "_preds": preds, "_p": p_te, "_y": y_te, "_X": X_te}


def write_comparison(rows: list[dict], a: dict | None, b: dict | None,
                     test, wave: int, dataset: str) -> None:
    """Deliverable 3 -- the A vs B table, auto-generated."""
    Path("results").mkdir(exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    delta = ""
    if a and b:
        d_pr = b["auc_pr"] - a["auc_pr"]
        d_f1 = b["f1"] - a["f1"]
        pct = 100 * d_pr / a["auc_pr"] if a["auc_pr"] else 0.0
        verdict = ("**Block-level data HELPS.**" if d_pr > 0.005 else
                   "**Block-level data does NOT help materially.**" if d_pr > -0.005 else
                   "**Block-level data HURTS.**")
        delta = f"""
## Model A -> B delta (the question the PS actually asks)

| metric | Model A | Model B | delta |
|---|---|---|---|
| AUC-PR | {a['auc_pr']:.4f} | {b['auc_pr']:.4f} | **{d_pr:+.4f}** ({pct:+.1f}%) |
| F1 (fail) | {a['f1']:.4f} | {b['f1']:.4f} | **{d_f1:+.4f}** |
| recall (fail) | {a['recall']:.4f} | {b['recall']:.4f} | {b['recall'] - a['recall']:+.4f} |
| precision (fail) | {a['precision']:.4f} | {b['precision']:.4f} | {b['precision'] - a['precision']:+.4f} |

{verdict}

A and B use the identical estimator and hyper-parameters; B differs only by the block
feature columns. The delta therefore measures the *data*, not two modelling choices.
"""

    conf = ""
    if b or a:
        best = b or a
        conf = ("\n## Confusion matrix (best model, tuned threshold)\n\n"
                + confusion_table(best["_y"], (best["_p"] >= best["_obj"].threshold_).astype(int)))

    doc = f"""# Model Comparison -- deliverable 3

*Auto-generated by `run_pipeline.py` (wave {wave}, `{dataset}` dataset). Evaluated on the
held-out test wafers, ELIGIBLE dies only (`old_label == 0`).*

## Results

{markdown_table(clean, ['model', 'auc_pr', 'f1', 'recall', 'precision', 'auc_roc', 'accuracy'])}

> **Read the AUC-PR column, not accuracy.** The all-pass baseline row shows why:
> it scores ~96% accuracy while catching zero failures. The PS says so explicitly.
{delta}{conf}
"""
    Path("results/comparison.md").write_text(doc, encoding="utf-8")
    print("\n  wrote results/comparison.md")


def write_submission(model, dataset: str) -> Path:
    """
    The hackathon's required prediction file.

    Format per the kit README: one row per die INCLUDING pre-test failures, which are
    forced to 1 -- they trivially remain failed.
    """
    val = load_split(dataset, "validation")
    p = model.predict_proba(val)
    pred = (p >= model.threshold_).astype(int)
    pred[val.dies["old_label"].to_numpy() == 1] = 1

    out = pd.DataFrame({
        "wafer_id": val.dies["wafer_id"].astype(str),
        "die_row": val.dies["die_row"].astype(int),
        "die_col": val.dies["die_col"].astype(int),
        "predicted_label": pred.astype(int),
    })
    path = Path("results/submission.csv")
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out):,} rows, {int(pred.sum()):,} predicted fails)")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="WaferLens end-to-end pipeline")
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--dataset", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--model", default=None, choices=["a", "b"],
                    help="run only one model (default: both)")
    ap.add_argument("--explain", action="store_true", help="run SHAP + figures")
    ap.add_argument("--all", action="store_true", help="both models, explain, report")
    ap.add_argument("--report", action="store_true", help="regenerate tables and figures")
    args = ap.parse_args()

    if args.all:
        args.model, args.explain, args.report = None, True, True

    set_seeds()
    t0 = time.time()
    print(f"WaferLens | wave {args.wave} | dataset '{args.dataset}'")

    train, val, test = load_all(args.dataset)
    for s in (train, val, test):
        print(" ", s.describe())

    base_rows, curves = run_baselines(train, val, test)

    rows = list(base_rows)
    a = b = None
    if args.model in (None, "a"):
        a = run_model("a", args.wave, train, val, test, explain=args.explain)
        rows.append(a)
        curves["Model A"] = (a["_y"], a["_p"])
    if args.model in (None, "b"):
        b = run_model("b", args.wave, train, val, test, explain=args.explain)
        rows.append(b)
        curves["Model B"] = (b["_y"], b["_p"])

    write_comparison(rows, a, b, test, args.wave, args.dataset)

    best = b or a
    if best is not None:
        write_submission(best["_obj"], args.dataset)

        # Cache predictions so the dashboard starts instantly and trains nothing.
        import pickle
        cache = Path(f"results/preds_{args.dataset}.pkl")
        with cache.open("wb") as fh:
            pickle.dump({
                "preds": best["_preds"],
                "metrics": {k: best[k] for k in
                            ("auc_pr", "f1", "recall", "precision", "accuracy")}
                           | {"threshold": best["_obj"].threshold_},
                "model": best["model"], "wave": args.wave,
            }, fh)
        print(f"  wrote {cache} (dashboard cache)")

    if args.report and best is not None:
        preds = best["_preds"]
        # Pick the wafer with the most true new failures -- the most informative demo.
        counts = {}
        for p in preds:
            counts[p.wafer_id] = counts.get(p.wafer_id, 0) + p.y_true
        wid = max(counts, key=counts.get)

        figures.wafer_heatmap(preds, test.dies, wid)
        figures.pr_curves(curves)

        riskiest = max(preds, key=lambda p: p.p_fail)
        figures.attribution_bar(riskiest)
        if best["_obj"].clf is not None:
            ex = make_explainer(best["_obj"])
            row = int(np.argmax(best["_p"]))
            figures.shap_waterfall(ex, best["_X"].iloc[np.flatnonzero(test.eligible)], row)

        print("\n" + "=" * 68)
        print("HEADLINE OUTPUT -- highest-risk die on the held-out set")
        print("=" * 68)
        print(explain_die(riskiest))
        print("=" * 68)
        print(f"\n  figures written to figures/ (demo wafer: {wid})")

    print(f"\nPipeline complete in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
