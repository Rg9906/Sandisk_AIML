"""
WaferLens dashboard -- the live demo.

    streamlit run app/app.py

Deliberately built around ONE interaction: pick a wafer, click a die, get a diagnosis.
The judges' takeaway should be that the output is a work order, not a probability.

Predictions are cached to results/preds_<dataset>.pkl by `run_pipeline.py --report`, so
the app never trains anything and starts instantly.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain import ROOT_CAUSE_TABLE, classify_signature  # noqa: E402
from src.io_data import load_split                            # noqa: E402

st.set_page_config(page_title="WaferLens", page_icon="🔬", layout="wide")

CACHE = Path("results")


@st.cache_data(show_spinner=False)
def load_cached(dataset: str):
    """Load predictions written by the pipeline. Returns None if not yet generated."""
    path = CACHE / f"preds_{dataset}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


@st.cache_data(show_spinner=False)
def load_dies(dataset: str) -> pd.DataFrame:
    s = load_split(dataset, "test", with_blocks=False)
    return s.dies


def wafer_panels(dies: pd.DataFrame, pdf: pd.DataFrame, wafer_id: str):
    """Pre-test / predicted-risk / truth as three aligned grids."""
    w = dies[dies["wafer_id"] == wafer_id]
    shape = (int(w["die_row"].max()) + 1, int(w["die_col"].max()) + 1)

    pre = np.full(shape, np.nan)
    pre[w["die_row"], w["die_col"]] = 0.0
    old = w[w["old_label"] == 1]
    pre[old["die_row"], old["die_col"]] = 1.0

    risk = np.full(shape, np.nan)
    truth = pre.copy()
    if len(pdf):
        risk[pdf["die_y"], pdf["die_x"]] = pdf["p_fail"]
        nf = pdf[pdf["y_true"] == 1]
        truth[nf["die_y"], nf["die_x"]] = 0.5
    return pre, risk, truth


def main() -> None:
    st.title("🔬 WaferLens")
    st.caption("Multi-resolution die yield prediction with interpretable spatial context "
               "— every prediction decomposed by resolution and mapped to a process module.")

    dataset = st.sidebar.selectbox("Dataset", ["full", "smoke"], index=0)
    cached = load_cached(dataset)

    if cached is None:
        st.warning(
            f"No cached predictions for `{dataset}`. Generate them with:\n\n"
            f"```\npython run_pipeline.py --wave 2 --dataset {dataset} --all\n```"
        )
        st.stop()

    preds = cached["preds"]
    metrics = cached.get("metrics", {})
    dies = load_dies(dataset)

    pdf_all = pd.DataFrame([{
        "wafer_id": p.wafer_id, "die_x": p.die_x, "die_y": p.die_y,
        "p_fail": p.p_fail, "y_true": p.y_true, "y_pred": p.y_pred,
        "die": p.attribution["die"], "spatial": p.attribution["spatial"],
        "block": p.attribution["block"], "root_cause": p.root_cause,
        "idx": i,
    } for i, p in enumerate(preds)])

    # ---------------------------------------------------------------- header
    valid_cons = [p for p in preds if getattr(p, "consistency", None)
                  and p.consistency.get("status") in ("AGREE", "PARTIAL", "DISCREPANT")]
    if valid_cons:
        c = st.columns(6)
        agreed = sum(1.0 if p.consistency["status"] == "AGREE" else 0.5 for p in valid_cons)
        c[0].metric("AUC-PR", f"{metrics.get('auc_pr', float('nan')):.4f}")
        c[1].metric("F1 (fail)", f"{metrics.get('f1', float('nan')):.4f}")
        c[2].metric("Recall", f"{metrics.get('recall', float('nan')):.3f}")
        c[3].metric("Precision", f"{metrics.get('precision', float('nan')):.3f}")
        c[4].metric("Threshold", f"{metrics.get('threshold', float('nan')):.3f}",
                    help="Tuned on the validation PR curve — never left at 0.5.")
        c[5].metric("Concordance", f"{100 * agreed / len(valid_cons):.1f}%",
                    help="Cross-validation agreement between spatial signature and sub-die blocks.")
    else:
        c = st.columns(5)
        c[0].metric("AUC-PR", f"{metrics.get('auc_pr', float('nan')):.4f}")
        c[1].metric("F1 (fail)", f"{metrics.get('f1', float('nan')):.4f}")
        c[2].metric("Recall", f"{metrics.get('recall', float('nan')):.3f}")
        c[3].metric("Precision", f"{metrics.get('precision', float('nan')):.3f}")
        c[4].metric("Threshold", f"{metrics.get('threshold', float('nan')):.3f}",
                    help="Tuned on the validation PR curve — never left at 0.5.")

    # ---------------------------------------------------------------- wafer
    order = (pdf_all.groupby("wafer_id")["y_true"].sum()
             .sort_values(ascending=False).index.tolist())
    wafer_id = st.sidebar.selectbox("Wafer", order,
                                    help="Sorted by number of true new failures.")
    pdf = pdf_all[pdf_all["wafer_id"] == wafer_id].reset_index(drop=True)

    pre, risk, truth = wafer_panels(dies, pdf, wafer_id)

    st.subheader(f"Wafer {wafer_id}")
    cols = st.columns(3)
    for col, (arr, title, cmap) in zip(cols, [
        (pre, "Pre-test (input)", "RdYlGn_r"),
        (risk, "Predicted risk", "inferno"),
        (truth, "Post-test truth", "RdYlGn_r"),
    ]):
        with col:
            st.markdown(f"**{title}**")
            st.image(_to_image(arr, cmap), use_container_width=True)

    st.caption("Left: green = passing, red = already failed. Middle: brighter = higher "
               "predicted risk. Right: orange = the new failures we are trying to catch.")

    # ---------------------------------------------------------------- die
    st.divider()
    st.subheader("Per-die diagnosis")

    top = pdf.sort_values("p_fail", ascending=False).head(40)
    labels = [f"({r.die_y}, {r.die_x})  risk {r.p_fail:.3f}"
              + ("  [TRUE FAIL]" if r.y_true == 1 else "")
              for r in top.itertuples()]
    pick = st.selectbox("Highest-risk dies on this wafer", labels)
    row = top.iloc[labels.index(pick)]
    pred = preds[int(row["idx"])]

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### Die ({pred.die_y}, {pred.die_x})")
        st.markdown(f"**Failure risk: `{pred.p_fail:.3f}`**  "
                    f"(threshold {metrics.get('threshold', 0.5):.3f} → "
                    f"**{'SCREEN' if pred.y_pred else 'pass'}**)")
        if pred.y_true >= 0:
            st.markdown(f"Ground truth: **{'FAIL' if pred.y_true else 'pass'}**")

        st.markdown("#### Attribution by resolution")
        att = pd.DataFrame({
            "resolution": ["die", "spatial", "block"],
            "share": [pred.attribution[k] for k in ("die", "spatial", "block")],
        }).set_index("resolution")
        st.bar_chart(att, horizontal=True)

    with right:
        st.markdown("#### Root-cause hypothesis")
        if pred.root_cause:
            sig = pred.root_cause.split(":")[0]
            cause, action = ROOT_CAUSE_TABLE.get(sig, ("unknown", "investigate"))
            st.error(f"**Signature: {sig}**")
            st.markdown(f"**Documented cause:** {cause}")
            st.success(f"**ACTION:** {action}")

        cons = getattr(pred, "consistency", None)
        if cons and cons.get("status") != "UNAVAILABLE":
            st.markdown("#### Multimodal Verification (Spatial vs Block)")
            status = cons["status"]
            score = cons.get("score", 0.0)
            evidence = cons.get("evidence", "")
            if status == "AGREE":
                st.success(f"**Consistency [{status}] ({int(score * 100)}%)**: {evidence}")
            elif status == "PARTIAL":
                st.info(f"**Consistency [{status}] ({int(score * 100)}%)**: {evidence}")
            else:
                st.warning(f"**Consistency [{status}] ({int(score * 100)}%)**: {evidence}")

        if pred.top_features:
            st.markdown("#### Top contributing features")
            st.dataframe(
                pd.DataFrame(pred.top_features, columns=["feature", "SHAP value"]),
                hide_index=True, use_container_width=True,
            )

    with st.expander("Why this matters"):
        st.markdown(
            "Edge-Loc and Scratch both look like elongated clusters near the wafer edge, "
            "but they demand **opposite** interventions — thermal tuning versus mechanical "
            "handling. Confusing them sends the fab after the wrong module. Separating "
            "them, and naming the module, is what turns an interpretable model into an "
            "actionable one."
        )

    # ---------------------------------------------------------------- economics
    st.divider()
    st.subheader("Where should the threshold go?")
    st.caption("The operating point is a business decision, not a statistical one. "
               "A false accept becomes a field failure and an RMA; a false reject is pure "
               "yield loss. F1 implicitly assumes those cost the same — they never do.")

    scored = pdf_all[pdf_all["y_true"] >= 0]
    if len(scored):
        y = scored["y_true"].to_numpy()
        p = scored["p_fail"].to_numpy()

        ratio = st.slider(
            "Cost of one field failure, in units of one die",
            min_value=5, max_value=1000, value=100, step=5,
            help="Automotive and datacentre parts sit in the hundreds or thousands; "
                 "consumer parts far lower.",
        )
        cap = st.slider("Maximum acceptable yield loss (%)", 1, 50, 25, 1)

        from src.economics import optimal_threshold
        opt = optimal_threshold(y, p, float(ratio), float(cap))

        c = st.columns(4)
        c[0].metric("Optimal threshold", f"{opt['threshold']:.3f}")
        c[1].metric("Yield loss", f"{opt['yield_loss_pct']:.1f}%")
        c[2].metric("Failures caught", f"{opt['recall']:.1%}")
        c[3].metric("Escapes", f"{opt['escape_rate_pct']:.1f}%")

        st.caption(
            f"At a cost ratio of {ratio}:1, screening at {opt['threshold']:.3f} catches "
            f"{opt['recall']:.0%} of true failures for {opt['yield_loss_pct']:.1f}% yield "
            f"loss — **{opt['savings_pct']:.0f}% cheaper** than shipping everything and "
            f"absorbing the field failures."
        )


def _to_image(arr: np.ndarray, cmap: str) -> np.ndarray:
    """Render a NaN-aware grid to an RGB array without pulling in a plotting backend."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    finite = np.isfinite(arr)
    norm = mcolors.Normalize(
        vmin=np.nanmin(arr) if finite.any() else 0,
        vmax=np.nanmax(arr) if finite.any() and np.nanmax(arr) > 0 else 1,
    )
    # mpl.colormaps[...] -- cm.get_cmap was removed in matplotlib 3.9.
    rgba = mpl.colormaps[cmap](norm(np.nan_to_num(arr)))
    rgba[~finite] = [0.94, 0.94, 0.94, 1.0]        # empty grid cells stay grey
    img = (rgba[:, :, :3] * 255).astype(np.uint8)
    return np.kron(img, np.ones((8, 8, 1), dtype=np.uint8))   # nearest-neighbour zoom


if __name__ == "__main__":
    main()
