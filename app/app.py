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
