"""
Model B -- die + spatial + block-level readings -> per-die failure probability.

Two tiers, and we ship whichever wins on measured AUC-PR:

  Tier 1 (wave 2) -- statistical pooling. The SAME LightGBM as Model A with 25 pooled
      block columns appended. Deliberately identical estimator and hyper-parameters, so
      the A->B delta measures the DATA rather than two different modelling choices. That
      is what deliverable 3 actually asks.

  Tier 2 (wave 3) -- gated attention-MIL with late fusion (`block_encoder.py`). Treats
      each die as a bag of block instances; attention weights double as the block
      interpretability artifact.

The PS asks whether the block signal improves prediction "or lack thereof". So does the
MIL-vs-pooling question, and we answer it the same way: measure, then report the winner
honestly. "We tested a learned encoder against statistical pooling and pooling won" is a
real finding, not a failure -- and the fallback ladder is documented rather than improvised.
"""
from __future__ import annotations

import numpy as np

from .model_a import WaferLensModel


def build_model_b(wave: int = 1, **kw) -> WaferLensModel:
    """Model B Tier 1: everything Model A sees, plus pooled block features. Deliverable 2."""
    return WaferLensModel(use_blocks=True, wave=wave, **kw)


def compare_block_encoders(train, val, test, *, epochs: int = 8,
                           time_budget_s: float = 1500.0, wave: int = 3) -> dict:
    """
    Tier 1 vs Tier 2 head-to-head on identical splits, features and eligible dies.

    Returns both models, their metrics, and the verdict. The caller decides what to ship;
    this function only measures.
    """
    from sklearn.metrics import average_precision_score

    from .block_encoder import MILBlockModel
    from .metrics import compute_metrics, log_run, tune_threshold

    el_tr, el_va, el_te = train.eligible, val.eligible, test.eligible
    y_tr, y_va, y_te = train.y()[el_tr], val.y()[el_va], test.y()[el_te]

    # ---- Tier 1: statistical pooling ------------------------------------------------
    print("\n  [Tier 1] statistical pooling (LightGBM)")
    pooling = build_model_b(wave=wave, die_version="v2", spatial_version="v2",
                            block_version="v2", two_stage=True).fit(train, val)
    p_pool = pooling.predict_proba(test)[el_te]
    m_pool = compute_metrics(y_te, p_pool, pooling.threshold_)
    print(f"    AUC-PR={m_pool['auc_pr']:.4f}  F1={m_pool['f1']:.4f}")

    # ---- Tier 2: attention-MIL ------------------------------------------------------
    # The MIL tabular branch gets Model A's matrix (die + spatial + wafer prior) so the
    # ONLY difference from Tier 1 is how the 2000 block readings are consumed.
    print("\n  [Tier 2] gated attention-MIL, late fusion")
    tab = WaferLensModel(use_blocks=False, wave=wave, die_version="v2",
                         spatial_version="v2", two_stage=True).fit(train, val)

    def prep(split, mask):
        X = tab.design_matrix(split).to_numpy(np.float32)
        return X[mask], np.flatnonzero(mask)

    X_tr, r_tr = prep(train, el_tr)
    X_va, r_va = prep(val, el_va)
    X_te, r_te = prep(test, el_te)

    mil = MILBlockModel(epochs=epochs, time_budget_s=time_budget_s).fit(
        train.blocks, X_tr, y_tr, r_tr, val=(val.blocks, X_va, y_va, r_va))

    p_mil_va = mil.predict_proba(val.blocks, X_va, r_va)
    thr_mil, _ = tune_threshold(y_va, p_mil_va)
    p_mil = mil.predict_proba(test.blocks, X_te, r_te)
    m_mil = compute_metrics(y_te, p_mil, thr_mil)
    print(f"    AUC-PR={m_mil['auc_pr']:.4f}  F1={m_mil['f1']:.4f}")

    for name, m in (("Model B / pooling", m_pool), ("Model B / attention-MIL", m_mil)):
        log_run(wave=wave, model=name, features_version="die=v2,sp=v2,bl=v2",
                metrics=m, n_train=len(y_tr), n_test=len(y_te),
                notes="tier comparison")

    delta = m_mil["auc_pr"] - m_pool["auc_pr"]
    winner = "mil" if delta > 0 else "pooling"
    print(f"\n  VERDICT: {'attention-MIL' if winner == 'mil' else 'statistical pooling'} "
          f"wins by {abs(delta):.4f} AUC-PR")

    return {
        "pooling": pooling, "mil": mil, "tab": tab,
        "m_pooling": m_pool, "m_mil": m_mil,
        "p_pooling": p_pool, "p_mil": p_mil,
        "thr_mil": thr_mil, "delta": delta, "winner": winner,
        "X_te": X_te, "rows_te": r_te, "y_te": y_te,
    }
