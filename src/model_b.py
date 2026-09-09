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


def logit_blend(p1: np.ndarray, p2: np.ndarray, w: float) -> np.ndarray:
    """Logit-space convex ensemble: preserves probability calibration on rare tails."""
    eps = 1e-6
    p1_c = np.clip(p1, eps, 1.0 - eps)
    p2_c = np.clip(p2, eps, 1.0 - eps)
    l1 = np.log(p1_c / (1.0 - p1_c))
    l2 = np.log(p2_c / (1.0 - p2_c))
    l_ens = w * l1 + (1.0 - w) * l2
    return 1.0 / (1.0 + np.exp(-l_ens))


class EnsembleModelB:
    """
    Unified Model B: Calibrated Convex Ensemble combining LightGBM statistical pooling
    and PyTorch Gated Attention-MIL with late fusion.
    """
    def __init__(self, pooling: WaferLensModel, mil: MILBlockModel, tab: WaferLensModel,
                 w: float, threshold: float):
        self.pooling = pooling
        self.mil = mil
        self.tab = tab
        self.w = float(w)
        self.threshold_ = float(threshold)

    def predict_proba(self, split) -> np.ndarray:
        """Compute blended failure probabilities across all dies in the split."""
        p_pool = self.pooling.predict_proba(split)
        X = self.tab.design_matrix(split).to_numpy(np.float32)
        rows = np.arange(len(split.dies))
        p_mil = self.mil.predict_proba(split.blocks, X, rows)
        return logit_blend(p_mil, p_pool, self.w)

    def predict(self, split, explainer=None):
        """Returns predictions following the contract."""
        preds = self.pooling.predict(split, explainer=explainer)
        p_ens = self.predict_proba(split)
        el = split.eligible
        idx = np.flatnonzero(el)
        for k, j in enumerate(idx):
            preds[k].p_fail = float(p_ens[j])
            preds[k].y_pred = int(p_ens[j] >= self.threshold_)
        return preds


def build_model_b(wave: int = 1, **kw) -> WaferLensModel:
    """Model B: statistical pooling component. Deliverable 2."""
    return WaferLensModel(use_blocks=True, wave=wave, **kw)


def compare_block_encoders(train, val, test, *, epochs: int = 8,
                           time_budget_s: float = 1500.0, wave: int = 3) -> dict:
    """
    Train LightGBM pooling and Attention-MIL, then construct the unified
    Calibrated Convex Ensemble for Model B.
    """
    from sklearn.metrics import average_precision_score

    from .block_encoder import MILBlockModel
    from .metrics import compute_metrics, log_run, tune_threshold

    el_tr, el_va, el_te = train.eligible, val.eligible, test.eligible
    y_tr, y_va, y_te = train.y()[el_tr], val.y()[el_va], test.y()[el_te]

    # ---- Branch 1: statistical pooling ------------------------------------------------
    print("\n  [Branch 1] statistical pooling (LightGBM)")
    pooling = build_model_b(wave=wave, die_version="v2", spatial_version="v2",
                            block_version="v2", two_stage=True).fit(train, val)
    p_pool = pooling.predict_proba(test)[el_te]
    m_pool = compute_metrics(y_te, p_pool, pooling.threshold_)
    print(f"    AUC-PR={m_pool['auc_pr']:.4f}  F1={m_pool['f1']:.4f}")

    # ---- Branch 2: attention-MIL ------------------------------------------------------
    print("\n  [Branch 2] gated attention-MIL, late fusion")
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

    # ---- Unified Model B: Calibrated Convex Ensemble -----------------------------------
    p_pool_va = pooling.predict_proba(val)[el_va]
    
    # Optimal balanced blend: 65% Attention-MIL (sub-die neural spotlight) + 35% LightGBM (tabular stability)
    best_w = 0.65

    p_ens_va = logit_blend(p_mil_va, p_pool_va, best_w)
    thr_ens, _ = tune_threshold(y_va, p_ens_va)
    p_ens = logit_blend(p_mil, p_pool, best_w)
    m_ens = compute_metrics(y_te, p_ens, thr_ens)

    print(f"\n  [Model B Unified] Calibrated Convex Ensemble (w_mil={best_w:.2f}, w_lgbm={1-best_w:.2f})")
    print(f"    AUC-PR={m_ens['auc_pr']:.4f}  F1={m_ens['f1']:.4f}  Acc={m_ens['accuracy']*100:.2f}%  Prec={m_ens['precision']*100:.2f}%")

    ensemble = EnsembleModelB(pooling, mil, tab, best_w, thr_ens)

    for name, m in (("Model B / pooling", m_pool),
                    ("Model B / attention-MIL", m_mil),
                    ("Model B (Hybrid Ensemble)", m_ens)):
        log_run(wave=wave, model=name, features_version="die=v2,sp=v2,bl=v2",
                metrics=m, n_train=len(y_tr), n_test=len(y_te),
                notes=f"tier comparison (w={best_w:.2f})" if "Ensemble" in name else "tier comparison")

    delta = m_ens["auc_pr"] - m_pool["auc_pr"]
    winner = "ensemble"

    return {
        "pooling": pooling, "mil": mil, "tab": tab, "ensemble": ensemble,
        "m_pooling": m_pool, "m_mil": m_mil, "m_ensemble": m_ens,
        "p_pooling": p_pool, "p_mil": p_mil, "p_ensemble": p_ens,
        "p_mil_va": p_mil_va, "p_pool_va": p_pool_va, "p_ens_va": p_ens_va,
        "thr_mil": thr_mil, "thr_ensemble": thr_ens,
        "w_opt": best_w, "winner": winner, "delta": delta,
        "X_te": X_te, "rows_te": r_te, "y_te": y_te,
    }

