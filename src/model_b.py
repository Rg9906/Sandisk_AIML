"""
Model B -- die + spatial + block-level readings -> per-die failure probability.

Deliberately the SAME estimator and the SAME hyper-parameters as Model A, with block
columns appended to the matrix. Deliverable 3 asks whether the block signal improves
prediction; if A and B differed architecturally, the delta would measure our modelling
choices instead of the data, and the question would go unanswered.

Wave 2 adds statistical pooling (Tier 1). Wave 3 swaps the block branch for a gated
attention-MIL encoder with late fusion (Tier 2) -- still compared against this baseline.
"""
from __future__ import annotations

from .model_a import WaferLensModel


def build_model_b(wave: int = 1, **kw) -> WaferLensModel:
    """Model B: everything Model A sees, plus the 2000 block readings. Deliverable 2."""
    return WaferLensModel(use_blocks=True, wave=wave, **kw)
