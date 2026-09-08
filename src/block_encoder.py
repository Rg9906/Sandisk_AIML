"""
Gated attention-MIL encoder over the 2000 block readings  --  the creativity 20%.

WHY MIL. Each die is a *bag* of sub-die block instances, and the die fails if some
unknown subset of its blocks is defective. That is the multiple-instance problem in its
textbook form. Gated attention pooling (Ilse, Tomczak & Welling, ICML 2018) is
permutation-invariant and, critically, its attention weights ARE the model -- so block
interpretability falls out of the architecture instead of requiring post-hoc SHAP over
2000 dimensions, which would be both slow and unstable.

WHY IT SHOULD WORK HERE. Recon Q2 measured lag-1 autocorrelation of 0.25 along the block
index against a 0.00 shuffled control: the readings are ORDERED, and the kit injects a
failing die's anomaly as a contiguous cluster. So there is a real, localised structure
for attention to find and point at, rather than diffuse noise.

    blocks (2000,) -> 100 chunks x 20 -> shared MLP -> (100, d)
                   -> GATED ATTENTION POOLING -> (d,) + attention (100,)
                   -> concat with the tabular branch          [LATE FUSION]
                   -> MLP head -> p_fail

LATE fusion, not early: the modalities are heterogeneous (dense tabular vs a long
sequence), and late fusion lets each branch learn its own representation before they
meet. Early concatenation would let 2000 raw block dims swamp 63 tabular ones.

This runs CPU-only on a laptop, so it is deliberately small and time-boxed. If it does
not beat statistical pooling by the wave-3 cutoff we ship pooling and report the finding
-- the PS asks for improvement "or lack thereof", and that is a real answer.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

SEED = 42
# Block readings are generated as N(100, 15); standardising keeps the encoder well-conditioned.
BLOCK_MEAN, BLOCK_STD = 100.0, 15.0


class GatedAttentionMIL(nn.Module):
    """Chunked block encoder + gated attention pooling + late fusion with tabular features."""

    def __init__(self, n_tabular: int, n_blocks: int = 2000, chunk: int = 20,
                 d_inst: int = 48, d_att: int = 32, d_tab: int = 32):
        super().__init__()
        assert n_blocks % chunk == 0, "block count must divide evenly into chunks"
        self.chunk = chunk
        self.n_chunks = n_blocks // chunk

        # Shared encoder applied to every chunk independently (weights tied across chunks).
        self.instance = nn.Sequential(
            nn.Linear(chunk, d_inst), nn.ReLU(),
            nn.Linear(d_inst, d_inst), nn.ReLU(),
        )

        # Gated attention: V is the "what is here" path, U the "how much does it matter"
        # gate. Their product feeds a single logit per instance, softmaxed into weights.
        self.att_V = nn.Linear(d_inst, d_att)
        self.att_U = nn.Linear(d_inst, d_att)
        self.att_w = nn.Linear(d_att, 1)

        self.tabular = nn.Sequential(
            nn.Linear(n_tabular, d_tab), nn.ReLU(), nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(d_inst + d_tab, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, blocks: torch.Tensor, tab: torch.Tensor,
                return_attention: bool = False):
        b = blocks.shape[0]
        h = self.instance(blocks.view(b, self.n_chunks, self.chunk))   # (b, C, d)

        gate = torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))
        a = torch.softmax(self.att_w(gate).squeeze(-1), dim=1)          # (b, C)

        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)                     # (b, d)
        logit = self.head(torch.cat([z, self.tabular(tab)], dim=1)).squeeze(-1)
        return (logit, a) if return_attention else logit


class MILBlockModel:
    """Trains the MIL encoder and exposes probabilities plus per-die attention."""

    def __init__(self, *, epochs: int = 8, batch_size: int = 512, lr: float = 2e-3,
                 chunk: int = 20, time_budget_s: float = 900.0, seed: int = SEED):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.chunk = chunk
        self.time_budget_s = time_budget_s
        self.seed = seed
        self.net: GatedAttentionMIL | None = None
        self.tab_mean_: np.ndarray | None = None
        self.tab_std_: np.ndarray | None = None
        self.history_: list[dict] = []

    # ------------------------------------------------------------------ utils
    def _standardise(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.tab_mean_ = X.mean(axis=0)
            self.tab_std_ = X.std(axis=0) + 1e-6
        return ((X - self.tab_mean_) / self.tab_std_).astype(np.float32)

    @staticmethod
    def _blocks_batch(blocks: np.ndarray, rows: np.ndarray) -> torch.Tensor:
        # Memmap fancy-indexing needs sorted rows to stay sequential on disk.
        arr = np.asarray(blocks[np.sort(rows)], dtype=np.float32)
        inv = np.argsort(np.argsort(rows))
        arr = arr[inv]
        return torch.from_numpy((arr - BLOCK_MEAN) / BLOCK_STD)

    # -------------------------------------------------------------------- fit
    def fit(self, blocks: np.ndarray, X_tab: np.ndarray, y: np.ndarray,
            rows: np.ndarray, *, val: tuple | None = None) -> "MILBlockModel":
        """
        `rows` maps each training example to its row in the (unfiltered) block memmap.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))

        Xs = self._standardise(X_tab, fit=True)
        self.net = GatedAttentionMIL(n_tabular=Xs.shape[1],
                                     n_blocks=blocks.shape[1], chunk=self.chunk)

        pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)

        n = len(y)
        rng = np.random.default_rng(self.seed)
        t0 = time.time()

        for epoch in range(self.epochs):
            self.net.train()
            perm = rng.permutation(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                sel = perm[i:i + self.batch_size]
                xb = self._blocks_batch(blocks, rows[sel])
                tb = torch.from_numpy(Xs[sel])
                yb = torch.from_numpy(y[sel].astype(np.float32))

                opt.zero_grad()
                loss = loss_fn(self.net(xb, tb), yb)
                loss.backward()
                opt.step()
                total += float(loss) * len(sel)

            rec = {"epoch": epoch + 1, "train_loss": total / n,
                   "elapsed_s": round(time.time() - t0, 1)}
            if val is not None:
                from sklearn.metrics import average_precision_score
                vb, vx, vy, vr = val
                p = self.predict_proba(vb, vx, vr)
                rec["val_auc_pr"] = float(average_precision_score(vy, p))
            self.history_.append(rec)
            print(f"    epoch {rec['epoch']:>2}  loss={rec['train_loss']:.4f}"
                  + (f"  val AUC-PR={rec['val_auc_pr']:.4f}" if "val_auc_pr" in rec else "")
                  + f"  ({rec['elapsed_s']:.0f}s)")

            # Hard time budget: wave 3 has a cutoff and we honour it in code.
            if time.time() - t0 > self.time_budget_s:
                print(f"    time budget {self.time_budget_s:.0f}s reached -- stopping early")
                break
        return self

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def predict_proba(self, blocks: np.ndarray, X_tab: np.ndarray,
                      rows: np.ndarray) -> np.ndarray:
        self.net.eval()
        Xs = self._standardise(X_tab)
        out = np.empty(len(rows), dtype=np.float32)
        for i in range(0, len(rows), self.batch_size):
            sel = slice(i, i + self.batch_size)
            xb = self._blocks_batch(blocks, rows[sel])
            tb = torch.from_numpy(Xs[sel])
            out[sel] = torch.sigmoid(self.net(xb, tb)).numpy()
        return out

    @torch.no_grad()
    def attention(self, blocks: np.ndarray, X_tab: np.ndarray,
                  rows: np.ndarray) -> np.ndarray:
        """
        Per-die attention over the 100 chunks -- the block interpretability artifact.

        These weights are not an explanation bolted on afterwards; they are the pooling
        operation itself, so they are exact by construction.
        """
        self.net.eval()
        Xs = self._standardise(X_tab)
        out = np.empty((len(rows), self.net.n_chunks), dtype=np.float32)
        for i in range(0, len(rows), self.batch_size):
            sel = slice(i, i + self.batch_size)
            xb = self._blocks_batch(blocks, rows[sel])
            tb = torch.from_numpy(Xs[sel])
            _, a = self.net(xb, tb, return_attention=True)
            out[sel] = a.numpy()
        return out
