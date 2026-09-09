import numpy as np
from sklearn.metrics import average_precision_score
from src.io_data import load_all
from src.model_a import WaferLensModel
from src.block_encoder import MILBlockModel
from src.metrics import compute_metrics, tune_threshold

tr, va, te = load_all('full')
el = {s.name: s.eligible for s in (tr, va, te)}
y = {s.name: s.y()[el[s.name]] for s in (tr, va, te)}

tab = WaferLensModel(use_blocks=False, wave=3, die_version='v2',
                     spatial_version='v2', two_stage=True).fit(tr, va)
mats = {s.name: tab.design_matrix(s) for s in (tr, va, te)}
sig = [c for c in mats['train'].columns if c.startswith('sp_sig_')]

# How badly do rare one-hots blow up under train-statistic standardisation?
Xtr = mats['train'].to_numpy(np.float32)[el['train']]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xte = mats['test'].to_numpy(np.float32)[el['test']]
z = np.abs((Xte - mu) / sd).max(0)
cols = list(mats['train'].columns)
worst = np.argsort(-z)[:5]
print("  largest standardised test values (train mean/std):", flush=True)
for i in worst:
    print(f"    {cols[i]:24s} max|z| = {z[i]:9.1f}   train sd = {sd[i]:.5f}", flush=True)

for tag, drop in (("with sp_sig_* one-hots", False), ("without sp_sig_*", True)):
    def prep(name, s):
        X = mats[name]
        if drop:
            X = X.drop(columns=sig)
        return X.to_numpy(np.float32)[el[name]], np.flatnonzero(el[name])
    Xt, rt = prep('train', tr); Xv, rv = prep('val', va); Xs, rs = prep('test', te)
    m = MILBlockModel(epochs=5, time_budget_s=420).fit(tr.blocks, Xt, y['train'], rt)
    pv = m.predict_proba(va.blocks, Xv, rv)
    thr, _ = tune_threshold(y['val'], pv)
    r = compute_metrics(y['test'], m.predict_proba(te.blocks, Xs, rs), thr)
    print(f"  MIL {tag:24s} val={average_precision_score(y['val'],pv):.4f}  "
          f"TEST={r['auc_pr']:.4f}", flush=True)
