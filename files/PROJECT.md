# WaferLens — Complete Project Specification
### Multi-Resolution Die Yield Prediction with Interpretable Spatial Context
**SanDisk Hackathon · 24 hours · 4 people · Build iteratively, ship every wave**

---

# 1. WHAT WE ARE BUILDING

## 1.1 The problem in plain terms

A semiconductor wafer is a disc of silicon holding thousands of near-identical chips ("dies") in a grid. After fabrication every die is electrically probed and marked pass or fail. Some dies are already known-failed before this round of testing (`old_label == 1`); the rest look fine going in.

**Our job: predict which of the currently-passing dies will fail.**

Three signals are available:
- **Die-level parametrics** — n electrical measurements per die
- **Spatial context** — where the die sits, and what its neighbours look like
- **Block-level readings** — ~2000 sub-die measurements per die, capturing fine-grained within-die process variation

The core scientific question the hackathon poses: **does the block-level signal actually improve prediction over die + spatial alone?** We must answer that honestly, either way.

## 1.2 Why spatial context matters (the domain insight)

Dies do not fail independently. Failures cluster into recognisable wafer-scale shapes — rings, donuts, scratches, edge bands — and each shape traces to a physical process fault. The industry has a twenty-year-old name for screening on this: **GDBN, "Good Die in a Bad Neighborhood"** — scrapping a die that tests fine but is surrounded by failures. The incumbent statistical methods are PAT → DPAT → NNR → GDBN, and they assume univariate Gaussian distributions, which breaks down at advanced nodes.

Our `m × m` neighbourhood window is a learned replacement for that heuristic.

## 1.3 The idea that wins

> **Every other team hands the judges a probability. We hand them a diagnosis.**

Our headline output for any predicted failure:

```
Die (14, 22)  —  failure risk 0.83   [conformal interval 0.71–0.91]
├── 0.31  DIE      → IDDQ sits 3.2σ above its spatially-expected value
├── 0.28  SPATIAL  → neighbourhood matches an Edge-Ring signature (5×5 BNR = 0.68)
└── 0.24  BLOCK    → attention concentrated on blocks 1420–1610
     ↳ ROOT CAUSE HYPOTHESIS: RTP / anneal temperature regulation
     ↳ ACTION: screen die; flag lot for thermal-module review
```

Three differentiators:
1. **Attribution decomposed by resolution** — not just "which features", but "how much came from each level of the hierarchy". Directly answers deliverable 4.
2. **Interpretability is architectural** — block explanations come from attention weights that *are* the model, not from post-hoc SHAP over 2000 dimensions.
3. **Root-cause mapping** — spatial signature → physical process module. Turns "interpretable" into "actionable", which is the PS's stated engineering objective.

---

# 2. DELIVERABLES & GRADING

## 2.1 Required deliverables (from the PS — these are non-negotiable)

| # | Deliverable | Where it lives |
|---|---|---|
| 1 | **Model A**: (m×n) die features + spatial context → (m×1) failure probabilities | `src/model_a.py` |
| 2 | **Model B**: die features + spatial + block readings → (m×1) failure probabilities | `src/model_b.py` |
| 3 | **Comparison analysis**: side-by-side F1, AUC-PR, accuracy showing improvement or lack thereof | `results/comparison.md` |
| 4 | **Interpretability output**: per-die feature importance + spatial contribution viz (A), plus block pattern analysis (B) | `src/explain.py`, `figures/` |
| 5 | **Imbalance / overlap analysis**: how each model behaves under 2–5% failure rate and overlapping distributions | `results/imbalance_analysis.md` |

> Note on notation: in "(m × n) matrix → (m × 1)", **m = number of dies, n = number of parametrics**. In "m × m window", m = neighbourhood size. Different m's. Output is one flat probability per die.

## 2.2 Rubric mapping

| Criterion | Weight | Our answer |
|---|---|---|
| Model A vs B performance (F1, AUC-PR on minority) | 30% | Wave 2 onward; baselines included for context |
| Handling of imbalance (no trivial all-pass) | 20% | Per-wafer adaptive weighting, PR-threshold tuning, calibration curve, all-pass baseline shown for contrast |
| Interpretability (attribution, block patterns, spatial heatmaps) | 30% | Resolution-decomposed attribution + 5 figure types + root-cause table |
| **Creative** multi-resolution fusion + A→B delta | 20% | Late-fusion attention-MIL; **the word "creative" is in the rubric — the architecture choice itself scores** |

## 2.3 Explicitly NOT required (do these only if time allows)
- DPAT / GDBN classical baselines *(strong differentiator, cap at 1 hour)*
- Yield-loss vs DPPM economic curve *(strong differentiator, wave 4)*
- Conformal prediction intervals *(strong differentiator, wave 4)*
- Streamlit dashboard *(no UI is prescribed; huge demo value, but never at the cost of §2.1)*

---

# 3. THE ITERATIVE BUILD PHILOSOPHY

**We do not build components. We build increasingly good versions of the whole system.**

```
WAVE 1  ████░░░░░░  everything exists, everything is terrible     → SUBMITTABLE
WAVE 2  ███████░░░  everything exists, core is good               → SUBMITTABLE
WAVE 3  █████████░  everything exists, models + explain are strong → SUBMITTABLE
WAVE 4  ██████████  differentiators layered on                    → SUBMITTABLE
WAVE 5  ██████████  frozen, presented
```

**Rules:**
- Every wave ends with `python run_pipeline.py` completing, metrics written to `results/results.csv`, and a git tag.
- If a wave overruns, ship what works and move on. Never carry an unfinished component into the next wave.
- The dashboard is wired to random numbers from **hour 3** and swaps in real predictions as they land.

**Wave exit criteria are listed per wave in §6. Do not start wave N+1 until wave N's exit criteria pass.**

---

# 4. THE FROZEN CONTRACT

Agreed at hour 1. **Never changed.** This is what lets four people work in parallel.

```python
# src/contract.py
from dataclasses import dataclass, field
import numpy as np

@dataclass
class DiePrediction:
    wafer_id: str
    die_x: int
    die_y: int
    y_true: int                     # 0 pass, 1 fail (post-test)
    old_label: int                  # 1 = pre-test failure; these are EXCLUDED from train/eval
    p_fail: float                   # calibrated probability
    y_pred: int                     # after tuned threshold
    attribution: dict               # {"die": float, "spatial": float, "block": float} sums to 1.0
    top_features: list              # [(feature_name, signed_shap_value), ...] top 5
    block_attention: np.ndarray | None = None   # shape (n_blocks,) or (n_chunks,), Model B only
    conformal_lo: float | None = None
    conformal_hi: float | None = None
    root_cause: str | None = None   # from the spatial-signature lookup
```

Every model returns `list[DiePrediction]`. The explainer, the dashboard, and the metrics module consume only this.

**Metrics log format** — append one row per run to `results/results.csv`:
```
run_id, timestamp, wave, model, features_version, f1, auc_pr, auc_roc, accuracy,
precision, recall, threshold, brier, n_train, n_test, fail_rate, notes
```

---

# 5. REPO LAYOUT

```
waferlens/
├── CLAUDE.md                  # hard constraints — read first
├── PROJECT.md                 # this file
├── PS.pdf                     # original problem statement
├── requirements-ours.txt
├── run_pipeline.py            # ONE command runs everything end to end
│
├── data/
│   ├── LSWMD.pkl              # downloaded
│   └── generated/             # output of their generate_data.py — DO NOT EDIT their script
│
├── src/
│   ├── contract.py            # §4 — write this first
│   ├── io_data.py             # load, wafer-level split, old_label filtering, assertions
│   ├── recon.py               # wave 0 data diagnostics (see §7)
│   ├── features_die.py        # P1 — parametric transforms, per-wafer z-scores
│   ├── features_spatial.py    # P1 — neighbourhood, residuals, radial, pattern class
│   ├── features_block.py      # P3 — statistical pooling of the 2000 readings
│   ├── baselines.py           # P2 — all-pass, DPAT, NNR, classic GDBN
│   ├── model_a.py             # P2 — GBDT
│   ├── block_encoder.py       # P3 — attention-MIL (wave 3)
│   ├── model_b.py             # P3 — pooling version (wave 2), fusion version (wave 3)
│   ├── imbalance.py           # P2 — weighting, focal loss, threshold tuning
│   ├── calibrate.py           # P2 — isotonic/Platt, conformal (wave 4)
│   ├── economics.py           # P2 — yield-loss vs escape curve (wave 4)
│   ├── explain.py             # P4 — SHAP, attribution decomposition, root-cause map
│   ├── figures.py             # P4 — all plots
│   └── metrics.py             # shared — F1, AUC-PR, accuracy, calibration, logging
│
├── app/
│   └── app.py                 # P4 — Streamlit dashboard
│
├── results/
│   ├── results.csv            # every run, every wave
│   ├── comparison.md          # deliverable 3
│   └── imbalance_analysis.md  # deliverable 5
│
├── figures/                   # deliverable 4 outputs
└── notebooks/                 # scratch only, never imported
```

**`run_pipeline.py` must support:**
```bash
python run_pipeline.py --wave 1              # run everything at wave-1 quality
python run_pipeline.py --model a --wave 2
python run_pipeline.py --model b --wave 3 --explain
python run_pipeline.py --all --report        # regenerate every table and figure
```

---

# 6. THE WAVES

## 🟤 WAVE 0 — Recon & Skeleton · H0–H2 · ALL FOUR
*No modelling. Everyone in one room.*

- [ ] Run their setup: venv → `pip install -r requirements.txt` → download LSWMD.pkl → `generate_data.py`
- [ ] Write `src/contract.py` and `src/metrics.py` — everyone codes against these
- [ ] Run `src/recon.py` and answer the five questions in §7
- [ ] Create repo, one branch per person, `results/results.csv` with headers
- [ ] Assign tracks (§9)

**Exit criteria:** contract file merged, five recon questions answered in writing, everyone can load the data.

---

## 🟢 WAVE 1 — THE SPIKE · H2–H5 · ALL FOUR
**Goal: the dumbest possible complete system. Every box in the architecture exists and does something.**

This is the most important wave in the project. Resist all temptation to make anything good.

| Component | Wave-1 version (deliberately trivial) |
|---|---|
| Data | Load, wafer-split, drop `old_label==1`, assert no leakage |
| Die features | Raw parametrics, nothing else |
| Spatial features | **Exactly three**: 3×3 bad-neighbour ratio, radial distance, wafer yield |
| Model A | `LGBMClassifier(class_weight='balanced')`, default params |
| Block features | **Exactly three**: mean, std, max of the 2000 readings |
| Model B | Same LightGBM, block features appended |
| Metrics | F1, AUC-PR, accuracy, threshold tuned on val PR curve |
| Explain | `shap.TreeExplainer` → one waterfall plot saved to `figures/` |
| Dashboard | Streamlit page: pick wafer → show risk heatmap (matplotlib is fine) |
| Report | `results/comparison.md` auto-generated with the A vs B table |

**Exit criteria (H5):**
- ✅ `python run_pipeline.py --wave 1` completes end to end
- ✅ A and B both produce `list[DiePrediction]`
- ✅ A row for each in `results.csv`
- ✅ One SHAP figure and one wafer heatmap on disk
- ✅ `git tag wave1-submittable`

> 🔒 **From this moment you have something to submit.** Everything after is upside.

---

## 🟡 WAVE 2 — DEPTH PASS 1 · H5–H10 · parallel tracks
**Goal: make the core genuinely good. Still no exotic architectures.**

### P1 — Spatial feature factory
- Multi-scale bad-neighbour ratio: 3×3, 5×5, 7×7, 9×9 *(fabs localise dynamic PAT to 5×5/7×7 windows for exactly this reason)*
- Directional asymmetry (N/S/E/W bad counts) → separates scratches from edge bands
- Longest run of consecutive bad neighbours
- **Parametric spatial residuals** ⭐ — for each parametric, `value − median(k nearest neighbours)` via `scipy.spatial.cKDTree`. This is NNR generalised, and spatial-trend modelling is the most-validated replacement for DPAT in the literature.
- Radial distance, polar angle, edge-proximity flag
- Per-wafer z-scores of every parametric
- Wafer yield, lot yield

### P2 — Model A properly
Imbalance ladder, **in this order, measuring after each**:
1. `scale_pos_weight` / `class_weight='balanced'`
2. Threshold tuned on val PR curve to maximise F1
3. **Per-wafer adaptive class weighting** ⭐ — compute the minority weight from *that wafer's* imbalance rather than globally
4. Focal loss as custom objective if 1–3 plateau
5. SMOTE **last, and expect it to hurt** — the PS warns about overlapping distributions and synthetic minority points land straight in the overlap

Plus: `baselines.py` (all-pass, DPAT ±3σ per wafer, NNR, GDBN threshold sweep). **One hour max.**

### P3 — Model B Tier 1 (statistical pooling)
Collapse 2000 readings to ~25 features: mean, std, median, skew, kurtosis, p10/p50/p90/p95/p99, min, max, range, IQR, count and fraction above a train-derived hot threshold, longest run of consecutive hot blocks, index of worst block, centroid and spread of hot blocks, distribution entropy.

Feed into the same LightGBM. **Measure the delta. This is a complete Model B.**

### P4 — Interpretability v1 + dashboard v2
- SHAP beeswarm + per-die waterfall
- **First version of the resolution-decomposed attribution bar** ⭐ (group SHAP by feature family: die / spatial / block, normalise)
- Wafer risk heatmap, pre-test vs predicted vs truth, side by side
- Dashboard: wafer selector, click-a-die, show attribution card

**Exit criteria (H10):**
- ✅ Model A beats all four baselines on AUC-PR
- ✅ Model B (pooling) trained, A→B delta measured and written to `comparison.md`
- ✅ Attribution decomposition renders for any die
- ✅ `git tag wave2-submittable`

---

## 🟠 WAVE 3 — DEPTH PASS 2 · H10–H16 · parallel tracks
**Goal: the parts that make judges remember you.**

### P3 — Attention-MIL block encoder ⭐ (the creativity 20%)
Treat each die as a **bag of 2000 block instances**. Gated attention pooling is permutation-invariant and its attention weights directly reveal each instance's contribution to the bag label (Ilse, Tomczak & Welling, ICML 2018). **Block interpretability falls out of the architecture** instead of running SHAP over 2000 dimensions.

```
blocks (2000,) → chunk into 100 × 20  (or 1D-conv if the readings are physically ordered)
               → shared MLP encoder → (100, d)
               → GATED ATTENTION POOLING → (d,) + attention weights (100,)
               → concat with Model-A branch embedding        [LATE FUSION]
               → MLP head → p_fail
```

Late fusion, not early: it suits heterogeneous modalities and lets each branch learn its own representation before combining — the same choice the current GDBN state of the art made.

**Fallback ladder if MIL underperforms:** 1D-CNN over the block sequence → PCA/UMAP embedding into GBDT → stay with wave-2 pooling.

**⏰ HARD CUTOFF H16.** If MIL isn't beating pooling by then, ship pooling and *report the finding*. "We tested a learned encoder against statistical pooling and pooling won" satisfies the rubric — the PS explicitly asks for improvement **or lack thereof**.

### P1 — Wafer defect-pattern classifier as a feature ⭐
Classify the pre-test wafer map into Center / Donut / Edge-Loc / Edge-Ring / Loc / Near-full / Random / Scratch, then feed class + confidence as die features. This is the "pattern-aware GDBN" upgrade: extending observation from a narrow window to the whole wafer.

Use an existing repo, do **not** train from scratch. Classical shape features (`skimage.regionprops`, Radon projections — the original WM-811K paper's approach) are a fast and legitimate alternative.

### P4 — Full interpretability suite (30% of the grade — do not compress)
Five artifacts, all auto-generated:
1. Per-die SHAP waterfall
2. **⭐ Resolution-decomposed attribution bar** — the signature visual
3. Wafer risk heatmap + heatmap of spatial-feature SHAP (*see where neighbourhood context drove decisions*)
4. **Block attention profile** — attention across blocks for one failing die, hot regions annotated (Model B)
5. Global summary — beesw arm, feature-family importance, failure-mode clusters

**⭐ ROOT-CAUSE MAPPING TABLE** — implement as a lookup in `explain.py`:

| Spatial signature | Documented process root cause | Engineering action |
|---|---|---|
| Center | Irregular RF operation / unusual liquid flow | Chamber RF + flow check |
| Donut | Residue resistant to photoresist cleaning | Clean / strip step review |
| Edge-Ring | Anomalous temperature regulation during RTP | Thermal module |
| Edge-Loc | Irregular anneal temperature, load-lock valve impurities | Load-lock inspection |
| Scratch | Mechanical handling | Handler / robot inspection |
| Random / Near-full | Systemic or gross process excursion | Escalate lot |

> **Say on stage:** Edge-Loc and Scratch look alike — elongated clusters near the edge — but demand *opposite* interventions: thermal tuning vs mechanical handling. Confusing them sends the fab after the wrong module. Our system separates them and names the module.

### P2 — Tuning + imbalance analysis (deliverable 5)
Optuna on **AUC-PR**, 100 trials max, timeboxed. Then write `imbalance_analysis.md`: PR curves for all models, class-overlap visualisation (parametric distributions for pass vs fail, showing the overlap the PS warns about), and how each model's behaviour changes across thresholds.

**Exit criteria (H16):**
- ✅ Model B final architecture chosen and justified with numbers
- ✅ All five interpretability artifacts generating automatically
- ✅ `imbalance_analysis.md` written
- ✅ `git tag wave3-submittable`

---

## 🔵 WAVE 4 — DIFFERENTIATORS · H16–H20 · P2 + P4
*Optional per the PS. Extremely high value if the required deliverables are already done.*

### 4.1 — Calibration + conformal prediction ⭐
- Reliability diagram + Brier score — are the probabilities *honest*?
- Per-die conformal intervals. Conformalized quantile regression achieved ~90% coverage guarantees on 5nm automotive silicon, distribution-free with no parametric assumption; weighted conformal scoring holds coverage under cross-lot shift where standard models fail.
- **Almost nobody at a hackathon knows this exists.** It reframes you from "students with a model" to "people who understand deployment."
- Tool: `MAPIE`, `sklearn.calibration`

### 4.2 — Yield-loss vs escape tradeoff curve ⭐
Sweep the threshold; plot yield loss (%) against escaped failures. The costs are asymmetric — a false accept becomes a field failure and RMA cost, a false reject is pure yield loss. Mark and justify the chosen operating point.

Reference for scale: the current GDBN SOTA reports **2.7% yield loss vs 6.3%** for prior methods and a **67.8% DPPM reduction** (129.4 from 407.8) against the prior 60.2% and 58%.

### 4.3 — Cost-slider in the demo
Expose the cost ratio (field-failure cost ÷ die cost) as a slider; the threshold and the wafer map update live. Judges love turning a knob.

### 4.4 — Dashboard polish
1. Wafer selector → pre-test / predicted-risk / truth heatmaps side by side
2. Click a die → attribution card + root-cause hypothesis + conformal interval
3. **Model A ↔ B toggle on the same die** — shows the delta *qualitatively*, not just in a table
4. Metrics tab: PR curves, confusion matrices, calibration plot, yield/escape curve
5. Cost slider

**Exit criteria (H20):** 🔒 **FULL FREEZE.** No more training. Whatever exists is what you present. `git tag final`.

---

## 🟣 WAVE 5 — DECK & REHEARSAL · H20–H24 · ALL FOUR

**Deck order — do not lead with your architecture:**
1. **The problem in fab language.** 2–5% of dies fail, failures cluster, and industry's incumbent answer is called GDBN.
2. **The incumbent is known-broken.** DPAT assumes univariate Gaussians; at advanced nodes distributions are non-Gaussian and multivariate parameter interactions carry signal it can't exploit. Show your baseline numbers.
3. **Model A beats it.** Table with baselines as rows.
4. **Model B — does resolution help?** The delta, with an honest verdict.
5. **⭐ LIVE DEMO — 40% of your time.** Click a die. Show the decomposition. Name the root cause.
6. **Imbalance & overlap.** PR curves, calibration, the overlap visualisation.
7. **Next steps.** Cross-lot generalisation, RL-based screening policy, real fab data.

**Killer evidence slide:** a randomized controlled field experiment at a semiconductor line using gradient-boosted trees + SHAP achieved **21.7% yield-loss reduction**, rising to **51.3%** on production rollout (p<0.001) — while conventional LASSO found *zero* meaningful parameters on the same data. Black-box accuracy without explanation produced no improvement. *(Senoner, Netland & Feuerriegel, Management Science, 2022.)*

**Rehearse three times, timed. Cut whatever doesn't survive run three.**

---

# 7. WAVE-0 RECON: THE FIVE QUESTIONS

`src/recon.py` must answer these and print them. **They determine the architecture.**

1. **What is the actual failure rate** in train/val/test among `old_label == 0` dies? Is it really 2–5%?
2. **Are block readings ordered** (index maps to physical position) or an unordered bag?
   → ordered ⇒ 1D-CNN is viable; unordered ⇒ MIL / pooling only.
3. **⭐ Does spatial↔parametric correlation exist?** Correlate each die's parametric z-score against its neighbourhood bad-ratio.
   → If yes, spatial features are gold and P1 leads. If no, the block signal must carry Model B and P3 leads.
   *(Context: the current GDBN SOTA synthesised parametrics randomly and admits this destroyed the spatial-parametric link, so their model structurally could not learn the interaction. If SanDisk's generator preserved it, we can model something the published SOTA could not.)*
4. **Are wafers cleanly separated** across the provided splits, or do dies from one wafer appear in several?
5. **How much do pass and fail distributions overlap** per parametric? (Plot them. This becomes a figure in deliverable 5.)

Also print: n dies, n wafers, n parametrics, n blocks, dtypes, missing-value counts, memory footprint.

---

# 8. TOOLS & OPEN-SOURCE RESOURCES

```bash
pip install numpy pandas pyarrow scipy scikit-learn scikit-image \
            lightgbm xgboost catboost optuna imbalanced-learn \
            shap captum torch mapie \
            streamlit plotly matplotlib seaborn umap-learn mlflow \
            balanced-loss imbalance-xgboost
```

| Need | Resource |
|---|---|
| Dataset | kaggle.com/datasets/qingyi/wm811k-wafer-map |
| Neighbourhood ops | `scipy.ndimage` (`uniform_filter`, `convolve`) |
| KNN spatial residuals | `scipy.spatial.cKDTree`, `sklearn.neighbors` |
| GP spatial detrend | `sklearn.gaussian_process`, [GPyTorch](https://github.com/cornellius-gp/gpytorch) |
| Wafer pattern CNN (reuse!) | [JingminSun/WM811K](https://github.com/JingminSun/WM811K) — full download→preprocess→train→eval pipeline |
| Alt wafer CNN | [abuebayyeh/waferCNN](https://github.com/abuebayyeh/waferCNN) |
| Self-supervised wafer embeddings | [faris-k/fastsiam-wafers](https://github.com/faris-k/fastsiam-wafers) |
| Supervised-contrastive wafer classifier | [YoungJaeBae/supcon_wmpc](https://github.com/YoungJaeBae/supcon_wmpc) |
| Classical shape features | `skimage.measure.regionprops`, `skimage.transform.radon` |
| GBDT | [LightGBM](https://github.com/microsoft/LightGBM) · [XGBoost](https://github.com/dmlc/xgboost) · [CatBoost](https://github.com/catboost/catboost) |
| Resampling | [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) |
| Focal loss — XGBoost | [Imbalance-XGBoost](https://github.com/jhwjhw0123/Imbalance-XGBoost) |
| Focal loss — CatBoost | [rahowa/catboost_focal_loss](https://github.com/rahowa/catboost_focal_loss) |
| Focal loss — PyTorch | [fcakyon/balanced-loss](https://github.com/fcakyon/balanced-loss) |
| **Attention-MIL (official)** | [AMLab-Amsterdam/AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) |
| Attention-MIL tutorial | [Keras MIL example](https://keras.io/examples/vision/attention_mil_classification) |
| Gated attention modules | [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) |
| Set Transformer (if 2000 is slow) | [juho-lee/set_transformer](https://github.com/juho-lee/set_transformer) |
| Attributions | [shap](https://github.com/shap/shap) · [captum](https://github.com/pytorch/captum) |
| Conformal | [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) |
| Tuning | [Optuna](https://github.com/optuna/optuna) |
| Dashboard | [Streamlit](https://github.com/streamlit/streamlit) · `plotly` |

---

# 9. TEAM

| | Owner | Owns | Primary files |
|---|---|---|---|
| **P1** | Spatial Lead | Feature factory, neighbourhood ops, spatial residuals, pattern classifier | `features_die.py`, `features_spatial.py` |
| **P2** | Model Lead | Model A, baselines, imbalance, tuning, calibration, economics | `model_a.py`, `baselines.py`, `imbalance.py`, `calibrate.py` |
| **P3** | Block Lead | Block pooling, MIL encoder, fusion, Model B | `features_block.py`, `block_encoder.py`, `model_b.py` |
| **P4** | Story Lead | SHAP suite, figures, dashboard, deck, README, rehearsal | `explain.py`, `figures.py`, `app/app.py` |

**Sync points:** H2, H5 (wave-1 gate), H10 (wave-2 gate), H16 (wave-3 gate), H20 (freeze). 10 minutes each, standing up.

**Sleep rotation:** P1+P3 rest H13–H17, P2+P4 rest H17–H21. Nobody presents on zero sleep.

**Working with Claude Code:** one session per track, each pointed at this file and `CLAUDE.md`. Give it one wave-scoped task at a time ("implement wave-2 spatial features per PROJECT.md §6"), review the diff, commit, next. Do **not** hand it the whole project and walk away.

---

# 10. RISK REGISTER

| Risk | Mitigation |
|---|---|
| MIL doesn't converge in time | Wave-2 pooling frozen at H10; hard cutoff H16; fallback ladder documented |
| Block signal has no predictive value | That IS a valid answer — PS asks for "improvement or lack thereof". Quantify rigorously and own it |
| SHAP too slow on 2000 raw features | Explain pooled features, use attention weights for the block story |
| Dashboard unfinished | Wired to fake data from H3; ugly-but-working beats beautiful-but-planned |
| Wafer-level leakage | Asserted in `io_data.py`, verified at H2 |
| `old_label==1` dies leak into eval | Asserted in `io_data.py` |
| Team exhausted at presentation | Enforced sleep rotation |
| Merge conflicts | Frozen contract, one file per person, sync every gate |
| Over-engineering early | Wave gates; no tuning before wave 3 |

---

# 11. ANTI-PATTERNS — DO NOT

- ❌ Build a wafer-map CNN from scratch. Saturated track (ViT-Tiny reports 98.4% F1) and not our task. Reuse a repo, use it as a feature.
- ❌ Reach for tabular foundation models (TabPFN) on 2000-dim block data — TabPFN v2 degrades on large-scale, high-dimensional datasets.
- ❌ Optimise accuracy. All-pass scores ~96% here. The PS says so explicitly.
- ❌ Leave the threshold at 0.5.
- ❌ Save interpretability for the last 3 hours. It's worth as much as the model.
- ❌ Spend wave 1 making anything good.
- ❌ Refactor before wave 4.

---

# 12. THE 60-SECOND PITCH

> "Fabs already have a method for this — it's called Good Die in a Bad Neighborhood, basically a threshold on how many of your neighbours failed. It's twenty years old and it breaks at advanced nodes because it assumes univariate Gaussian distributions.
>
> We built two models. Model A replaces GDBN with learned spatial context — multi-scale neighbourhood statistics and spatially-detrended parametric residuals. Model B adds two thousand block-level readings through a gated attention mechanism, fused late.
>
> But here's what we actually care about: our explanation is decomposed by resolution. For any predicted failure we tell you how much came from the die's own parameters, how much from its neighbourhood, and how much from its internal blocks — and we map the spatial signature to a physical process module.
>
> So the output isn't a probability. It's a work order."

---

**Wave 1 by hour 5. Everything else is upside. Go. 🔥**
