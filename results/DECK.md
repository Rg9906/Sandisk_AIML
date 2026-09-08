# WaferLens — presentation running order

*Do not lead with the architecture. Lead with the problem in fab language, show the
incumbent is broken, then earn the architecture.*

Target: ~10 minutes, with **4 minutes on the live demo**. Rehearse three times, timed,
and cut whatever does not survive run three.

---

## 1 · The problem, in fab language (60s)

A wafer holds a few hundred to a few thousand dies. After fabrication every die is
probed. Some are already known-failed — those are the red ones, and they are *input*, not
our target. Our job is to predict which of the **green** dies turn orange after the next
test round.

- ~4% of eligible dies newly fail. That number is the whole difficulty.
- The fab already has a method for this. It is called **GDBN — Good Die in a Bad
  Neighborhood** — and it is twenty years old: scrap a die that tests fine but is
  surrounded by failures.

**Open the dashboard on a wafer here.** Pre-test / predicted / truth, three panels.

---

## 2 · We tried the incumbent first, and measured it (90s)

We implemented the actual ladder the industry uses — PAT → DPAT → NNR → GDBN — before
building anything of our own.

| method | AUC-PR | fail-F1 | accuracy |
|---|---|---|---|
| all-pass ("predict everything passes") | 0.042 | **0.000** | **0.958** |
| DPAT (3σ, per wafer) | ~0.046 | 0.083 | 0.802 |
| GDBN (5×5) | ~0.052 | 0.104 | 0.831 |
| NNR (k=8) | ~0.075 | 0.115 | 0.897 |

> **The line to say out loud:** "Predicting that every die passes scores 95.8% accuracy.
> If accuracy is your metric, the best model is the one that does nothing. That is why
> every number we show you today is AUC-PR on the failure class."

**Why the incumbents fail here:** DPAT assumes each parameter is univariate Gaussian and
thresholds it. We measured the separability of all 500 parametrics — median Cohen's d is
**0.134**, best single feature 0.248. There is no parameter to threshold. This is the
published critique of DPAT at advanced nodes, reproduced on our own data.

---

## 3 · What the data actually told us (2 min) — *the honest slide*

Three findings, two of which contradicted our plan. **This is the slide that separates us
from teams who only report what worked.**

**(a) The GDBN premise barely holds on this data.** Pooled across wafers, bad-neighbour
ratio correlates **−0.025** with new failures — zero, wrong sign, non-monotonic. Within a
wafer it is **+0.025**: right sign, still weak. The confounder is large — between wafers,
mean bad-neighbour ratio correlates **−0.376** with the wafer's new-failure rate.

**(b) So the real spatial signal is the wafer's own failure rate.** That rate is a hidden
per-wafer latent. It looks irreducible from one die — but prediction is **transductive**:
at inference we hold the entire wafer, so its own score distribution estimates the latent.
Two-stage design; **+0.046 AUC-PR, our single largest win.**

**(c) 500 weak features beat 5 strong ones — if you pool them.** Individually useless
(d = 0.134), but near-independent, so they add in quadrature to d = 3.22. An explicit
pooled anomaly score is worth **+0.028** while collapsing 500 columns to 38.

| what we added | Δ AUC-PR | expected? |
|---|---|---|
| transductive wafer prior | **+0.046** | no — added late, won biggest |
| multi-scale spatial factory | **−0.008** | no — we expected this to carry the model |
| pooled anomaly score | **+0.028** | yes |
| block features | **+0.038** | yes |

> **Say this:** "The multi-scale neighbourhood factory was the thing we were most
> confident about. It made the model slightly worse. We kept the measurement and changed
> our minds."

---

## 4 · Model A → Model B: does resolution help? (60s)

The PS asks whether the 2000 block readings improve prediction "or lack thereof".

**Model A and Model B are the same estimator with the same hyper-parameters.** B differs
only by appended block columns. If they differed architecturally, the delta would measure
our modelling choices rather than the data — and the question would go unanswered.

| | Model A | Model B | Δ |
|---|---|---|---|
| AUC-PR | 0.551 | 0.586 | **+0.035** |
| F1 (fail) | 0.529 | 0.550 | +0.021 |

**Yes — block-level data helps.** But the interesting part is *how little it takes*:

> **The 2000 block readings reduce to one number.** A single feature — the mean of the
> 2000 readings — captures **98%** of the entire block gain (0.5852 of 0.5859).

We did not assume that. We went looking for something better and failed, four times:

| attempt | Cohen's d | why it lost |
|---|---|---|
| plain mean | **0.69** | — |
| matched filter (sliding-window max) | 0.13 | max over ~2000 positions inflates the null more than it gains signal |
| mean − median | 0.06 | 5% contamination at 0.47σ shifts the whole distribution, it does not make a tail |
| tail-mass / soft-threshold | 0.10–0.21 | same reason |
| **gated attention-MIL** | — | **0.5801 vs pooling's 0.5859 — a near-tie** |

> **Say this:** "We built the fancy attention architecture. It tied with taking an
> average. We shipped the average — it is simpler, 40x faster on a CPU, and
> deterministic — and kept the attention weights only for the picture they draw."

*Be precise if pushed:* the gap is 0.0058 AUC-PR, which we would not defend as a real
difference in ranking quality. The ship decision is on engineering grounds, not accuracy.
It is also a number that MOVED: before we fixed a train-matrix bug the MIL scored 0.5481
and pooling looked like a comfortable 0.038 winner. The bug was handicapping the MIL.

*Why the mean is right, in one line:* the defect adds a **fixed total mass** across the
blocks, so the sum — the mean — is the sufficient statistic. Every cleverer detector
throws information away.

---

## 5 · ⭐ LIVE DEMO — 4 minutes, 40% of the talk

**Do not narrate the architecture. Click things.**

1. Pick a wafer with real failures. Three panels: pre-test → predicted risk → truth.
2. Click the highest-risk die.
3. **The attribution bar** — this is the moment. "We do not just tell you this die is
   risky. We tell you *which resolution* the risk came from: its own parameters, its
   neighbourhood, or its internal blocks."
4. **The root-cause card.** "Edge-Ring → anomalous temperature regulation during RTP →
   flag the lot for thermal-module review."
5. Say the Edge-Loc / Scratch line:
   > "Edge-Loc and Scratch both look like elongated clusters near the wafer edge. They
   > demand *opposite* interventions — thermal tuning versus mechanical handling.
   > Confusing them sends the fab after the wrong module. We separate them and name the
   > module."
6. **The output is not a probability. It is a work order.**

---

## 6 · Imbalance, honestly (60s)

We climbed the standard ladder and measured after every rung (`imbalance_analysis.md`).

| rung | AUC-PR | F1 |
|---|---|---|
| no weighting, threshold 0.5 | 0.5849 | 0.5347 |
| class_weight balanced, threshold 0.5 | 0.5859 | 0.4230 |
| balanced + **tuned threshold** | 0.5859 | **0.5491** |
| per-wafer adaptive weights | 0.5729 | 0.5407 |
| **focal loss (γ=2)** | **0.5880** | **0.5492** |
| SMOTE | 0.5607 | 0.5177 |

- **Threshold tuning is the biggest lever and it is free** — identical model, identical
  AUC-PR, F1 moves 0.4230 → 0.5491. Ranking was never the problem; the 0.5 cut point was.
- **Class weighting trades precision for recall; it adds no information** — it relabels
  the same ranking.
- **Per-wafer adaptive weighting did not help** — we expected it to. The wafer-prior
  features already hand the model the per-wafer rate, so re-encoding it in sample weights
  is redundant.
- **SMOTE hurt, as predicted.** With a median Cohen's d of 0.134, interpolating between
  failures lands synthetic points inside the pass distribution. You cannot resolve an
  overlap by manufacturing more points inside it.
- **Hyper-parameter tuning was worth +0.0026 — noise.** 40 Optuna trials on AUC-PR, run
  last and timeboxed. Feature and design choices moved the metric 6–10× more. Worth saying
  out loud: the win here was never in the hyper-parameters.

Show `figures/class_overlap.png` — even the six *most separable* of 500 parametrics
overlap almost completely.

---

## 7 · Deployment-grade extras (45s)

- **Calibration.** Reliability diagram + Brier before/after isotonic. Ranking well is not
  the same as being right about 0.30 meaning 30%.
- **Venn-ABERS intervals.** Distribution-free probability intervals per die — assumes only
  exchangeability, no Gaussian assumption. Relevant precisely because we measured these
  distributions and they are not Gaussian.
- **Yield-loss vs escape curve, with a cost slider.** The threshold is a business
  decision: a false accept becomes a field failure and an RMA, a false reject is pure
  yield loss. F1 implicitly assumes those cost the same. They never do.

---

## 8 · What we would do next (30s)

- Cross-lot generalisation — our splits are by wafer; the real test is by lot and by date.
- The wafer prior currently uses one round of stage-1 scores; iterating it, or modelling
  the rate hierarchically, should sharpen it.
- Feed real fab parametrics. Our strongest finding — that the wafer rate dominates the
  neighbourhood — is a property of this generator and must be re-measured on real silicon.

---

## Anticipated questions

**"Is the wafer prior leakage?"**
No. It uses only test-time inputs — every die's features on the wafer being scored, no
labels. It is transductive, not supervised, and training scores are cross-fitted by wafer
so the second stage never sees an in-sample stage-1 score.

**"Why is accuracy so high for everything?"**
Because 96% of eligible dies pass. That is exactly why we never select on it. See the
all-pass row.

**"Did you just fit the data generator?"**
We read the provided generator, as anyone could — it ships in the zip. It told us the
block readings are ordered and that the wafer rate is drawn per wafer, and we then
*measured* both claims on the data before acting. Every design decision is backed by an
ablation, not by the source.

**"Why did the spatial factory fail?"**
Because the neighbourhood signal on this data is genuinely weak once you condition on the
wafer, and the wafer-prior features already capture what spatial information there is. We
kept the factory because the root-cause classifier and spatial SHAP heatmap depend on its
features, and reported the cost.

**"Reproducible?"**
Seeds fixed everywhere; two consecutive runs produce identical metrics to 10 decimal
places. Every run ever made is in `results/results.csv`. One command regenerates every
figure and table.

**"Did you find any bugs in your own work?"** *(A good question to invite — the answer
is a credibility slide, not a liability.)*
Three, all found by measurement rather than by reading code:
1. **Focal loss scored 0.026 — below the 0.042 base rate.** The `(pt − y)` gradient form
   in most published snippets is correct only for the positive class and silently inverts
   on negatives. Once fixed it became the best rung in the ladder.
2. **Venn-ABERS intervals were degenerate.** Textbook-correct, but width shrinks as O(1/n),
   so on 36k calibration dies they collapsed to 0.001 wide with 10% band coverage. We
   measured them, rejected them, and shipped a bootstrap interval at 90% coverage.
3. **A train-matrix mismatch the hyper-parameter search exposed.** Optuna reported a best
   validation score *worse* than the defaults we had seeded as trial 0 — impossible unless
   the matrices differed. They did: the wafer-prior columns were being rebuilt in-sample
   for re-training consumers. Fixing it made the imbalance ladder reproduce Model B's
   0.5859 exactly, which it had not done before.
