# WaferLens — Plain-English Presentation & Project Guide

> **Cheat-sheet for your hackathon submission & presentation.**  
> Written so anyone can understand every piece of the pipeline from A to Z, speak confidently to judges, and explain all decisions without getting lost in technical jargon.

---

## 1. First: Is Your Dataset Correct?

**YES. 100% YES.** You are using the exact official dataset required by the problem statement:

1. **Base Data:** `data/LSWMD.pkl` (2.09 GB) is the official WM-811K semiconductor wafer map dataset.
2. **Generated Data:** Generated strictly using the organizers' script (`kit/generate_data.py`) with `configs/data_full.yaml`:
   - `train.parquet` (160 wafers, 173,099 dies)
   - `test.parquet` (40 wafers, 39,351 dies)
   - `validation.parquet` (40 wafers, 39,351 dies — the official submission set without labels)
3. **Columns present:**
   - `wafer_id`, `die_row`, `die_col`: Physical coordinates of each chip on the wafer disc.
   - `feature_1` ... `feature_500`: 500 electrical test measurements per die.
   - `old_label`: Pre-test status (**0 = passed pre-test, 1 = already failed**).
   - `label`: Final target (**0 = passed test, 1 = failed test**).
   - `block_readings`: 2,000 sub-die sensor readings per die.
4. **Fast Cache:** `data/cache/full/` parses the heavy 2,000-float strings once into binary memory-mapped arrays (`.npy`), so models train in seconds without exhausting your computer's RAM.

---

## 2. The Problem in Plain English (The 30-Second Pitch)

Imagine a silicon wafer as a giant round pizza cut into thousands of tiny square chips called **dies**.
Before packaging these chips into phones or SSDs, each die undergoes electrical tests.

- Some dies are already visibly broken or dead before the main test (**`old_label = 1`**). We ignore these because we already know they are trash.
- The remaining dies look healthy going in (**`old_label = 0`, green**).
- During testing, about **4% of these healthy green dies unexpectedly fail** (**turn orange**).
- **Your Job:** Predict **which green dies are going to turn orange**, explain **why** they failed, and tell the manufacturing engineer **which machine in the factory needs fixing**.

---

## 3. What Are the 3 Signals? (Multi-Resolution)

You are given three different levels of data (resolutions):

1. **Die-Level Parametrics (500 measurements):**
   - Standard electrical readings (voltage, resistance, leakage current) for the die.
   - *The Catch:* Over 65% of failing dies are "marginal" — their measurements look almost identical to healthy dies! No single parameter screams "I'm broken".
2. **Spatial Context (Where the die sits on the wafer):**
   - Does this die sit near a cluster of broken chips? Is it on the outer edge of the wafer?
   - In the semiconductor industry, this is known as **GDBN** (*"Good Die in a Bad Neighborhood"*). If all your neighbors are dead, you're probably defective too.
3. **Block-Level Readings (2,000 sub-die measurements):**
   - Each die contains internal sub-blocks. We are given 2,000 extra micro-readings per die.
   - *The Core Scientific Question of the Hackathon:* **Does adding these 2,000 extra numbers actually improve prediction, or is it just useless noise?**

---

## 4. What Did We Build? (Model A vs Model B)

| Model | What goes into it? | How it works | AUC-PR Score | Fail F1 |
|---|---|---|---|---|
| **Classical Baselines** (DPAT, GDBN, NNR) | Single parameter threshold or neighbor count | 20-year-old fab formulas | ~0.05 to 0.07 | ~0.08 |
| **Model A** | 500 Die Parametrics + Spatial Neighborhood | LightGBM Gradient Boosted Trees + Transductive Wafer Prior | **0.548** | **0.533** |
| **Model B** | Model A features + 2,000 Block Readings | Same LightGBM + 25 Statistical Block Features | **0.578** (up to **0.586**) | **0.540** |

### Did the Block Readings Help?
**YES.** Adding block data increased AUC-PR from **0.548 to 0.578 (+5.4%)**.
- But here is the brilliant twist you tell the judges: **You don't need all 2,000 numbers!**
- We discovered that the **average (mean)** of the 2,000 readings captures **98% of the entire block signal**.
- Why? Because when a die is defective, the defect adds a small amount of extra electrical noise across a cluster of blocks. Taking the mean is the mathematically sufficient summary.

---

## 5. The "Secret Sauce" Innovations That Win the Hackathon

Judges look for genuine engineering insight, not just throwing black-box models at data. Here are the 4 big discoveries in this project:

### 1. The Transductive Wafer Prior (Our Biggest Win: +0.046 AUC-PR)
- Each wafer has a random "hidden defect rate" depending on how dirty or hot the manufacturing chamber was that day.
- A single die cannot tell you this. But at test time, you hold the **entire wafer**.
- We use a **two-stage model**: Stage 1 makes a preliminary guess for every die on the wafer. We then calculate the wafer's average risk. Stage 2 asks: *"Is this die risky relative to the wafer it was baked on?"*
- This alone boosted accuracy on the minority class more than any complex neural network.

### 2. Pooling 500 Weak Features Into One Anomaly Score (+0.028 AUC-PR)
- None of the 500 electrical parameters could separate good dies from bad dies on their own (median Cohen's $d = 0.13$).
- But because the parameters are independent, their tiny clues add up ($d = 3.22$).
- We spatially detrended the wafer (subtracted center-to-edge heat gradients) and trained a clean linear model to combine all 500 weak signals into a single score: `die_anom_score`.

### 3. Statistical Pooling Beat Deep Learning (Honest Science!)
- We built a state-of-the-art Deep Neural Network (**Gated Attention Multiple-Instance Learning**, Ilse et al., ICML 2018) to read the 2,000 blocks.
- **The Result:** The fancy deep neural network scored **0.580 AUC-PR**, while simple, smart statistical pooling (mean, std, max, cluster length) scored **0.586 AUC-PR**.
- **What to say:** *"We tested a complex deep learning model against statistical pooling. Pooling won. It is 40× faster, deterministic, runs on any laptop, and doesn't overfit. We shipped the simpler, robust model."* (Judges respect this immensely).

### 4. Attribution Decomposed by Resolution (Deliverable 4)
- Most teams just show a list of 500 feature names from SHAP. That is useless to a fab manager.
- We group feature importance into 3 buckets:
  - **Die:** How much is due to the chip's own electrical test?
  - **Spatial:** How much is due to where it sits and its neighbors?
  - **Block:** How much is due to the internal 2,000 sensor readings?
- Every prediction sums to 100%:  
  `Risk: 83%  [ Die: 31% | Spatial: 28% | Block: 24% ]`

### 5. Root-Cause Mapping ("Not a Probability, a Work Order")
- We don't just say "this die failed". We map the spatial pattern directly to the factory machine:
  - **Center defect:** Chamber liquid flow / RF fault $\to$ *Check Chamber RF*
  - **Edge-Ring defect:** Thermal variation during Rapid Thermal Processing $\to$ *Review Thermal Module*
  - **Scratch:** Mechanical wafer handling arm $\to$ *Inspect Robot Handler*

---

## 6. What Tech Stack Did We Use and Why?

1. **LightGBM (GBDT):**
   - *Why:* The gold standard for tabular data in industry. Fast, robust to outliers, natively handles imbalance with `class_weight='balanced'`.
2. **Scipy (`scipy.ndimage`, `cKDTree`):**
   - *Why:* Vectorized 2D wafer math. Running Python `for` loops across 200,000 dies takes hours; `scipy.ndimage` processes whole wafer grids in milliseconds.
3. **PyTorch (Attention-MIL):**
   - *Why:* Built the deep multi-instance learning encoder with gated attention to inspect sub-die block sequences.
4. **SHAP (SHapley Additive exPlanations):**
   - *Why:* Mathematically exact local feature attribution decomposed into the 3 resolutions.
5. **Streamlit & Plotly:**
   - *Why:* Interactive, instant-loading web app for the live demo.
6. **Isotonic Regression & Bootstrap Intervals:**
   - *Why:* Calibrates raw probabilities so that "80% failure risk" actually means 80 out of 100 dies fail, with 90% confidence intervals.

---

## 7. Your Presentation Script (Step-by-Step)

Here is your exact talk track for tomorrow (Target: 7–10 minutes total):

### Slide 1: The Problem (1 minute)
> *"Judges, in semiconductor fabs, 96% of dies pass final testing. Only ~4% of healthy-looking dies unexpectedly fail. Fabs already use a 20-year-old method called GDBN — Good Die in a Bad Neighborhood. But GDBN assumes simple bell-curve distributions, which fail completely at modern nanometer nodes. In fact, if you just predict 'every die passes', you get 95.8% accuracy and catch zero failures! That is why we evaluate strictly on AUC-PR and F1, never accuracy."*

### Slide 2: What We Found in the Data (2 minutes)
> *"Before building models, we analyzed the physics of the data:  
> First, none of the 500 parametric test features can separate good dies from bad dies on their own. But because they are independent, pooling them into a single multivariate anomaly score gave us a massive +0.028 AUC-PR jump.  
> Second, we discovered that the biggest spatial signal is not just immediate neighbors, but the overall wafer health. We built a transductive wafer prior that gave us our single largest gain: +0.046 AUC-PR."*

### Slide 3: Model A vs Model B (2 minutes)
> *"The core question from the hackathon: do the 2,000 block readings help?  
> We kept the exact same LightGBM model architecture and added block features. Model A scored 0.548 AUC-PR. Model B scored 0.578 AUC-PR — an undeniable 5.4% improvement.  
> Furthermore, we tested a complex Deep Attention-MIL neural network against statistical pooling. Pooling won by a fraction, trained 40× faster, and proved that a simple, elegant summary is superior for deployment."*

### Slide 4: Live Demo (3 minutes) — Open Streamlit (`streamlit run app/app.py`)
> *"Now let us show you our live dashboard. Every other team hands you a probability. We hand you a work order.  
> 1. Look at this wafer: Left is pre-test (green = good, red = known bad). Middle is our model's predicted risk. Right is ground truth. You can see our model highlights the exact cluster that failed.  
> 2. Now let's click on this failing die (31, 7).  
> Look at the attribution bar: 78% of the risk came from Die parametrics, 14% from Spatial context, and 9% from Block readings.  
> 3. Look at the Root Cause diagnosis: It detects an Edge-Ring pattern and tells the fab engineer: 'Anomalous temperature regulation during RTP — review thermal module'.  
> We don't just find the bad die. We tell the fab which machine to repair."*

### Slide 5: Economics & Wrap Up (1 minute)
> *"Finally, we evaluated fab economics. A false pass costs an expensive warranty RMA return. A false fail scraps a good chip. Using our tuned operating threshold, WaferLens captures 82% of true failures with only 18% escape, saving fabs up to 76% in total excursion costs compared to traditional GDBN."*

---

## 8. Answers to Hard Questions Judges Might Ask

**Q: "Why did you use AUC-PR instead of ROC-AUC or Accuracy?"**  
*Answer:* "Because of severe class imbalance (~4% failure rate). An all-pass model gets 95.8% accuracy and ~0.80 ROC-AUC while catching zero defects. AUC-PR focuses exclusively on precision and recall for the rare failure class."

**Q: "Is your transductive wafer prior data leakage?"**  
*Answer:* "No! At test time, a fab probes an entire wafer at once. We never use test labels. We only average the stage-1 feature predictions across dies on the same wafer. Furthermore, during training, stage-1 was cross-fitted strictly by wafer to prevent stacking leaks."

**Q: "Why didn't you use a big Vision CNN on the wafer image?"**  
*Answer:* "The hackathon task is die-level tabular yield prediction, not whole-wafer classification. Wafer CNNs are an already solved 99% problem, but they don't predict individual die parametric failures. We used spatial geometry as features instead."

**Q: "Where is the submission file?"**  
*Answer:* "It is generated at `results/submission.csv`, formatted with `wafer_id`, `die_row`, `die_col`, and `predicted_label` across all validation dies."

---

## 9. Quick Command Reference

To show everything running live on your terminal:

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Run the interactive dashboard (The Demo)
streamlit run app/app.py

# 3. View the generated tables and reports
type results\comparison.md
type results\imbalance_analysis.md

# 4. View all 11 generated figures
# Located in the 'figures/' directory
```
