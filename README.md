# Cost of Quality under Class Imbalance: Comparing a Causal Transformer to Classical Machine Learning for CNC Tool-Fault Detection

Fair-comparison code for a conference extended abstract that reframes model benchmarking through a cost-of-quality lens: not just which model scores higher on average, but which one gives quality management a more *predictable* signal.

## 1. Paper & Conference

| Item | Detail |
|---|---|
| Title | Cost of Quality under Class Imbalance: Comparing a Causal Transformer to Classical Machine Learning for CNC Tool-Fault Detection |
| Author | Li Yang Chou¹* (周理陽) |
| Affiliation | ¹Department of Mechanical Engineering, National Central University |
| Keywords | Cost of Quality, Predictive Maintenance, Causal Transformer, Class Imbalance, Model Benchmarking |

| Conference | Detail |
|---|---|
| Full name | 中華民國品質學會第62屆年會暨2026國際品質管理研討會 — Chinese Society for Quality (CSQ) 62nd Annual Meeting & 2026 International Symposium on Quality Management (ISQM 2026) |
| Organizers | Chinese Society for Quality (CSQ), Department of Industrial Engineering and Management, Yuan Ze University |
| Date | 7 November 2026 |
| Venue | You Yang Hall, Yuan Ze University, Taoyuan, Taiwan |
| Official website | https://sites.google.com/view/isqm2026 |
| Status | Accepted |

### Abstract

Predictive maintenance decisions translate into a cost-of-quality trade-off: a missed fault is a failure cost, and a false alarm is an appraisal cost. Under severe class imbalance, this trade-off is easily hidden by reporting only a discriminative-power number. This extended abstract compares a causal-masked Transformer with multi-task learning against Random Forest and Gradient Boosting for CNC tool-fault detection on the AI4I 2020 dataset. To avoid self-referential leakage, the Transformer's windows are aligned to predict one step ahead (t+1), pooled from the last time step, and trained jointly for fault classification (Focal Loss) and wear regression (MSE); all three model families are evaluated under an identical blocked cross-validation protocol. The Transformer attains higher mean AUROC and R² than both baselines on every metric. More consequential for quality management, the Transformer's wear R² is far more stable across folds than either classical model, indicating a more predictable, lower-cost-of-variability signal.

## 2. Data

UCI Machine Learning Repository — **AI4I 2020 Predictive Maintenance Dataset**: 10,000 synthetic CNC machine records; this study focuses on the two fault modes with enough positive samples for stable AUROC (TWF, OSF) plus tool-wear regression.

Details and citation: [`data/README_data.md`](data/README_data.md).

## 3. Method

### Why "fair comparison" needs its own design

The causal Transformer consumes a `(50, 8)` window tensor; Random Forest and Gradient Boosting need a fixed-length feature vector. If the two families used different data splits, different seed counts, or different evaluation code, any performance gap could just be an artifact of the comparison itself rather than the architecture. This repo holds everything except model family fixed: identical blocked-CV folds, identical seed list per fold, identical held-out test windows.

- **Causal Transformer** (`src/model.py`): 2-layer, 4-head causal-masked encoder, `d_model=64`, pooling from the **last time step only** (the one position that always sees the full window under a causal mask), trained jointly with Focal Loss (γ=2, classification) and MSE (regression), α=0.7/β=0.3.
- **Random Forest / Gradient Boosting** (`src/run_vs_baselines.py`): trained on a 28-dim tabular summary of each window — mean, SD, min, max, and last value per continuous channel, plus machine type — deliberately mirroring SPC control-chart statistics (X-bar, range) rather than a naive flatten, with `class_weight='balanced'` sampling for the severe class imbalance.
- **Windowing**: identical t+1-aligned windows for all three model families (see the companion repository [causal-transformer-cnc-design-principles](https://github.com/LeoChou-design/causal-transformer-cnc-design-principles) for the full design-principle study behind this choice).
- **Evaluation**: blocked time-series CV, multiple seeds per fold, fold-level results averaged across seeds before a paired t-test compares model families.

### Cost-of-quality framing

Statistical quality control treats inspection as a balance between the cost of missing a defect (failure cost) and the cost of a false alarm (appraisal cost) — Feigenbaum's (1956) classic categories. A single mean AUROC or R² says nothing about how consistently that performance holds across conditions, and an unstable signal is itself a cost driver: it forces wider safety margins and more manual verification. This repo therefore reports the **cross-fold standard deviation as a first-class result**, not just the mean.

## 4. Results

**These are numbers this repository's code actually produced when run** (8-fold blocked CV, 3 seeds per fold — averaged per fold before the t-test). The paper's own table uses 8 seeds per fold; per the project scope, this repo does not need to reproduce the paper's exact published numbers, only to run correctly end-to-end — the qualitative conclusions below match the paper throughout.

| Metric | Random Forest | Gradient Boosting | **Causal Transformer** | p (vs. RF) | p (vs. GB) |
|---|---|---|---|---|---|
| TWF AUROC | 0.912 ± 0.046 | 0.955 ± 0.005 | **0.959 ± 0.016** | 0.0082 | 0.51 (not significant) |
| OSF AUROC | 0.877 ± 0.049 | 0.898 ± 0.035 | **0.932 ± 0.013** | 0.0139 | 0.0222 |
| Wear R² | 0.855 ± 0.039 | 0.850 ± 0.035 | **0.872 ± 0.004** | 0.27 (not significant) | 0.11 (not significant) |

The Transformer has the highest mean on every metric, reaching significance for TWF AUROC vs. Random Forest and OSF AUROC vs. both baselines; the remaining comparisons are directionally in the Transformer's favor but not significant — reported in full rather than selectively.

**The more consequential result is variance.** The Transformer's wear-regression R² has a cross-fold SD of **0.004**, versus **0.039** for Random Forest (9.6× larger) and **0.035** for Gradient Boosting (8.6× larger) — a gap of the same order the paper reports (Transformer 0.008 vs. RF 0.038, GB 0.079). A model whose quality signal swings unpredictably across conditions forces wider safety margins and more manual verification, costs that a single average number hides entirely.

Figures: `figures/fig1_mean_performance.png` (mean ± SD bars), `figures/fig2_cost_of_variability.png` (the variance result on its own), `figures/fig3_per_fold_boxplot.png` (full per-fold distributions).

## 5. File Structure

```
cost-of-quality-causal-transformer-cnc/
├─ data/
│  ├─ ai4i2020.csv                Raw AI4I 2020 dataset
│  └─ README_data.md
├─ src/
│  ├─ data_windows.py             t+1-aligned windowing, blocked CV folds, tabular summary
│  ├─ model.py                    CausalTransformerMTL (causal + last-token pooling)
│  ├─ train_eval.py               Shared train/eval loop + paired t-test
│  ├─ run_vs_baselines.py         Transformer vs. RandomForest vs. GradientBoosting, fair CV protocol
│  └─ make_figures.py
├─ results/                       Raw + fold-averaged results, significance tests
├─ figures/                       fig1–fig3
├─ references/
└─ requirements.txt
```

## 6. How to Run

```bash
pip install -r requirements.txt

python src/run_vs_baselines.py --folds 8 --seeds 42 123 2024
python src/make_figures.py
```

`--max-epochs` / `--patience` control the Transformer's training budget (default 60/8). A CUDA GPU is used automatically if available; the full comparison above takes about 8 minutes on a single consumer GPU (Random Forest / Gradient Boosting training is CPU-only and fast; the Transformer is the dominant cost).

## 7. References

See [`references/README.md`](references/README.md). Full-text PDFs of third-party papers are not redistributed in this repository.

## 8. License

Code and documentation authored for this project (`src/`, this README) are released under the MIT License — see [`LICENSE`](LICENSE).

The following are **not** covered by that license and remain under their own terms:

- **Dataset** (`data/`): UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset, CC BY 4.0 — cite Stephan & Matzka (2020), see `data/README_data.md`.
- **References** (`references/`): copyright of the original authors/publishers — see `references/README.md`.
