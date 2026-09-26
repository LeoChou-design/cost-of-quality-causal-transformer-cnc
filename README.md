<a id="zh"></a>

**中文** | [English](#english)

# 類別不平衡下的品質成本：因果 Transformer 與傳統機器學習於 CNC 刀具故障偵測的比較

本專案為一篇研討會延伸摘要的公平比較程式，以「品質成本（cost of quality）」的角度重新看待模型基準比較：不只看哪個模型平均分數較高，更看哪個模型能給品質管理一個更**可預測**的訊號。

## 一、論文與研討會資料

| 項目 | 內容 |
|---|---|
| 論文題目 | Cost of Quality under Class Imbalance: Comparing a Causal Transformer to Classical Machine Learning for CNC Tool-Fault Detection（類別不平衡下的品質成本：因果 Transformer 與傳統機器學習於 CNC 刀具故障偵測的比較，論文以英文撰寫） |
| 作者 | 周理陽¹*（Li Yang Chou） |
| 單位 | ¹國立中央大學機械工程學系 |
| 關鍵字 | 品質成本、預測性維護、因果 Transformer、類別不平衡、模型基準比較 |

| 研討會 | 內容 |
|---|---|
| 全名 | 中華民國品質學會第62屆年會暨2026國際品質管理研討會（Chinese Society for Quality 62nd Annual Meeting & 2026 International Symposium on Quality Management, ISQM 2026） |
| 主辦 | 中華民國品質學會（CSQ）、元智大學工業工程與管理學系 |
| 日期 | 2026 年 11 月 7 日 |
| 地點 | 元智大學有庠廳（桃園市） |
| 官方網站 | https://sites.google.com/view/isqm2026 |
| 狀態 | 已錄取 |

### 論文摘要（中文為譯文，原文見下方英文部分）

預測性維護的決策可轉化為品質成本的取捨：漏報故障是失效成本，誤報則是鑑定成本。在嚴重類別不平衡下，若只報告單一鑑別力數字，這個取捨很容易被掩蓋。本延伸摘要以 AI4I 2020 資料集，比較具多任務學習的因果遮罩 Transformer 與隨機森林、梯度提升於 CNC 刀具故障偵測的表現。為避免自我參照洩漏，Transformer 的窗口對齊到預測下一步（t+1）、自最末時間點池化，並以 Focal Loss 進行故障分類、以 MSE 進行磨損迴歸的聯合訓練；三個模型家族皆在相同的區塊交叉驗證流程下評估。Transformer 在每項指標上的平均 AUROC 與 R² 都高於兩個基準。對品質管理更具意義的是，Transformer 的磨損 R² 在各折之間遠比兩個傳統模型穩定，代表它提供的是更可預測、變異成本更低的訊號。

## 二、資料

UCI Machine Learning Repository 的 **AI4I 2020 預測性維護資料集**：10,000 筆合成 CNC 機台紀錄；本研究聚焦於正樣本足以穩定計算 AUROC 的兩種故障模式（TWF、OSF），外加刀具磨損迴歸。

細節與引用方式見 [`data/README_data.md`](data/README_data.md)。

## 三、方法

### 為什麼「公平比較」本身也需要設計

因果 Transformer 吃的是 `(50, 8)` 的窗口張量；隨機森林與梯度提升需要固定長度的特徵向量。若兩個家族使用不同的資料切分、不同的種子數量、或不同的評估程式，效能差距就可能只是比較方式造成的假象，而非架構本身的差異。本 repo 除了模型家族之外一律固定：相同的區塊 CV 折、每折相同的種子清單、相同的保留測試窗口。

- **因果 Transformer**（`src/model.py`）：2 層、4 個注意力頭的因果遮罩編碼器，`d_model=64`，**只從最末時間點池化**（在因果遮罩下唯一總是看過整個窗口的位置），以 Focal Loss（γ=2，分類）與 MSE（迴歸）聯合訓練，α=0.7／β=0.3。
- **隨機森林／梯度提升**（`src/run_vs_baselines.py`）：以每個窗口的 28 維表格摘要訓練——每個連續通道的平均、標準差、最小、最大與最末值，加上機型——刻意仿照 SPC 管制圖統計量（X-bar、全距），而非隨意攤平，並以類別平衡權重處理嚴重的類別不平衡。
- **窗口化**：三個模型家族使用相同的 t+1 對齊窗口（此設計背後的完整原則研究見姊妹 repo [causal-transformer-cnc-design-principles](https://github.com/LeoChou-design/causal-transformer-cnc-design-principles)）。
- **評估**：時間區塊 CV、每折多個種子，先在折內對種子取平均，再以成對 t 檢定比較各模型家族。

### 品質成本的觀點

統計品質管制把檢驗視為「漏檢缺陷的成本（失效成本）」與「誤報的成本（鑑定成本）」之間的平衡——即 Feigenbaum（1956）的經典分類。單一的平均 AUROC 或 R² 並不能說明該表現在不同條件下是否穩定，而不穩定的訊號本身就是成本來源：它迫使工廠拉大安全邊際、增加人工複查。因此本 repo 把**跨折標準差當作第一級結果**來報告，而不只是平均值。

## 四、執行結果

**以下為本 repo 程式碼實際執行所產出的數字**（8 折時間區塊交叉驗證，每折 3 個種子，折內平均後再做 t 檢定）。論文本身的表格採用每折 8 個種子；依專案範圍，本 repo 不需重現論文的確切數字，只需正確地端到端執行——下列質性結論與論文全程一致。

| 指標 | 隨機森林 | 梯度提升 | **因果 Transformer** | p（對 RF） | p（對 GB） |
|---|---|---|---|---|---|
| TWF AUROC | 0.912 ± 0.046 | 0.955 ± 0.005 | **0.959 ± 0.016** | 0.0082 | 0.51（不顯著） |
| OSF AUROC | 0.877 ± 0.049 | 0.898 ± 0.035 | **0.932 ± 0.013** | 0.0139 | 0.0222 |
| 磨損 R² | 0.855 ± 0.039 | 0.850 ± 0.035 | **0.872 ± 0.004** | 0.27（不顯著） | 0.11（不顯著） |

Transformer 在每項指標都有最高的平均值，TWF AUROC 對隨機森林、OSF AUROC 對兩個基準達到顯著；其餘比較方向上偏向 Transformer 但不顯著——如實完整列出，而非選擇性報告。

**更具意義的結果是變異。** Transformer 的磨損迴歸 R² 跨折標準差為 **0.004**，隨機森林為 **0.039**（大 9.6 倍），梯度提升為 **0.035**（大 8.6 倍）——與論文報告的差距同一量級（Transformer 0.008 vs. RF 0.038、GB 0.079）。品質訊號在不同條件下起伏不定的模型，會迫使工廠拉大安全邊際與增加人工驗證——這些成本被單一平均數字完全掩蓋。

圖表：`figures/fig1_mean_performance.png`（平均 ± SD 長條圖）、`figures/fig2_cost_of_variability.png`（單獨呈現變異結果）、`figures/fig3_per_fold_boxplot.png`（完整的逐折分布）。

## 五、檔案結構

```
cost-of-quality-causal-transformer-cnc/
├─ data/
│  ├─ ai4i2020.csv                AI4I 2020 原始資料
│  └─ README_data.md
├─ src/
│  ├─ data_windows.py             t+1 對齊窗口、時間區塊 CV 折、表格摘要
│  ├─ model.py                    CausalTransformerMTL（因果 + 最末時間點池化）
│  ├─ train_eval.py               共用訓練／評估迴圈 + 成對 t 檢定
│  ├─ run_vs_baselines.py         Transformer vs. 隨機森林 vs. 梯度提升，公平 CV 流程
│  └─ make_figures.py
├─ results/                       原始與折內平均結果、顯著性檢定
├─ figures/                       fig1–fig3
├─ references/
└─ requirements.txt
```

## 六、如何執行

```bash
pip install -r requirements.txt

python src/run_vs_baselines.py --folds 8 --seeds 42 123 2024
python src/make_figures.py
```

`--max-epochs`／`--patience` 控制 Transformer 的訓練預算（預設 60／8）。若有 CUDA GPU 會自動使用；上述完整比較在一張一般消費級 GPU 上約需 8 分鐘（隨機森林／梯度提升只用 CPU 且很快，Transformer 是主要成本）。

## 七、參考文獻

見 [`references/README.md`](references/README.md)。第三方論文全文不隨本 repo 散布。

## 八、授權

本專案自行撰寫的程式碼（`src/`）與文件以 MIT License 釋出，詳見 [`LICENSE`](LICENSE)。

以下內容不在本授權範圍內，各自沿用原本的條款：

- **資料集**（`data/`）：UCI Machine Learning Repository 的 AI4I 2020 預測性維護資料集，CC BY 4.0，引用 Stephan & Matzka (2020)，見 `data/README_data.md`。
- **參考文獻**（`references/`）：著作權歸各作者與出版方所有，見 `references/README.md`。

## 九、致謝

誠摯感謝指導教授國立臺灣科技大學王鵬凱老師在研究設計與論文品質上的悉心指導與嚴格審查。

## 十、AI 使用揭露

所有研究設計、方法與結論皆由本人獨立主導。AI 工具作為輔助，用於英文文法潤飾、對本人撰寫之程式進行除錯與重構、將實驗筆記本整理為可執行腳本，以及撰寫與翻譯 repo 文件。本人已逐行驗證所有代碼、結果與文稿，對研究真實性負完全責任。

---

<a id="english"></a>

[中文](#zh) | **English**

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

## 9. Acknowledgments

Sincere thanks to my advisor, Prof. Peng-Kai Wang of National Taiwan University of Science and Technology, for careful guidance and rigorous review of the research design and paper quality. (English translation of the Chinese text above.)

## 10. AI Use Disclosure

All research design, methods, and conclusions were led and completed independently by the author. AI tools were used as an aid for English grammar polishing, debugging and refactoring of code written by the author, organizing experiment notebooks into runnable scripts, and drafting and translating the documentation in this repository. The author has verified all code, results, and manuscripts line by line and takes full responsibility for the authenticity of the research. (English translation of the Chinese text above.)
