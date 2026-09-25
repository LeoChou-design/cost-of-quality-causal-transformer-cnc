# Data: AI4I 2020 Predictive Maintenance Dataset

10,000 synthetic CNC machine records, in original time (UDI) order, with
five documented failure modes.

| File | Content | Size |
|---|---|---|
| `ai4i2020.csv` | Raw dataset as released by the UCI Machine Learning Repository | 10,000 rows × 14 columns |

## Sequence modeling setup

- **Input features (8-dim)**: 5 continuous columns (air temperature, process
  temperature, rotational speed, torque, tool wear) + 3 one-hot columns for
  machine type (L/M/H).
- **Targets**: two fault labels used in this comparison (`TWF`, `OSF` — the
  two with enough positive samples for stable AUROC) + tool wear (continuous).
- **Window alignment**: a sliding window of length 50 covers `[t-49 ... t]`;
  fault labels and tool wear are predicted for `t+1`, never for the window's
  own last step, so a model cannot copy tool wear directly from its input
  (tool wear is both a feature and part of the physical definition of
  TWF/OSF). See `src/data_windows.py::make_sliding_windows`.
- **Tabular summary for tree models**: each window is also summarized into
  a 28-dim feature vector (mean/SD/min/max/last-value per continuous
  channel, plus the one-hot machine type) for the Random Forest / Gradient
  Boosting baselines — see `src/data_windows.py::window_to_tabular`.

## Source

- Dataset page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Citation: Stephan, M., & Matzka, S. (2020). AI4I 2020 Predictive
  Maintenance Dataset. UCI Machine Learning Repository.
  https://doi.org/10.1109/AI4I49448.2020.00023 (companion paper)
- License: CC BY 4.0 (UCI Machine Learning Repository standard license).
