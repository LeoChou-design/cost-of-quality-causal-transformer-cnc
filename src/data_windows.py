"""AI4I 2020 sequence data pipeline: t+1-aligned sliding windows (Design
Principle 1), blocked time-series CV fold construction, and a tabular
window summary for the Random Forest / Gradient Boosting baselines.

Window-alignment principle: tool wear is both an input feature and part of
the physical definition of two failure modes (TWF, OSF). If the prediction
target were read from the window's own last time step t, the model could
simply copy the answer from its input. Instead, every window covers
[t-L+1 ... t] and the label is read from t+1 — a time step that never
appears inside the window — which structurally prevents this leakage
regardless of model architecture.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")

FAULT_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
CONTINUOUS_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
WINDOW_LENGTH = 50
N_CONTINUOUS = len(CONTINUOUS_COLS)


def load_sequence_frame():
    """Load ai4i2020.csv and assemble the columns needed for sequence
    modeling: continuous features, one-hot Type, fault labels, and a raw
    (unstandardized) copy of tool wear used only as the t+1 regression
    target."""
    df = pd.read_csv(CSV_PATH)
    assert (df["UDI"].values == np.arange(1, len(df) + 1)).all(), \
        "Rows are not in original UDI (time) order — required for sequence windowing."

    type_onehot = pd.get_dummies(df["Type"], prefix="Type").astype(float)
    feature_cols = CONTINUOUS_COLS + list(type_onehot.columns)

    df_seq = pd.concat([
        df[CONTINUOUS_COLS], type_onehot, df[FAULT_COLS],
        df[["Tool wear [min]"]].rename(columns={"Tool wear [min]": "Tool_wear_raw"}),
    ], axis=1)
    return df_seq, feature_cols


def make_sliding_windows(df_part, feature_cols, fault_cols=FAULT_COLS, window_length=WINDOW_LENGTH):
    """Windows cover [t-L+1 ... t]; labels are read from t+1 (Design
    Principle 1 — window-alignment). The last possible window is dropped
    because it has no t+1 label."""
    features = df_part[feature_cols].to_numpy(dtype=np.float32)
    faults = df_part[fault_cols].to_numpy(dtype=np.float32)
    wear = df_part["Tool_wear_raw"].to_numpy(dtype=np.float32)

    n_rows = len(df_part)
    if n_rows - window_length <= 0:
        raise ValueError(f"Not enough rows ({n_rows}) for window_length={window_length} plus a t+1 label")

    windows_incl_last = np.lib.stride_tricks.sliding_window_view(features, window_shape=window_length, axis=0)
    windows_incl_last = windows_incl_last.transpose(0, 2, 1)  # (n_rows-L+1, L, n_features)
    X_windows = windows_incl_last[:-1]  # drop the window with no t+1 label

    y_fault = faults[window_length:]
    y_wear = wear[window_length:]
    return X_windows, y_fault, y_wear


def single_split(df_seq, train_frac=0.8, val_frac=0.1):
    """Time-ordered 80/10/10 split with train-only Z-score standardization
    of the continuous columns."""
    n_total = len(df_seq)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    df_train = df_seq.iloc[:n_train].reset_index(drop=True)
    df_val = df_seq.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_seq.iloc[n_train + n_val:].reset_index(drop=True)

    mean, std = df_train[CONTINUOUS_COLS].mean(), df_train[CONTINUOUS_COLS].std()

    def normalize(block):
        block = block.copy()
        block[CONTINUOUS_COLS] = (block[CONTINUOUS_COLS] - mean) / std
        return block

    return normalize(df_train), normalize(df_val), normalize(df_test)


def build_loaders(X, yf, yw, wear_mean, wear_std, batch_size=64, shuffle=False):
    yw_norm = (yw - wear_mean) / wear_std
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(yf, dtype=torch.float32),
        torch.tensor(yw_norm, dtype=torch.float32).reshape(-1, 1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_blocks(df_seq, n_blocks):
    block_size = len(df_seq) // n_blocks
    return [df_seq.iloc[i * block_size:(i + 1) * block_size].reset_index(drop=True) for i in range(n_blocks)]


def build_fold_data(fold_idx, blocks, n_blocks, feature_cols, batch_size=64, window_length=WINDOW_LENGTH):
    """Blocked time-series CV: block `fold_idx` is the test set, the next
    block (wrapping around) is validation, and the rest is training. Each
    fold is standardized and windowed independently using only its own
    training-block statistics (windows never cross block boundaries)."""
    test_idx = fold_idx
    val_idx = (fold_idx + 1) % n_blocks
    train_indices = [i for i in range(n_blocks) if i not in (test_idx, val_idx)]

    train_raw = pd.concat([blocks[i] for i in train_indices], axis=0)
    fold_mean, fold_std = train_raw[CONTINUOUS_COLS].mean(), train_raw[CONTINUOUS_COLS].std()

    def normalize(block):
        block = block.copy()
        block[CONTINUOUS_COLS] = (block[CONTINUOUS_COLS] - fold_mean) / fold_std
        return block

    def window_block(block):
        return make_sliding_windows(normalize(block), feature_cols, window_length=window_length)

    X_parts, yf_parts, yw_parts = [], [], []
    for i in train_indices:
        Xw, yfw, yww = window_block(blocks[i])
        X_parts.append(Xw); yf_parts.append(yfw); yw_parts.append(yww)
    X_train, yf_train, yw_train = np.concatenate(X_parts), np.concatenate(yf_parts), np.concatenate(yw_parts)

    X_val, yf_val, yw_val = window_block(blocks[val_idx])
    X_test, yf_test, yw_test = window_block(blocks[test_idx])

    w_mean, w_std = yw_train.mean(), yw_train.std()
    return {
        "train_loader": build_loaders(X_train, yf_train, yw_train, w_mean, w_std, batch_size, shuffle=True),
        "val_loader": build_loaders(X_val, yf_val, yw_val, w_mean, w_std, batch_size, shuffle=False),
        "test_loader": build_loaders(X_test, yf_test, yw_test, w_mean, w_std, batch_size, shuffle=False),
        "yf_test": yf_test, "yw_test": yw_test, "w_mean": w_mean, "w_std": w_std,
    }


def build_fold_windows_raw(fold_idx, blocks, n_blocks, feature_cols, window_length=WINDOW_LENGTH):
    """Same test-block assignment as build_fold_data, but returns raw numpy
    arrays (no standardization, validation folded into training) for the
    tree-based baselines, which don't need early stopping or feature scaling."""
    test_idx = fold_idx
    train_indices = [i for i in range(n_blocks) if i != test_idx]

    X_parts, yf_parts, yw_parts = [], [], []
    for i in train_indices:
        Xw, yfw, yww = make_sliding_windows(blocks[i], feature_cols, window_length=window_length)
        X_parts.append(Xw); yf_parts.append(yfw); yw_parts.append(yww)
    X_train, yf_train, yw_train = np.concatenate(X_parts), np.concatenate(yf_parts), np.concatenate(yw_parts)

    X_test, yf_test, yw_test = make_sliding_windows(blocks[test_idx], feature_cols, window_length=window_length)
    return X_train, yf_train, yw_train, X_test, yf_test, yw_test


def window_to_tabular(X_windows, n_continuous=N_CONTINUOUS):
    """Summarize each (L, D) window into a fixed-length tabular feature
    vector for RandomForest/GradientBoosting: mean/std/min/max/last-value
    per continuous channel (SPC control-chart style statistics) plus the
    (time-invariant) one-hot Type columns."""
    cont = X_windows[:, :, :n_continuous]
    type_last = X_windows[:, -1, n_continuous:]
    return np.concatenate([
        cont.mean(axis=1), cont.std(axis=1), cont.min(axis=1), cont.max(axis=1), cont[:, -1, :],
        type_last,
    ], axis=1)
