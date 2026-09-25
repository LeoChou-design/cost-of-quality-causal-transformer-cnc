"""Fair-comparison experiment: the causal Transformer (final configuration —
causal mask + last-token pooling) vs. Random Forest and Gradient Boosting,
under an identical blocked time-series CV protocol, identical folds, and
identical seeds for all three model families.

Cost-of-quality framing: a missed fault is a failure cost, a false alarm is
an appraisal cost (Feigenbaum, 1956; Schiffauerova & Thomson, 2006). A
single mean AUROC/R2 says nothing about how consistently that performance
holds — and a volatile signal is itself a cost, because it forces wider
safety margins and more manual inspection. This script reports both the
mean AND the cross-fold standard deviation for every metric, and treats the
SD as a first-class result, not an afterthought.

Fairness notes:
- All three model families see the exact same blocked-CV fold assignment
  and the exact same seed list (an earlier iteration of this experiment
  asymmetrically gave the tree models more seeds than the Transformer,
  which the multi-seed protocol here is specifically designed to rule out
  as a confound).
- Random Forest / Gradient Boosting cannot consume a (L, D) window tensor
  directly, so each window is summarized into a fixed-length tabular
  feature vector using SPC-control-chart-style statistics (mean/std/min/
  max/last-value per continuous channel) rather than a naive flatten —
  see data_windows.window_to_tabular.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from data_windows import build_fold_windows_raw, load_sequence_frame, make_blocks, window_to_tabular
from model import CausalTransformerMTL, init_weights_xavier
from train_eval import evaluate_model, paired_test, train_model
from data_windows import build_loaders

FAULT_COLS_OF_INTEREST = ["TWF", "OSF"]  # the two fault modes with enough positives for stable AUROC
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def build_model_builders(seed):
    return {
        "RandomForest": {
            "clf": lambda: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
            "reg": lambda: RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        },
        "GradientBoosting": {
            "clf": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed),
            "reg": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed),
        },
    }


def run_tree_baselines(fold, blocks, n_blocks, feature_cols, seed):
    X_train_w, yf_train, yw_train, X_test_w, yf_test, yw_test = build_fold_windows_raw(
        fold, blocks, n_blocks, feature_cols
    )
    F_train, F_test = window_to_tabular(X_train_w), window_to_tabular(X_test_w)
    fault_cols_all = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    rows = []
    for model_name, builders in build_model_builders(seed).items():
        aurocs = {}
        for col in FAULT_COLS_OF_INTEREST:
            idx = fault_cols_all.index(col)
            y_tr, y_te = yf_train[:, idx].astype(int), yf_test[:, idx].astype(int)
            if len(np.unique(y_tr)) < 2 or not (0 < y_te.sum() < len(y_te)):
                aurocs[col] = None
                continue
            sw = compute_sample_weight("balanced", y_tr)
            clf = builders["clf"]()
            clf.fit(F_train, y_tr, sample_weight=sw)
            aurocs[col] = roc_auc_score(y_te, clf.predict_proba(F_test)[:, 1])

        reg = builders["reg"]()
        reg.fit(F_train, yw_train)
        wear_r2 = r2_score(yw_test, reg.predict(F_test))

        rows.append({"fold": fold, "seed": seed, "config": model_name,
                     "auroc_TWF": aurocs.get("TWF"), "auroc_OSF": aurocs.get("OSF"), "wear_r2": wear_r2})
    return rows


def run_transformer(fold, blocks, n_blocks, feature_cols, seed, device, max_epochs, patience):
    X_train_w, yf_train, yw_train, X_test_w, yf_test, yw_test = build_fold_windows_raw(
        fold, blocks, n_blocks, feature_cols
    )
    # Reuse the raw (unstandardized) split, but the Transformer needs
    # standardized continuous features and a val split for early stopping —
    # carve 10% of the training windows out as validation.
    n_val = max(1, int(0.1 * len(X_train_w)))
    X_tr, X_va = X_train_w[:-n_val], X_train_w[-n_val:]
    yf_tr, yf_va = yf_train[:-n_val], yf_train[-n_val:]
    yw_tr, yw_va = yw_train[:-n_val], yw_train[-n_val:]

    n_continuous = 5
    cont_mean = X_tr[:, :, :n_continuous].reshape(-1, n_continuous).mean(axis=0)
    cont_std = X_tr[:, :, :n_continuous].reshape(-1, n_continuous).std(axis=0)

    def standardize(X):
        X = X.copy()
        X[:, :, :n_continuous] = (X[:, :, :n_continuous] - cont_mean) / cont_std
        return X

    w_mean, w_std = yw_tr.mean(), yw_tr.std()
    train_loader = build_loaders(standardize(X_tr), yf_tr, yw_tr, w_mean, w_std, shuffle=True)
    val_loader = build_loaders(standardize(X_va), yf_va, yw_va, w_mean, w_std, shuffle=False)
    test_loader = build_loaders(standardize(X_test_w), yf_test, yw_test, w_mean, w_std, shuffle=False)

    model = CausalTransformerMTL(input_dim=len(feature_cols), use_causal_mask=True, pooling="last")
    model.apply(init_weights_xavier)
    model.to(device)
    model = train_model(model, train_loader, val_loader, device, alpha=0.7, beta=0.3,
                         seed=seed, max_epochs=max_epochs, patience=patience)
    res = evaluate_model(model, test_loader, yf_test, yw_test, w_mean, w_std, device)
    return {"fold": fold, "seed": seed, "config": "CausalTransformer (final config)",
            "auroc_TWF": res["auroc"]["TWF"], "auroc_OSF": res["auroc"]["OSF"], "wear_r2": res["wear"]["r2"]}


def main(n_blocks, seeds, max_epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_seq, feature_cols = load_sequence_frame()
    blocks = make_blocks(df_seq, n_blocks)

    rows = []
    for fold in range(n_blocks):
        print(f"--- Fold {fold + 1}/{n_blocks} ---")
        for seed in seeds:
            rows.extend(run_tree_baselines(fold, blocks, n_blocks, feature_cols, seed))
            rows.append(run_transformer(fold, blocks, n_blocks, feature_cols, seed, device, max_epochs, patience))
        print(f"  fold {fold + 1} done (3 model families x {len(seeds)} seeds)")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "vs_baselines_raw.csv"), index=False)

    # Average across seeds within each fold first (matches the significance
    # protocol in the companion architecture-design repo).
    fold_avg = df.groupby(["fold", "config"])[["auroc_TWF", "auroc_OSF", "wear_r2"]].mean().reset_index()
    fold_avg.to_csv(os.path.join(RESULTS_DIR, "vs_baselines_fold_avg.csv"), index=False)

    print(f"\n=== {n_blocks}-fold blocked CV x {len(seeds)} seeds/fold: mean +/- std (cost-of-quality view) ===")
    summary = fold_avg.groupby("config")[["auroc_TWF", "auroc_OSF", "wear_r2"]].agg(["mean", "std"])
    print(summary)
    summary.to_csv(os.path.join(RESULTS_DIR, "vs_baselines_summary.csv"))

    sig = {}
    transformer_cfg = "CausalTransformer (final config)"
    for metric in ["auroc_TWF", "auroc_OSF", "wear_r2"]:
        for baseline_cfg in ["RandomForest", "GradientBoosting"]:
            ma, mb, t, p = paired_test(fold_avg, baseline_cfg, transformer_cfg, metric)
            key = f"{metric}: {baseline_cfg} vs {transformer_cfg}"
            sig[key] = {baseline_cfg: ma, transformer_cfg: mb, "t": t, "p": p}
            print(f"{key}: {baseline_cfg}={ma:.4f}, Transformer={mb:.4f}, t={t:.3f}, p={p:.4f}")

    with open(os.path.join(RESULTS_DIR, "vs_baselines_significance.json"), "w") as f:
        json.dump(sig, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    main(args.folds, args.seeds, args.max_epochs, args.patience)
