"""Shared training loop, evaluation, and paired significance testing used by
every experiment script in this repository."""

import numpy as np
import torch
import torch.optim as optim
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from model import FocalLoss

FAULT_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

criterion_fault = FocalLoss(gamma=2.0)
criterion_wear = torch.nn.MSELoss()


def train_model(model, train_loader, val_loader, device, alpha=0.7, beta=0.3,
                 use_dynamic_weighting=False, seed=42, max_epochs=60, patience=8, verbose=False):
    torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    def run_epoch(loader, is_train):
        model.train() if is_train else model.eval()
        total_loss, n = 0.0, 0
        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for X, y_fault, y_wear in loader:
                X, y_fault, y_wear = X.to(device), y_fault.to(device), y_wear.to(device)
                bs = X.size(0)
                if is_train:
                    optimizer.zero_grad()
                fault_logits, wear_pred = model(X)
                loss_fault = criterion_fault(fault_logits, y_fault)
                loss_wear = criterion_wear(wear_pred, y_wear)
                loss = model.dynamic_loss(loss_fault, loss_wear) if use_dynamic_weighting \
                    else alpha * loss_fault + beta * loss_wear
                if is_train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * bs
                n += bs
        return total_loss / n

    best_val, no_improve, best_state = float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)
        scheduler.step()
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"    epoch {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f}")
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_loader, y_fault_true, y_wear_true, wear_mean, wear_std,
                    device, eval_fault=True, eval_wear=True):
    model.eval()
    all_fault_logits, all_wear_preds = [], []
    with torch.no_grad():
        for X, y_fault, y_wear in test_loader:
            X = X.to(device)
            fl, wp = model(X)
            all_fault_logits.append(fl.cpu())
            all_wear_preds.append(wp.cpu())
    fault_logits = torch.cat(all_fault_logits).numpy()
    wear_pred_norm = torch.cat(all_wear_preds).numpy().flatten()
    fault_prob = 1 / (1 + np.exp(-fault_logits))

    result = {}
    if eval_fault:
        aurocs = {}
        for i, col in enumerate(FAULT_COLS):
            yt = y_fault_true[:, i]
            aurocs[col] = roc_auc_score(yt, fault_prob[:, i]) if 0 < yt.sum() < len(yt) else None
        result["auroc"] = aurocs
    if eval_wear:
        wear_pred = wear_pred_norm * wear_std + wear_mean
        result["wear"] = {
            "rmse": float(np.sqrt(mean_squared_error(y_wear_true, wear_pred))),
            "mae": float(mean_absolute_error(y_wear_true, wear_pred)),
            "r2": float(r2_score(y_wear_true, wear_pred)),
        }
    return result


def paired_test(df, config_a, config_b, metric, group_col="config", value_col=None, fold_col="fold"):
    value_col = value_col or metric
    a = df[df[group_col] == config_a].sort_values(fold_col)[value_col].values
    b = df[df[group_col] == config_b].sort_values(fold_col)[value_col].values
    t_stat, p_val = stats.ttest_rel(b, a)
    return a.mean(), b.mean(), t_stat, p_val
