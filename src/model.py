"""Causal Transformer + multi-task learning (MTL) model for joint fault
classification and tool-wear regression, with the pooling strategy exposed
as a hyperparameter (Design Principle 2 — last-token pooling vs. GAP)."""

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CausalTransformerMTL(nn.Module):
    """Shared causal (or bidirectional) Transformer encoder with two task
    heads: multi-label fault classification and tool-wear regression.

    pooling: 'gap' (global average pooling) or 'last' (last-token pooling).
    Under a causal mask, position i can only attend to i+1 <= L earlier
    positions, so early-window positions carry far less context than the
    final position. GAP averages these unevenly-informed representations
    together and dilutes the pooled vector; last-token pooling instead
    reads only the final position, which — under a causal mask — has
    always seen the entire window. This is Design Principle 2.
    """

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
                 dropout=0.3, num_fault_labels=5, max_len=50,
                 use_causal_mask=True, pooling="last"):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fault_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_fault_labels))
        self.wear_head = nn.Linear(d_model, 1)

    def forward(self, x):
        seq_len = x.size(1)
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        if self.use_causal_mask:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            h = self.encoder(h, mask=causal_mask, is_causal=True)
        else:
            h = self.encoder(h)

        pooled = h.mean(dim=1) if self.pooling == "gap" else h[:, -1, :]
        return self.fault_head(pooled), self.wear_head(pooled)


def init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FocalLoss(nn.Module):
    """FL(p_t) = -(1-p_t)^gamma * log(p_t), for severe class imbalance
    (fault base rates of 0.2%-1.5% in AI4I 2020)."""

    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce
        return focal.mean() if self.reduction == "mean" else focal.sum()


class CausalTransformerMTLDynamic(CausalTransformerMTL):
    """Adds Kendall et al. (2018) homoscedastic uncertainty weighting:
    learnable log-variances replace the fixed alpha/beta joint-loss
    weights, so the model learns how much to trust each task itself."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var_fault = nn.Parameter(torch.zeros(1))
        self.log_var_wear = nn.Parameter(torch.zeros(1))

    def dynamic_loss(self, loss_fault, loss_wear):
        precision_fault = torch.exp(-self.log_var_fault)
        precision_wear = torch.exp(-self.log_var_wear)
        total = (precision_fault * loss_fault + self.log_var_fault
                 + precision_wear * loss_wear + self.log_var_wear)
        return total.squeeze()
