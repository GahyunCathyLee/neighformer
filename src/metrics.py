#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py — Loss functions and evaluation metrics for EncDecFormer.

모든 metric 함수는 절대 좌표(absolute positions) 기준이며, per-sample 값을 반환합니다.
배치 평균이 필요하면 .mean()을 호출하세요.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def multimodal_loss(
    pred: torch.Tensor,
    y_abs: torch.Tensor,
    score_logits: Optional[torch.Tensor],
    w_ade: float = 1.0,
    w_fde: float = 0.0,
    w_rmse: float = 0.0,
    w_cls: float = 0.1,
    **_,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Winner-takes-all multimodal trajectory loss.

    Parameters
    ----------
    pred         : (B, M, Tf, 2)  absolute predicted trajectories
    y_abs        : (B, Tf, 2)     ground-truth absolute positions
    score_logits : (B, M) or None  mode logits for classification loss

    Returns
    -------
    loss     : scalar tensor
    best_idx : (B,)  index of the best mode (min ADE)
    """
    B, M, Tf, _ = pred.shape

    err_dist = torch.norm(pred - y_abs[:, None, :, :], dim=-1)       # (B, M, Tf)
    best_idx  = torch.argmin(err_dist.mean(dim=-1), dim=1)            # (B,)
    best_dist = err_dist[torch.arange(B, device=pred.device), best_idx]  # (B, Tf)

    loss = pred.new_zeros(())
    if w_ade  > 0.0: loss = loss + w_ade  * best_dist.mean()
    if w_fde  > 0.0: loss = loss + w_fde  * best_dist[:, -1].mean()
    if w_rmse > 0.0: loss = loss + w_rmse * torch.sqrt(best_dist.pow(2).mean() + 1e-6)
    if w_cls  > 0.0 and score_logits is not None:
        loss = loss + w_cls * F.cross_entropy(score_logits, best_idx)

    return loss, best_idx


# ──────────────────────────────────────────────────────────────────────────────
# Metrics  (per-sample, call .mean() for batch average)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ade(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """Average Displacement Error — (B, Tf, 2) → (B,)"""
    return torch.norm(pred_abs - y_abs, dim=-1).mean(dim=-1)


@torch.no_grad()
def fde(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """Final Displacement Error — (B, Tf, 2) → (B,)"""
    return torch.norm(pred_abs[:, -1, :] - y_abs[:, -1, :], dim=-1)


@torch.no_grad()
def rmse(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """Root Mean Square Error — (B, Tf, 2) → (B,)"""
    return torch.norm(pred_abs - y_abs, dim=-1).pow(2).mean(dim=-1).sqrt()