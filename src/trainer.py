#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainer.py — Train / Evaluate loops for EncDecFormer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import ade, fde, multimodal_loss, rmse
from src.utils import _to_int, measure_latency_ms

# ──────────────────────────────────────────────────────────────────────────────
# Label constants  (scenario_label.py 기준)
# ──────────────────────────────────────────────────────────────────────────────
EVENT_LABELS: List[str] = ["cut_in", "lane_change", "lane_following", "unknown"]
STATE_LABELS: List[str] = ["dense", "free_flow", "unknown"]

# ──────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    use_amp:        bool  = True
    grad_clip_norm: float = 1.0
    w_ade:          float = 1.0
    w_fde:          float = 1.0
    w_rmse:         float = 0.0
    w_cls:          float = 0.1


@dataclass
class EvalConfig:
    use_amp:       bool  = True
    w_ade:         float = 1.0
    w_fde:         float = 1.0
    w_rmse:        float = 0.0
    w_cls:         float = 0.1
    data_hz:       float = 3.0
    # stratified evaluation
    labels_lut:      Optional[dict] = None
    save_event_path: Optional[Path] = None
    save_state_path: Optional[Path] = None
    # latency
    measure_latency: bool = False
    latency_iters:   int  = 200
    latency_warmup:  int  = 30


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _any_nonfinite(x: torch.Tensor) -> bool:
    return not torch.isfinite(x).all()


def _finite_diff(x: torch.Tensor, hz: float) -> torch.Tensor:
    """
    Finite-difference derivative along the time axis.

    Parameters
    ----------
    x   : (B, Tf, D)
    hz  : sampling frequency — scales dx to physical units

    Returns
    -------
    (B, Tf, D)  — first value duplicated to preserve shape
    """
    if x.shape[1] <= 1:
        return torch.zeros_like(x)
    dx = (x[:, 1:, :] - x[:, :-1, :]) * hz
    return torch.cat([dx[:, :1, :], dx], dim=1)


def _forward(
    model: nn.Module,
    x_ego: torch.Tensor,
    x_nb: torch.Tensor,
    nb_mask: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run model forward and unpack (traj, scores) or (traj, None)."""
    out = model(x_ego, x_nb, nb_mask)
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return out, None


def _resolve_pred_abs(
    pred: torch.Tensor,
    scores: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal output (B,M,Tf,2)에서 best mode를 선택합니다.

    Returns
    -------
    pred_abs : (B, Tf, 2)
    best_idx : (B,)
    """
    if scores is not None:
        best_idx = torch.argmax(scores, dim=1)
    else:
        best_idx = torch.zeros(pred.shape[0], device=pred.device, dtype=torch.long)

    pred_abs = pred[torch.arange(pred.shape[0], device=pred.device), best_idx]
    return pred_abs, best_idx


# ──────────────────────────────────────────────────────────────────────────────
# Train loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    optimizer:  torch.optim.Optimizer,
    scheduler:  Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler:     Optional[torch.cuda.amp.GradScaler],
    cfg:        TrainConfig,
    global_step: int = 0,
    epoch:       int = 0,
) -> Dict[str, float]:
    """
    One training epoch.

    Returns
    -------
    {"loss", "ade", "fde", "global_step_end"}
    """
    model.train()

    sum_loss = sum_ade = sum_fde = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Train", dynamic_ncols=True, leave=False)

    for it, batch in enumerate(pbar):
        x_ego      = batch["x_ego"].to(device, non_blocking=True)
        x_nb       = batch["x_nb"].to(device, non_blocking=True)
        nb_mask    = batch["nb_mask"].to(device, non_blocking=True)
        y_abs      = batch["y"].to(device, non_blocking=True)

        if _any_nonfinite(x_ego) or _any_nonfinite(x_nb) or _any_nonfinite(y_abs):
            raise RuntimeError(f"[BAD INPUT] ep={epoch} it={it}")

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=cfg.use_amp, dtype=torch.bfloat16):
            pred, scores = _forward(model, x_ego, x_nb, nb_mask)
            loss, best_idx = multimodal_loss(
                pred=pred, y_abs=y_abs, score_logits=scores,
                w_ade=cfg.w_ade, w_fde=cfg.w_fde,
                w_rmse=cfg.w_rmse, w_cls=cfg.w_cls,
            )

        if _any_nonfinite(loss):
            raise RuntimeError(f"[BAD LOSS]  ep={epoch} it={it} loss={loss.item():.6f}")

        if cfg.use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred_abs = pred[torch.arange(pred.shape[0], device=device), best_idx]
            a = ade(pred_abs, y_abs).mean()
            f = fde(pred_abs, y_abs).mean()

        sum_loss += loss.detach().item()
        sum_ade  += a.detach().item()
        sum_fde  += f.detach().item()
        n += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", ADE=f"{a.item():.3f}")

    return {
        "loss":            sum_loss / max(1, n),
        "ade":             sum_ade  / max(1, n),
        "fde":             sum_fde  / max(1, n),
        "global_step_end": global_step,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg:    EvalConfig,
    epoch:  Optional[int] = None,
) -> Dict[str, float]:
    """
    Full evaluation pass.

    Returns
    -------
    dict with keys:
      loss, ade, fde, rmse, vel, acc, jerk, n_samples,
      rmse_1s … rmse_5s,
      matched, matched_ratio  (only when cfg.labels_lut is not None)
    """
    if cfg.data_hz <= 0:
        raise ValueError(f"data_hz must be > 0, got {cfg.data_hz}")

    model.eval()
    hz = float(cfg.data_hz)

    # ── latency measurement (optional) ───────────────────────────────────────
    if cfg.measure_latency:
        _run_latency(model, loader, device, cfg)

    # ── accumulators ─────────────────────────────────────────────────────────
    sum_loss = sum_ade = sum_fde = sum_rmse = 0.0
    sum_vel  = sum_acc = sum_jerk = 0.0
    n_samples = 0

    eval_secs = [1, 2, 3, 4, 5]
    rmse_se    = {s: 0.0 for s in eval_secs}
    rmse_cnt   = {s: 0   for s in eval_secs}

    # stratified accumulators: label -> [sum_ade, sum_fde, sum_vel, sum_acc, sum_jerk, count]
    ev_stats: dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0])
    st_stats: dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0])
    n_matched = 0

    pbar = tqdm(loader, desc="Val", dynamic_ncols=True, leave=False)

    for batch in pbar:
        x_ego      = batch["x_ego"].to(device, non_blocking=True)
        x_nb       = batch["x_nb"].to(device, non_blocking=True)
        nb_mask    = batch["nb_mask"].to(device, non_blocking=True)
        y_abs      = batch["y"].to(device, non_blocking=True)
        y_vel      = batch["y_vel"].to(device, non_blocking=True)
        y_acc      = batch["y_acc"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=cfg.use_amp, dtype=torch.bfloat16):
            pred, scores = _forward(model, x_ego, x_nb, nb_mask)
            loss, _ = multimodal_loss(
                pred=pred, y_abs=y_abs, score_logits=scores,
                w_ade=cfg.w_ade, w_fde=cfg.w_fde,
                w_rmse=cfg.w_rmse, w_cls=cfg.w_cls,
            )
            pred_abs, _ = _resolve_pred_abs(pred, scores)

        B      = pred_abs.shape[0]
        Tf_cur = pred_abs.shape[1]

        ade_s  = ade(pred_abs, y_abs)
        fde_s  = fde(pred_abs, y_abs)
        rmse_s = rmse(pred_abs, y_abs)

        # kinematic errors via finite difference
        pred_vel  = _finite_diff(pred_abs, hz)
        pred_acc  = _finite_diff(pred_vel,  hz)
        pred_jerk = _finite_diff(pred_acc,  hz)
        gt_jerk   = _finite_diff(y_acc,     hz)

        vel_s  = torch.norm(pred_vel  - y_vel,  dim=-1).mean(dim=1)
        acc_s  = torch.norm(pred_acc  - y_acc,  dim=-1).mean(dim=1)
        jerk_s = torch.norm(pred_jerk - gt_jerk, dim=-1).mean(dim=1)

        # horizon-specific RMSE
        dist = torch.norm(pred_abs - y_abs, dim=-1)   # (B, Tf)

        for sec in eval_secs:
            idx = int(sec * hz) - 1
            if 0 <= idx < Tf_cur:
                rmse_se[sec]  += (dist[:, idx] ** 2).sum().item()
                rmse_cnt[sec] += B

        sum_loss += loss.item() * B
        sum_ade  += ade_s.sum().item()
        sum_fde  += fde_s.sum().item()
        sum_rmse += rmse_s.sum().item()
        sum_vel  += vel_s.sum().item()
        sum_acc  += acc_s.sum().item()
        sum_jerk += jerk_s.sum().item()
        n_samples += B

        # stratified stats
        if cfg.labels_lut is not None and "meta" in batch:
            ade_np  = ade_s.cpu().numpy()
            fde_np  = fde_s.cpu().numpy()
            vel_np  = vel_s.cpu().numpy()
            acc_np  = acc_s.cpu().numpy()
            jerk_np = jerk_s.cpu().numpy()

            for i, meta in enumerate(batch["meta"]):
                if i >= B:
                    break
                key = (
                    _to_int(meta.get("recordingId")),
                    _to_int(meta.get("trackId")),
                    _to_int(meta.get("t0_frame")),
                )
                lab = cfg.labels_lut.get(key)
                if lab is None:
                    continue
                n_matched += 1
                ev = lab.get("event_label") or "unknown"
                st = lab.get("state_label") or "unknown"

                for stats, lbl in ((ev_stats, ev), (st_stats, st)):
                    stats[lbl][0] += float(ade_np[i])
                    stats[lbl][1] += float(fde_np[i])
                    stats[lbl][2] += float(vel_np[i])
                    stats[lbl][3] += float(acc_np[i])
                    stats[lbl][4] += float(jerk_np[i])
                    stats[lbl][5] += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ADE=f"{ade_s.mean().item():.3f}",
            FDE=f"{fde_s.mean().item():.3f}",
        )

    # ── aggregate results ─────────────────────────────────────────────────────
    N = max(1, n_samples)
    results: Dict[str, float] = {
        "loss":      sum_loss / N,
        "ade":       sum_ade  / N,
        "fde":       sum_fde  / N,
        "rmse":      sum_rmse / N,
        "vel":       sum_vel  / N,
        "acc":       sum_acc  / N,
        "jerk":      sum_jerk / N,
        "n_samples": float(n_samples),
    }

    for sec in eval_secs:
        results[f"rmse_{sec}s"] = (
            math.sqrt(rmse_se[sec] / rmse_cnt[sec]) if rmse_cnt[sec] > 0 else float("nan")
        )

    if cfg.labels_lut is not None:
        results["matched"]       = float(n_matched)
        results["matched_ratio"] = float(n_matched) / N

    # ── stratified CSV export ─────────────────────────────────────────────────
    if cfg.labels_lut is not None and epoch is not None:
        _save_stratified_csv(
            ev_stats, EVENT_LABELS, cfg.save_event_path, epoch, n_matched, n_samples
        )
        _save_stratified_csv(
            st_stats, STATE_LABELS, cfg.save_state_path, epoch, n_matched, n_samples
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_latency(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg:    EvalConfig,
) -> None:
    """Measure and print per-batch inference latency."""
    first      = next(iter(loader))
    x_ego_l    = first["x_ego"].to(device, non_blocking=True)
    x_nb_l     = first["x_nb"].to(device, non_blocking=True)
    nb_mask_l  = first["nb_mask"].to(device, non_blocking=True)

    def _infer():
        with autocast(device_type=device.type, enabled=cfg.use_amp, dtype=torch.bfloat16):
            pred, scores = _forward(model, x_ego_l, x_nb_l, nb_mask_l)
            if pred.dim() == 4:
                _resolve_pred_abs(pred, scores)

    lat = measure_latency_ms(
        fn=_infer, device=device,
        iters=cfg.latency_iters, warmup=cfg.latency_warmup,
    )
    print(f"[Latency] avg={lat['avg_ms']:.2f}ms  p50={lat['p50_ms']:.2f}ms  "
          f"p90={lat['p90_ms']:.2f}ms  (batch_size={x_ego_l.shape[0]})")


def _save_stratified_csv(
    stats:    dict,
    labels:   List[str],
    path:     Optional[Path],
    epoch:    int,
    matched:  int,
    total:    int,
) -> None:
    """Append one epoch row of per-label ADE/FDE/vel/acc/jerk to a CSV."""
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, object] = {
        "epoch":         epoch,
        "matched":       matched,
        "total":         total,
        "matched_ratio": matched / max(1, total),
    }
    for lbl in labels:
        sa, sf, sv, sac, sj, c = stats.get(lbl, [0.0, 0.0, 0.0, 0.0, 0.0, 0])
        c = int(c)
        row[f"{lbl}_count"] = c
        row[f"{lbl}_ADE"]   = float(sa  / c) if c > 0 else float("nan")
        row[f"{lbl}_FDE"]   = float(sf  / c) if c > 0 else float("nan")
        row[f"{lbl}_vel"]   = float(sv  / c) if c > 0 else float("nan")
        row[f"{lbl}_acc"]   = float(sac / c) if c > 0 else float("nan")
        row[f"{lbl}_jerk"]  = float(sj  / c) if c > 0 else float("nan")

    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)