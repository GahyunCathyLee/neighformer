#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py — Stats filename, validation, and computation for normalization.

stats는 feature 조합(ego_mode, nb_kin_mode 등)마다 달라지므로
train 시점에 feature 설정이 결정된 후 계산합니다.

Public API
──────────
  make_stats_filename()      파일명 생성
  assert_stats_dims()        차원 검증
  compute_stats_if_needed()  없을 때만 자동 계산
  compute_stats()            실제 계산 및 저장
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Filename convention
# ──────────────────────────────────────────────────────────────────────────────

def make_stats_filename(
    *,
    ego_mode:     str,
    nb_kin_mode:  str,
    use_s_x:      bool = False,
    use_s_y:      bool = False,
    use_dim:      bool = False,
    use_I:        bool = False,
) -> str:
    """
    활성화된 ego/neighbor feature 조합으로 stats 파일명을 생성합니다.

    Examples
    --------
    >>> make_stats_filename(ego_mode="p", nb_kin_mode="pva", use_s_x=True)
    'ego_p__pva_sx.npz'
    >>> make_stats_filename(ego_mode="pva", nb_kin_mode="pv", use_s_x=True, use_s_y=True)
    'ego_pva__pv_sx_sy.npz'
    """
    nb_parts = [nb_kin_mode.lower()]
    if use_s_x:      nb_parts.append("sx")
    if use_s_y:      nb_parts.append("sy")
    if use_dim:      nb_parts.append("dim")
    if use_I:        nb_parts.append("I")
    return f"ego_{ego_mode.lower()}__{('_'.join(nb_parts))}.npz"


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def assert_stats_dims(
    stats: Dict[str, torch.Tensor],
    ego_dim: int,
    nb_dim: int,
    stats_path: Path,
) -> None:
    """
    stats의 ego/nb 차원이 dataset.ego_feature_dim / nb_feature_dim 과 일치하는지 검증합니다.
    불일치 시 RuntimeError.
    """
    actual_ego = int(stats["ego_mean"].numel())
    actual_nb  = int(stats["nb_mean"].numel())

    if actual_ego != ego_dim:
        raise RuntimeError(
            f"[STATS MISMATCH] ego_mean dim={actual_ego} ≠ ego_dim={ego_dim} "
            f"(stats={stats_path})"
        )
    if actual_nb != nb_dim:
        raise RuntimeError(
            f"[STATS MISMATCH] nb_mean dim={actual_nb} ≠ nb_dim={nb_dim} "
            f"(stats={stats_path})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Auto-compute
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats_if_needed(
    *,
    stats_path:   Path,
    data_dir:     Path,
    splits_dir:   Path,
    ego_mode:     str  = "pva",
    nb_kin_mode:  str  = "pva",
    use_s_x:      bool = False,
    use_s_y:      bool = False,
    use_dim:      bool = False,
    use_I:        bool = False,
) -> None:
    """stats_path 가 없을 때 compute_stats() 를 호출합니다. 이미 존재하면 skip."""
    if Path(stats_path).exists():
        return

    compute_stats(
        data_dir     = data_dir,
        splits_dir   = splits_dir,
        stats_path   = stats_path,
        ego_mode     = ego_mode,
        nb_kin_mode  = nb_kin_mode,
        use_s_x      = use_s_x,
        use_s_y      = use_s_y,
        use_dim      = use_dim,
        use_I        = use_I,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_stats(
    *,
    data_dir:   Path,
    splits_dir: Path,
    stats_path: Path,
    ego_mode:     str  = "pva",
    nb_kin_mode:  str  = "pva",
    use_s_x:      bool = False,
    use_s_y:      bool = False,
    use_dim:      bool = False,
    use_I:        bool = False,
) -> Dict[str, np.ndarray]:
    """
    train split 기준으로 ego/nb feature 의 mean/std 를 계산하고 저장합니다.

    mmap 배열을 직접 읽어 numpy로 계산합니다.
    DataLoader 없이 동작하므로 train.py / evaluate.py 어디서든 호출 가능합니다.

    Parameters
    ----------
    data_dir   : preprocess.py 가 생성한 mmap 디렉터리
    splits_dir : split.py 가 생성한 splits 디렉터리 (train_indices.npy 위치)
    stats_path : 저장 경로 (make_stats_filename() 으로 생성 권장)

    Returns
    -------
    {"ego_mean", "ego_std", "nb_mean", "nb_std"}  as np.ndarray
    """
    from src.dataset import (
        _EGO_MODE_SLICES,
        _NB_KIN_MODE_SLICES, _NB_KIN_MODES_NONCONTIGUOUS,
        _NB_IDX_SX, _NB_IDX_SY, _NB_IDX_DIM, _NB_IDX_I,
    )

    print(f"[STATS] Computing stats → {stats_path}")

    # ── train indices ─────────────────────────────────────────────────────────
    idx_path = Path(splits_dir) / "train_indices.npy"
    if not idx_path.exists():
        raise FileNotFoundError(f"train_indices.npy not found: {idx_path}")
    train_idx = np.load(idx_path)

    # ── mmap 로드 ─────────────────────────────────────────────────────────────
    x_ego = np.load(Path(data_dir) / "x_ego.npy", mmap_mode="r")  # (N, T, 6)
    x_nb  = np.load(Path(data_dir) / "x_nb.npy",  mmap_mode="r")  # (N, T, K, 10)

    # ── ego 슬라이싱 ──────────────────────────────────────────────────────────
    ego_sl   = _EGO_MODE_SLICES[ego_mode.lower()]
    ego_raw  = x_ego[train_idx]                       # (n, T, 6)
    ego_data = ego_raw[..., ego_sl]                   # (n, T, ego_dim)
    ego_flat = ego_data.reshape(-1, ego_data.shape[-1])  # (n*T, ego_dim)

    # ── nb feature 슬라이싱 ───────────────────────────────────────────────────
    nb_raw = x_nb[train_idx]                          # (n, T, K, 10)

    nb_parts = []
    nm = nb_kin_mode.lower()
    if nm != "none":
        if nm == "pa":
            nb_parts.append(nb_raw[..., 0:2])
            nb_parts.append(nb_raw[..., 4:6])
        elif nm in _NB_KIN_MODE_SLICES:
            nb_parts.append(nb_raw[..., _NB_KIN_MODE_SLICES[nm]])
        else:
            raise ValueError(f"Unknown nb_kin_mode: '{nb_kin_mode}'")

    if use_s_x:      nb_parts.append(nb_raw[..., _NB_IDX_SX   : _NB_IDX_SX   + 1])
    if use_s_y:      nb_parts.append(nb_raw[..., _NB_IDX_SY   : _NB_IDX_SY   + 1])
    if use_dim:      nb_parts.append(nb_raw[..., _NB_IDX_DIM  : _NB_IDX_DIM  + 1])
    if use_I:        nb_parts.append(nb_raw[..., _NB_IDX_I    : _NB_IDX_I    + 1])

    nb_data = np.concatenate(nb_parts, axis=-1)       # (n, T, K, nb_dim)
    nb_flat = nb_data.reshape(-1, nb_data.shape[-1])  # (n*T*K, nb_dim)

    # ── mean / std ────────────────────────────────────────────────────────────
    ego_mean = ego_flat.mean(axis=0).astype(np.float32)
    ego_std  = ego_flat.std(axis=0).astype(np.float32)
    nb_mean  = nb_flat.mean(axis=0).astype(np.float32)
    nb_std   = nb_flat.std(axis=0).astype(np.float32)

    # std=0 방지
    ego_std = np.where(ego_std < 1e-6, 1.0, ego_std)
    nb_std  = np.where(nb_std  < 1e-6, 1.0, nb_std)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, ego_mean=ego_mean, ego_std=ego_std,
             nb_mean=nb_mean, nb_std=nb_std)

    print(f"[STATS] Saved  ego_dim={ego_mean.shape[0]}  nb_dim={nb_mean.shape[0]}"
          f"  n_train={len(train_idx):,}")

    return {"ego_mean": ego_mean, "ego_std": ego_std,
            "nb_mean":  nb_mean,  "nb_std":  nb_std}
