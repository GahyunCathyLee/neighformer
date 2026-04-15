#!/usr/bin/env python3
"""
slot_weight_probe.py — slot별 가중치를 학습하고 시각화하는 standalone 스크립트.

기존 모델(EncDecFormer)은 건드리지 않음.
- SlotWeightNet + 작은 MLP predictor를 trajectory prediction loss로 end-to-end 학습.
- 학습 후 val set에서 slot별 learned weight를 시각화.

--by_lane_level 플래그를 켜면 ego의 lane-level 위치(leftmost/middle/rightmost)별로
별도 모델을 학습하고 3-panel 비교 시각화를 생성합니다.
  - meta_lane_level.npy가 필요합니다 (etc/analyze_lane_level.py 먼저 실행).
  - highD 전용 (exiD는 복잡한 lanelet map 구조로 미지원).

Usage:
    python slot_weight_probe.py --data highD
    python slot_weight_probe.py --data exiD
    python slot_weight_probe.py --data both
    python slot_weight_probe.py --data highD --by_lane_level
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

K       = 8
T       = 6
Tf      = 15
EGO_DIM = 6
NB_DIM  = 13
NB_KIN  = 4   # dx, dy, dvx, dvy (idx 0-3)

SLOT_NAMES = [
    "preceding",      # 0  앞-동일차선
    "following",      # 1  뒤-동일차선
    "leftPreceding",  # 2  앞-왼쪽차선
    "leftAlongside",  # 3  옆-왼쪽차선
    "leftFollowing",  # 4  뒤-왼쪽차선
    "rightPreceding", # 5  앞-오른쪽차선
    "rightAlongside", # 6  옆-오른쪽차선
    "rightFollowing", # 7  뒤-오른쪽차선
]

LANE_LEVEL_NAMES = {0: "leftmost (fast)", 1: "middle", 2: "rightmost (slow)"}


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SlotProbeDataset(Dataset):
    def __init__(
        self,
        mmap_dir: Path,
        indices: np.ndarray,
        ego_mean: np.ndarray,
        ego_std: np.ndarray,
        nb_mean: np.ndarray,
        nb_std: np.ndarray,
    ):
        self.x_ego   = np.load(mmap_dir / "x_ego.npy",   mmap_mode="r")
        self.x_nb    = np.load(mmap_dir / "x_nb.npy",    mmap_mode="r")
        self.nb_mask = np.load(mmap_dir / "nb_mask.npy", mmap_mode="r")
        self.y       = np.load(mmap_dir / "y.npy",       mmap_mode="r")
        self.idx     = indices
        self.ego_mean = ego_mean   # (EGO_DIM,)
        self.ego_std  = ego_std
        self.nb_mean  = nb_mean    # (NB_DIM,)
        self.nb_std   = nb_std

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        n = self.idx[i]
        ego  = (self.x_ego[n].astype(np.float32) - self.ego_mean) / self.ego_std   # (T, EGO_DIM)
        nb   = (self.x_nb[n].astype(np.float32)  - self.nb_mean)  / self.nb_std    # (T, K, NB_DIM)
        mask = self.nb_mask[n].copy()                                                # (T, K)  bool  (mmap은 read-only → copy 필요)
        y    = self.y[n].astype(np.float32)                                          # (Tf, 2)
        return ego, nb, mask, y


def _require_meta(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 etc/analyze_lane_level.py를 실행하세요."
        )
    return np.load(path, mmap_mode="r")


def filter_by_lane_level(
    mmap_dir: Path, indices: np.ndarray, lane_level: int,
) -> np.ndarray:
    all_levels = _require_meta(mmap_dir / "meta_lane_level.npy")
    return indices[np.isin(indices, np.where(all_levels == lane_level)[0])]


def filter_by_lc(
    mmap_dir: Path,
    indices: np.ndarray,
    lc_type: int,
    lc_phase: int,
) -> np.ndarray:
    """
    lc_type  (0-7) 과 lc_phase (-1/0/1/2) 조건 둘 다 만족하는 indices 반환.
    lc_type=-1 이면 type 필터 없음.  lc_phase=-99 이면 phase 필터 없음.
    """
    all_lct = _require_meta(mmap_dir / "meta_lc_type.npy")
    all_lcp = _require_meta(mmap_dir / "meta_lc_phase.npy")

    mask = np.ones(len(indices), bool)
    if lc_type >= 0:
        mask &= np.isin(indices, np.where(all_lct == lc_type)[0])
    if lc_phase != -99:
        mask &= np.isin(indices, np.where(all_lcp == lc_phase)[0])
    return indices[mask]


def filter_by_lc_types(
    mmap_dir: Path,
    indices: np.ndarray,
    lc_types: list[int],
    lc_phase: int,
) -> np.ndarray:
    """
    lc_types 리스트 중 하나라도 해당(OR)하고, lc_phase 조건을 만족하는 indices 반환.
    lc_phase=-99 이면 phase 필터 없음.
    """
    all_lct = _require_meta(mmap_dir / "meta_lc_type.npy")
    all_lcp = _require_meta(mmap_dir / "meta_lc_phase.npy")

    type_mask = np.isin(indices, np.where(np.isin(all_lct, lc_types))[0])
    phase_mask = np.ones(len(indices), bool)
    if lc_phase != -99:
        phase_mask = np.isin(indices, np.where(all_lcp == lc_phase)[0])
    return indices[type_mask & phase_mask]


LC_TYPE_NAMES: dict[int, str] = {
    0: "leftmost→middle",
    1: "leftmost→rightmost",
    2: "middle→leftmost",
    3: "middle→rightmost",
    4: "rightmost→leftmost",
    5: "rightmost→middle",
    6: "middle→middle (right)",
    7: "middle→middle (left)",
}
LC_PHASE_NAMES: dict[int, str] = {0: "pre-LC", 1: "during-LC", 2: "post-LC"}

# 4-group directional LC grouping
LC_GROUPS: dict[int, list[int]] = {
    0: [0, 1],    # leftmost → right
    1: [3, 6],    # middle   → right
    2: [2, 7],    # middle   → left
    3: [4, 5],    # rightmost → left
}
LC_GROUP_NAMES: dict[int, str] = {
    0: "leftmost → right (→middle / →rightmost)",
    1: "middle → right (→rightmost / →middle-right)",
    2: "middle → left (→leftmost / →middle-left)",
    3: "rightmost → left (→leftmost / →middle)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Normalization stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(
    mmap_dir: Path,
    train_idx: np.ndarray,
    max_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Training data의 subsample로 mean/std 계산."""
    sub     = train_idx[:max_samples]
    x_ego   = np.load(mmap_dir / "x_ego.npy",   mmap_mode="r")[sub]   # (N, T, 6)
    x_nb    = np.load(mmap_dir / "x_nb.npy",    mmap_mode="r")[sub]   # (N, T, K, 13)
    nb_mask = np.load(mmap_dir / "nb_mask.npy", mmap_mode="r")[sub]   # (N, T, K)

    ego_mean = x_ego.reshape(-1, EGO_DIM).mean(axis=0).astype(np.float32)
    ego_std  = x_ego.reshape(-1, EGO_DIM).std(axis=0).clip(1e-6).astype(np.float32)

    # nb: neighbor가 존재하는 위치만 사용
    valid_nb = x_nb[nb_mask]  # (M, NB_DIM)
    nb_mean  = valid_nb.mean(axis=0).astype(np.float32)
    nb_std   = valid_nb.std(axis=0).clip(1e-6).astype(np.float32)

    return ego_mean, ego_std, nb_mean, nb_std


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Transformer 기반 시계열 인코더 (슬롯·ego 공용).

    - 입력의 존재 여부를 padding_mask로 처리.
    - 완전히 absent한 시퀀스는 출력이 0 벡터.
    - 존재하는 타임스텝만 masked mean pool로 집약.

    Input:
      x            (N, T, input_dim)
      padding_mask (N, T) bool   — True = absent (padding)
    Output:
      (N, d_enc)
    """

    def __init__(
        self,
        input_dim: int,
        d_enc: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_enc)
        self.pos_emb    = nn.Embedding(T, d_enc)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_enc, nhead=n_heads,
            dim_feedforward=d_enc * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        _, T_len, _ = x.shape
        pos = torch.arange(T_len, device=x.device)
        h   = self.input_proj(x) + self.pos_emb(pos)              # (N, T, d_enc)

        # all-masked row → NaN 방지: 첫 타임스텝만 임시로 un-mask
        fully_absent         = padding_mask.all(dim=1)             # (N,)
        safe_mask            = padding_mask.clone()
        safe_mask[fully_absent, 0] = False

        h = self.transformer(h, src_key_padding_mask=safe_mask)    # (N, T, d_enc)

        # masked mean pool (존재하는 타임스텝만)
        present   = ~safe_mask                                     # (N, T)
        n_present = present.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        h_mean    = (h * present.unsqueeze(-1)).sum(dim=1) / n_present  # (N, d_enc)

        # fully absent 슬롯은 0 벡터
        h_mean = h_mean * (~fully_absent).float().unsqueeze(-1)    # (N, d_enc)
        return h_mean


class SlotWeightNet(nn.Module):
    """
    Temporal encoding → per-slot weight logits.

    Input:
      slot_enc (B, K, d_enc)  — per-slot temporal encodings
      ego_enc  (B, d_enc)     — ego temporal encoding
    Output:
      logits   (B, K)
    """

    def __init__(self, K: int, d_enc: int, d_hidden: int = 64):
        super().__init__()
        self.K        = K
        self.slot_pos = nn.Embedding(K, d_enc)    # slot 위치/역할 인코딩
        self.mlp      = nn.Sequential(
            nn.Linear(d_enc * 3, d_hidden),       # slot_enc ⊕ ego_enc ⊕ slot_pos
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, slot_enc: torch.Tensor, ego_enc: torch.Tensor) -> torch.Tensor:
        B        = slot_enc.shape[0]
        slot_idx = torch.arange(self.K, device=slot_enc.device)
        slot_pos = self.slot_pos(slot_idx).unsqueeze(0).expand(B, -1, -1)  # (B, K, d_enc)
        ego_exp  = ego_enc.unsqueeze(1).expand(-1, self.K, -1)             # (B, K, d_enc)
        feat     = torch.cat([slot_enc, ego_exp, slot_pos], dim=-1)        # (B, K, 3*d_enc)
        return self.mlp(feat).squeeze(-1)                                   # (B, K)


class SlotWeightProbe(nn.Module):
    """
    Temporal encoding + per-slot trajectory prediction probe.

    설계 원칙
    ─────────
    1. time averaging 없음: 각 슬롯·ego의 T-step 시계열을 Transformer로 인코딩.
       → 급감속, 차선 변경 시점 등 temporal dynamics를 온전히 반영.

    2. prediction-space weighted sum:
         slot_preds (B, K, Tf, 2) — 각 슬롯이 독립적으로 ego trajectory 예측
         pred = Σ_k  w_k * slot_pred_k
       → w_k gradient = ∂loss/∂pred × slot_pred_k
         "k번째 슬롯의 예측이 실제 trajectory와 얼마나 일치하는가"를 직접 반영.

    2-stage 학습
    ────────────
    Stage 1 (slot_encoder + ego_encoder + slot_emb + slot_predictor):
      uniform weights 아래서 각 슬롯이 독립적으로 ego trajectory를 학습.
      slot_predictor가 weight distribution 변화를 보정하는 능력을 차단.

    Stage 2 (weight_net만):
      고정된 slot_predictor를 기준으로 weight_net이
      "어느 슬롯의 예측을 믿을지"를 학습. gradient가 명확해짐.
    """

    def __init__(
        self,
        K: int       = 8,
        nb_dim: int  = NB_DIM,
        ego_dim: int = EGO_DIM,
        d_enc: int   = 64,
        d_hidden: int = 64,
        n_heads: int  = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        Tf: int = 15,
    ):
        super().__init__()
        self.K  = K
        self.Tf = Tf

        # ── temporal encoders ──
        self.slot_encoder = TemporalEncoder(nb_dim,  d_enc, n_heads, n_layers, dropout)
        self.ego_encoder  = TemporalEncoder(ego_dim, d_enc, n_heads, n_layers, dropout)

        # ── slot weight generator ──
        self.weight_net = SlotWeightNet(K, d_enc, d_hidden)

        # ── per-slot predictor (Stage 1 학습 대상) ──
        # 입력: slot_enc (d_enc) ⊕ ego_enc (d_enc) ⊕ slot_emb (d_enc)
        self.slot_emb       = nn.Embedding(K, d_enc)
        self.slot_predictor = nn.Sequential(
            nn.Linear(d_enc * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, Tf * 2),
        )

    # ── shared helpers ────────────────────────────────────────────────────────

    def _encode(
        self,
        x_nb: torch.Tensor,
        x_ego: torch.Tensor,
        nb_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        시계열 전체를 Transformer로 인코딩.

        Returns
        -------
        slot_enc (B, K, d_enc)
        ego_enc  (B, d_enc)
        nb_exist (B, K) bool  — 슬롯이 history 중 한 번이라도 존재했는지
        """
        B, T_len, K, _ = x_nb.shape

        # slot: (B, T, K, NB_DIM) → (B*K, T, NB_DIM)
        x_slot     = x_nb.permute(0, 2, 1, 3).reshape(B * K, T_len, -1)
        slot_pmask = ~nb_mask.permute(0, 2, 1).reshape(B * K, T_len)   # True=absent
        slot_enc   = self.slot_encoder(x_slot, slot_pmask).view(B, K, -1)  # (B, K, d_enc)

        # ego: always fully present
        ego_pmask = torch.zeros(B, T_len, dtype=torch.bool, device=x_ego.device)
        ego_enc   = self.ego_encoder(x_ego, ego_pmask)                      # (B, d_enc)

        nb_exist = nb_mask.any(dim=1)                                        # (B, K)
        return slot_enc, ego_enc, nb_exist

    def _slot_preds(
        self,
        slot_enc: torch.Tensor,
        ego_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        slot_enc : (B, K, d_enc)
        ego_enc  : (B, d_enc)
        Returns  : (B, K, Tf, 2)
        """
        B        = slot_enc.shape[0]
        slot_idx = torch.arange(self.K, device=slot_enc.device)
        slot_pos = self.slot_emb(slot_idx).unsqueeze(0).expand(B, -1, -1)  # (B, K, d_enc)
        ego_exp  = ego_enc.unsqueeze(1).expand(-1, self.K, -1)             # (B, K, d_enc)
        feat     = torch.cat([slot_enc, ego_exp, slot_pos], dim=-1)        # (B, K, 3*d_enc)
        return self.slot_predictor(feat).view(B, self.K, self.Tf, 2)       # (B, K, Tf, 2)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_nb: torch.Tensor,
        x_ego: torch.Tensor,
        nb_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_nb    : (B, T, K, NB_DIM)
        x_ego   : (B, T, EGO_DIM)
        nb_mask : (B, T, K) bool

        Returns
        -------
        pred    : (B, Tf, 2)
        weights : (B, K)   — absent slots은 0
        """
        slot_enc, ego_enc, nb_exist = self._encode(x_nb, x_ego, nb_mask)

        logits  = self.weight_net(slot_enc, ego_enc)
        logits  = logits.masked_fill(~nb_exist, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)                       # (B, K)

        slot_preds = self._slot_preds(slot_enc, ego_enc)                   # (B, K, Tf, 2)
        pred = (weights[:, :, None, None] * slot_preds).sum(dim=1)        # (B, Tf, 2)
        return pred, weights

    def forward_stage1(
        self,
        x_nb: torch.Tensor,
        x_ego: torch.Tensor,
        nb_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1 전용: per-slot direct supervision.

        각 슬롯이 자신의 temporal encoding만으로 ego trajectory를 독립 예측하도록
        ground truth를 K개 슬롯 각각에 직접 적용.

        → slot_pred_k들이 서로 다른 예측을 내도록 강제
          (uniform-weighted MSE에서는 모든 슬롯이 marginal mean trajectory로 수렴하는 문제 해소)
        → Stage 2에서 weight_net이 실제로 "어느 슬롯이 더 맞는가"를 선택할 수 있게 됨

        Returns
        -------
        slot_preds : (B, K, Tf, 2)  — 슬롯별 ego trajectory 예측
        nb_exist   : (B, K) bool    — loss masking용 (absent slot 제외)
        """
        slot_enc, ego_enc, nb_exist = self._encode(x_nb, x_ego, nb_mask)
        slot_preds = self._slot_preds(slot_enc, ego_enc)   # (B, K, Tf, 2)
        return slot_preds, nb_exist

    def forward_uniform(
        self,
        x_nb: torch.Tensor,
        x_ego: torch.Tensor,
        nb_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """균등 가중치 weighted sum — Stage 1 val 모니터링용."""
        slot_enc, ego_enc, nb_exist = self._encode(x_nb, x_ego, nb_mask)

        nb_exist_f = nb_exist.float()
        weights    = nb_exist_f / nb_exist_f.sum(dim=-1, keepdim=True).clamp(min=1.0)

        slot_preds = self._slot_preds(slot_enc, ego_enc)
        pred = (weights[:, :, None, None] * slot_preds).sum(dim=1)
        return pred, weights


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stats(weights: np.ndarray, occupancy: np.ndarray) -> tuple:
    """weights/occupancy → (means, stds, occs, uniform)."""
    means, stds, occs = [], [], []
    for k in range(K):
        active = weights[occupancy[:, k], k]
        means.append(active.mean() if len(active) > 0 else 0.0)
        stds.append(active.std()   if len(active) > 0 else 0.0)
        occs.append(occupancy[:, k].mean() * 100)
    mean_occupied = occupancy.sum(axis=1).mean()
    uniform = 1.0 / mean_occupied if mean_occupied > 0 else 1.0 / K
    return means, stds, occs, uniform, mean_occupied


def visualize(weights: np.ndarray, occupancy: np.ndarray, tag: str, out_path: str) -> None:
    """
    weights   : (N_val, K)  — softmax weights (occupied slots의 합=1)
    occupancy : (N_val, K)  — bool, slot이 존재했는지
    """
    means, stds, occs, uniform, mean_occupied = _compute_stats(weights, occupancy)

    x = np.arange(K)
    fig, axes = plt.subplots(2, 1, figsize=(22, 16))

    # ── Weight bar ──
    ax = axes[0]
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.85, label="mean ± std")
    ax.set_xticks(x)
    ax.set_xticklabels(SLOT_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Learned weight (softmax)")
    ax.set_title(f"Per-slot learned weights — {tag}", fontsize=13)
    ax.legend()

    # ── Occupancy bar ──
    ax2 = axes[1]
    ax2.bar(x, occs, color="coral", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SLOT_NAMES, rotation=30, ha="right", fontsize=10)
    ax2.set_ylabel("Slot occupancy (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Slot occupancy rate (val set)", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved → {out_path}")

    # ── Console table ──
    print(f"\nUniform baseline: 1 / {mean_occupied:.1f} = {uniform:.4f}")
    print(f"\n{'Slot':<20} {'weight_mean':>12} {'weight_std':>11} {'occupancy':>11} {'vs_uniform':>11}")
    print("─" * 70)
    for k, name in enumerate(SLOT_NAMES):
        ratio = means[k] / uniform if uniform > 0 else float("nan")
        print(f"{name:<20} {means[k]:>12.4f} {stds[k]:>11.4f} {occs[k]:>10.1f}%  {ratio:>10.2f}x")


def visualize_by_lane_level(
    results: dict[int, tuple[np.ndarray, np.ndarray]],
    tag: str,
    out_path: str,
) -> None:
    """
    lane-level별 학습된 slot weight를 3-panel로 비교 시각화.

    results : {lane_level: (weights, occupancy)}
      lane_level: 0=leftmost, 1=middle, 2=rightmost
    """
    levels    = [lvl for lvl in [0, 1, 2] if lvl in results]
    colors    = {0: "steelblue", 1: "seagreen", 2: "darkorange"}
    n_levels  = len(levels)

    fig, axes = plt.subplots(2, n_levels, figsize=(7 * n_levels, 10), sharey="row")
    if n_levels == 1:
        axes = axes[:, np.newaxis]

    x = np.arange(K)

    for col, lvl in enumerate(levels):
        weights, occupancy = results[lvl]
        means, stds, occs, uniform, mean_occupied = _compute_stats(weights, occupancy)
        level_name = LANE_LEVEL_NAMES[lvl]
        n_samples  = len(weights)

        # ── weight panel ──
        ax = axes[0, col]
        ax.bar(x, means, yerr=stds, capsize=4,
               color=colors[lvl], alpha=0.85, label="mean ± std")
        ax.set_xticks(x)
        ax.set_xticklabels(SLOT_NAMES, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Learned weight (softmax)" if col == 0 else "")
        ax.set_title(f"{level_name}\n(N={n_samples:,})", fontsize=11)
        ax.legend(fontsize=8)

        # ── occupancy panel ──
        ax2 = axes[1, col]
        ax2.bar(x, occs, color=colors[lvl], alpha=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(SLOT_NAMES, rotation=35, ha="right", fontsize=9)
        ax2.set_ylabel("Slot occupancy (%)" if col == 0 else "")
        ax2.set_ylim(0, 105)

        # ── console table ──
        print(f"\n{'='*60}")
        print(f"  Lane level: {level_name}  (N={n_samples:,})")
        print(f"  Uniform baseline: 1/{mean_occupied:.1f} = {uniform:.4f}")
        print(f"{'─'*60}")
        print(f"  {'Slot':<20} {'w_mean':>8} {'w_std':>7} {'occ%':>7} {'vs_uniform':>11}")
        print(f"  {'─'*55}")
        for k, name in enumerate(SLOT_NAMES):
            ratio = means[k] / uniform if uniform > 0 else float("nan")
            print(f"  {name:<20} {means[k]:>8.4f} {stds[k]:>7.4f} {occs[k]:>6.1f}%  {ratio:>10.2f}x")

    fig.suptitle(f"Per-slot learned weights by lane level — {tag}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


def visualize_by_lc_group(
    results: dict[int, tuple[np.ndarray, np.ndarray]],
    tag: str,
    out_path: str,
) -> None:
    """
    4-group directional LC별 학습된 slot weight 비교 시각화.

    results : {group_id: (weights, occupancy)}
      group_id: 0=leftmost→right, 1=middle→right, 2=middle→left, 3=rightmost→left
    """
    groups   = [g for g in [0, 1, 2, 3] if g in results]
    colors   = {0: "steelblue", 1: "seagreen", 2: "darkorange", 3: "crimson"}
    n_groups = len(groups)

    fig, axes = plt.subplots(2, n_groups, figsize=(7 * n_groups, 10), sharey="row")
    if n_groups == 1:
        axes = axes[:, np.newaxis]

    x = np.arange(K)

    for col, grp in enumerate(groups):
        weights, occupancy = results[grp]
        means, stds, occs, uniform, mean_occupied = _compute_stats(weights, occupancy)
        group_name = LC_GROUP_NAMES[grp]
        n_samples  = len(weights)

        # ── weight panel ──
        ax = axes[0, col]
        ax.bar(x, means, yerr=stds, capsize=4,
               color=colors[grp], alpha=0.85, label="mean ± std")
        ax.set_xticks(x)
        ax.set_xticklabels(SLOT_NAMES, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Learned weight (softmax)" if col == 0 else "")
        ax.set_title(f"[G{grp}] {group_name}\n(N={n_samples:,})", fontsize=10)
        ax.legend(fontsize=8)

        # ── occupancy panel ──
        ax2 = axes[1, col]
        ax2.bar(x, occs, color=colors[grp], alpha=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(SLOT_NAMES, rotation=35, ha="right", fontsize=9)
        ax2.set_ylabel("Slot occupancy (%)" if col == 0 else "")
        ax2.set_ylim(0, 105)

        # ── console table ──
        print(f"\n{'='*60}")
        print(f"  [G{grp}] {group_name}  (N={n_samples:,})")
        print(f"  LC types: {LC_GROUPS[grp]} = {[LC_TYPE_NAMES[t] for t in LC_GROUPS[grp]]}")
        print(f"  Uniform baseline: 1/{mean_occupied:.1f} = {uniform:.4f}")
        print(f"{'─'*60}")
        print(f"  {'Slot':<20} {'w_mean':>8} {'w_std':>7} {'occ%':>7} {'vs_uniform':>11}")
        print(f"  {'─'*55}")
        for k, name in enumerate(SLOT_NAMES):
            ratio = means[k] / uniform if uniform > 0 else float("nan")
            print(f"  {name:<20} {means[k]:>8.4f} {stds[k]:>7.4f} {occs[k]:>6.1f}%  {ratio:>10.2f}x")

    fig.suptitle(f"Per-slot learned weights by LC direction group — {tag}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


def visualize_lc_group_surroundings(
    results: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]],
    tag: str,
    out_path: str,
) -> None:
    """
    4-group directional LC별 pre / during / post slot weights 비교 시각화.

    results : {group_id: {lc_phase: (weights, occupancy)}}
      group_id: 0=leftmost→right, 1=middle→right, 2=middle→left, 3=rightmost→left
      lc_phase: 0=pre-LC, 1=during-LC, 2=post-LC
    """
    groups = [g for g in [0, 1, 2, 3] if g in results]
    if not groups:
        print("[WARN] 시각화할 LC 그룹 데이터 없음")
        return

    phase_colors  = {0: "royalblue", 1: "darkorange", 2: "seagreen"}
    phase_offsets = {0: -0.3, 1: 0.0, 2: 0.3}
    n_groups = len(groups)

    fig, axes = plt.subplots(2, n_groups, figsize=(7 * n_groups, 10), sharey="row")
    if n_groups == 1:
        axes = axes[:, np.newaxis]

    x = np.arange(K)

    for col, grp in enumerate(groups):
        phases = results[grp]
        ax  = axes[0, col]
        ax2 = axes[1, col]

        for ph in sorted(phases.keys()):
            weights, occupancy = phases[ph]
            means, stds, occs, uniform, _ = _compute_stats(weights, occupancy)
            off = phase_offsets[ph]
            ax.bar(x + off, means, width=0.28, yerr=stds, capsize=3,
                   color=phase_colors[ph], alpha=0.8,
                   label=f"{LC_PHASE_NAMES[ph]} (N={len(weights):,})")
            ax2.bar(x + off, occs, width=0.28,
                    color=phase_colors[ph], alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(SLOT_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"[G{grp}] {LC_GROUP_NAMES[grp]}", fontsize=9)
        ax.set_ylabel("Learned weight" if col == 0 else "")
        ax.legend(fontsize=7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(SLOT_NAMES, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("Slot occupancy (%)" if col == 0 else "")
        ax2.set_ylim(0, 105)

        # ── console table ──
        print(f"\n{'='*60}")
        print(f"  [G{grp}] {LC_GROUP_NAMES[grp]}")
        print(f"  LC types: {LC_GROUPS[grp]} = {[LC_TYPE_NAMES[t] for t in LC_GROUPS[grp]]}")
        for ph in sorted(phases.keys()):
            w, occ = phases[ph]
            means, stds, occs, uniform, _ = _compute_stats(w, occ)
            print(f"\n  --- {LC_PHASE_NAMES[ph]} (N={len(w):,}) ---")
            for k, name in enumerate(SLOT_NAMES):
                ratio = means[k] / uniform if uniform > 0 else float("nan")
                print(f"    {name:<20} {means[k]:>7.4f}  occ={occs[k]:>5.1f}%  {ratio:>6.2f}x")

    fig.suptitle(f"Slot weights: pre / during / post LC by group — {tag}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


def visualize_lc_surroundings(
    results: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]],
    tag: str,
    out_path: str,
) -> None:
    """
    LC type별 pre / post slot weights 비교 시각화 (during 제외).

    results : {lc_type: {lc_phase: (weights, occupancy)}}
      lc_type  0-5  (LC_TYPE_NAMES)
      lc_phase 0=pre (t0 이후 5s 내 LC), 2=post (hist_start 직전 5s 내 LC)
    """
    lc_types = sorted(results.keys())
    if not lc_types:
        print("[WARN] 시각화할 LC 데이터 없음")
        return

    phase_colors  = {0: "royalblue", 1: "darkorange", 2: "seagreen"}
    phase_offsets = {0: -0.3, 1: 0.0, 2: 0.3}
    n_types = len(lc_types)
    fig, axes = plt.subplots(2, n_types, figsize=(7 * n_types, 10), sharey="row")
    if n_types == 1:
        axes = axes[:, np.newaxis]

    x = np.arange(K)

    for col, lc_tp in enumerate(lc_types):
        phases = results[lc_tp]
        ax  = axes[0, col]
        ax2 = axes[1, col]

        ref_uniform = None
        for ph in sorted(phases.keys()):
            if ph not in phases:
                continue
            weights, occupancy = phases[ph]
            means, stds, occs, uniform, _ = _compute_stats(weights, occupancy)
            if ref_uniform is None:
                ref_uniform = uniform
            off = phase_offsets[ph]
            ax.bar(x + off, means, width=0.38, yerr=stds, capsize=3,
                   color=phase_colors[ph], alpha=0.8,
                   label=f"{LC_PHASE_NAMES[ph]} (N={len(weights):,})")
            ax2.bar(x + off, occs, width=0.38,
                    color=phase_colors[ph], alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(SLOT_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"[{lc_tp}] {LC_TYPE_NAMES[lc_tp]}", fontsize=10)
        ax.set_ylabel("Learned weight" if col == 0 else "")
        ax.legend(fontsize=7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(SLOT_NAMES, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("Slot occupancy (%)" if col == 0 else "")
        ax2.set_ylim(0, 105)

        # ── console table ────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  [{lc_tp}] {LC_TYPE_NAMES[lc_tp]}")
        for ph in sorted(phases.keys()):
            if ph not in phases:
                continue
            w, occ = phases[ph]
            means, stds, occs, uniform, _ = _compute_stats(w, occ)
            print(f"\n  --- {LC_PHASE_NAMES[ph]} (N={len(w):,}) ---")
            for k, name in enumerate(SLOT_NAMES):
                ratio = means[k] / uniform if uniform > 0 else float("nan")
                print(f"    {name:<20} {means[k]:>7.4f}  occ={occs[k]:>5.1f}%  {ratio:>6.2f}x")

    fig.suptitle(f"Slot weights: pre / post LC (±5s) — {tag}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    args: argparse.Namespace,
    lane_level: int | None = None,
    lc_type:    int | None = None,
    lc_types:   list[int] | None = None,
    lc_phase:   int | None = None,
    label: str = "",
) -> tuple[DataLoader, DataLoader]:
    """
    lane_level: None=전체, 0/1/2=해당 lane level
    lc_type   : None=필터없음, 0-7=해당 LC 방향 (단일)
    lc_types  : None=필터없음, [0,1,...]=해당 LC 방향 OR 필터 (복수; lc_type 무시)
    lc_phase  : None=필터없음, 0/1/2=pre/during/post
    """
    ds_paths = []
    if args.data in ("highD", "both"):
        ds_paths.append(Path("data/highD"))
    if args.data in ("exiD", "both"):
        ds_paths.append(Path("data/exiD"))

    train_list, val_list = [], []
    for ds_path in ds_paths:
        mmap_dir   = ds_path / "mmap"
        splits_dir = ds_path / "splits"
        train_idx  = np.load(splits_dir / "train_indices.npy")
        val_idx    = np.load(splits_dir / "val_indices.npy")

        if lane_level is not None:
            train_idx = filter_by_lane_level(mmap_dir, train_idx, lane_level)
            val_idx   = filter_by_lane_level(mmap_dir, val_idx,   lane_level)

        if lc_types is not None:
            ph = lc_phase if lc_phase is not None else -99
            train_idx = filter_by_lc_types(mmap_dir, train_idx, lc_types, ph)
            val_idx   = filter_by_lc_types(mmap_dir, val_idx,   lc_types, ph)
        elif lc_type is not None or lc_phase is not None:
            tp = lc_type  if lc_type  is not None else -1
            ph = lc_phase if lc_phase is not None else -99
            train_idx = filter_by_lc(mmap_dir, train_idx, tp, ph)
            val_idx   = filter_by_lc(mmap_dir, val_idx,   tp, ph)

        print(f"[{ds_path.name}]{' '+label if label else ''} stats 계산 중 ...")
        ego_mean, ego_std, nb_mean, nb_std = compute_stats(
            mmap_dir, train_idx, max_samples=args.stat_samples
        )

        train_list.append(SlotProbeDataset(mmap_dir, train_idx, ego_mean, ego_std, nb_mean, nb_std))
        val_list.append(  SlotProbeDataset(mmap_dir, val_idx,   ego_mean, ego_std, nb_mean, nb_std))

    train_ds = ConcatDataset(train_list) if len(train_list) > 1 else train_list[0]
    val_ds   = ConcatDataset(val_list)   if len(val_list)   > 1 else val_list[0]

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def _eval_val(
    model: SlotWeightProbe,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Val 루프 공통 로직 → (val_loss, weights, occupancy)."""
    model.eval()
    val_loss, all_weights, all_occupancy = 0.0, [], []
    with torch.no_grad():
        for ego, nb, mask, y in val_loader:
            ego, nb, mask, y = ego.to(device), nb.to(device), mask.to(device), y.to(device)
            pred, w = model(nb, ego, mask)
            val_loss += criterion(pred, y).item() * len(ego)
            all_weights.append(w.cpu().numpy())
            all_occupancy.append(mask.any(dim=1).cpu().numpy())
    val_loss /= len(val_loader.dataset)
    return (
        val_loss,
        np.concatenate(all_weights,   axis=0),
        np.concatenate(all_occupancy, axis=0),
    )


def train_one(
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    2-stage 학습 → best val set의 (weights, occupancy) 반환.

    Stage 1 (pretrain_epochs): slot_encoder/ego_encoder/slot_predictor 학습, weight_net 동결.
      - per-slot direct supervision: 각 슬롯이 ground truth trajectory를 독립적으로 예측.
      - loss = mean_k[ MSE(slot_pred_k, y) ]  (absent slot 제외)
      - 모든 슬롯이 각자 다른 예측을 내도록 강제 → slot_pred 다양성 확보.
      - (uniform-weighted MSE를 쓰면 모든 슬롯이 marginal mean으로 수렴하는 문제 해소)

    Stage 2 (epochs): weight_net만 학습, Stage 1 파라미터 동결.
      - 다양한 slot_pred들 중 실제 trajectory와 가장 일치하는 슬롯을 선택.
      - slot_pred가 서로 다르므로 weight gradient가 실질적인 의미를 가짐.
    """
    model     = SlotWeightProbe(
        d_enc=args.d_enc, d_hidden=args.d_hidden,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    safe_tag  = tag.replace(" ", "_").replace("/", "-")
    criterion = nn.MSELoss()

    # ── Stage 1: per-slot direct supervision (weight_net 동결) ──────────────────
    print(f"  [{tag}] Stage 1: per-slot direct supervision ({args.pretrain_epochs} epochs)")
    for p in model.weight_net.parameters():
        p.requires_grad_(False)

    stage1_params = (
        list(model.slot_encoder.parameters()) +
        list(model.ego_encoder.parameters()) +
        list(model.slot_emb.parameters()) +
        list(model.slot_predictor.parameters())
    )
    opt1 = torch.optim.Adam(stage1_params, lr=args.lr)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.pretrain_epochs)

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        train_loss = 0.0
        for ego, nb, mask, y in train_loader:
            ego, nb, mask, y = ego.to(device), nb.to(device), mask.to(device), y.to(device)

            slot_preds, nb_exist = model.forward_stage1(nb, ego, mask)  # (B,K,Tf,2), (B,K)
            # 슬롯별 MSE: absent slot 제외 후 평균
            y_exp      = y.unsqueeze(1).expand_as(slot_preds)           # (B, K, Tf, 2)
            slot_mse   = (slot_preds - y_exp).pow(2).mean(dim=(-2, -1)) # (B, K)
            nb_exist_f = nb_exist.float()
            loss = (slot_mse * nb_exist_f).sum() / nb_exist_f.sum().clamp(min=1.0)

            opt1.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(stage1_params, 1.0)
            opt1.step()
            train_loss += loss.item() * len(ego)
        train_loss /= len(train_loader.dataset)

        # val: uniform weighted sum MSE (Stage 2 시작 기준선 모니터링)
        val_loss, _, _ = _eval_val(model, val_loader, criterion, device)
        sch1.step()
        print(f"    Epoch {epoch:3d}/{args.pretrain_epochs} | train(per-slot)={train_loss:.6f}  val(uniform)={val_loss:.6f}")

    # slot_predictor + slot_emb 동결, weight_net 해제
    for p in stage1_params:
        p.requires_grad_(False)
    for p in model.weight_net.parameters():
        p.requires_grad_(True)

    # ── Stage 2: weight_net 학습 (predictor 동결) ──────────────────────────────
    print(f"  [{tag}] Stage 2: weight_net training ({args.epochs} epochs, predictor frozen)")
    opt2 = torch.optim.Adam(model.weight_net.parameters(), lr=args.lr)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epochs)

    best_val_loss  = float("inf")
    best_weights   = None
    best_occupancy = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for ego, nb, mask, y in train_loader:
            ego, nb, mask, y = ego.to(device), nb.to(device), mask.to(device), y.to(device)
            pred, _ = model(nb, ego, mask)
            loss = criterion(pred, y)
            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.weight_net.parameters(), 1.0)
            opt2.step()
            train_loss += loss.item() * len(ego)
        train_loss /= len(train_loader.dataset)

        val_loss, all_w, all_occ = _eval_val(model, val_loader, criterion, device)
        sch2.step()
        print(f"    Epoch {epoch:3d}/{args.epochs} | train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = all_w
            best_occupancy = all_occ
            torch.save(model.state_dict(), f"slot_weight_probe_{safe_tag}.pt")

    assert best_weights is not None
    return best_weights, best_occupancy


def _run_jobs(
    args: argparse.Namespace,
    device: torch.device,
    jobs: list[dict],
) -> list[dict]:
    """
    job 목록을 순차 실행하여 결과 리스트 반환.

    각 job dict는 build_loaders 키워드 + 'tag' 를 포함:
      lane_level, lc_type, lc_types, lc_phase, label, tag

    반환값: 입력 job dict에 'weights', 'occupancy' 를 추가한 리스트.
    샘플 없는 job은 결과에서 제외됨.

    Note: GPU 학습은 단일 프로세스가 효율적. 데이터 로딩 병렬화는
    DataLoader의 num_workers(현재 {args.num_workers})가 담당.
    """
    completed = []
    for job in jobs:
        tag        = job["tag"]
        lane_level = job.get("lane_level")
        lc_type    = job.get("lc_type")
        lc_types   = job.get("lc_types")
        lc_phase   = job.get("lc_phase")
        label      = job.get("label", tag)

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        train_loader, val_loader = build_loaders(
            args, lane_level=lane_level, lc_type=lc_type, lc_types=lc_types,
            lc_phase=lc_phase, label=label,
        )
        if len(train_loader.dataset) == 0:
            print(f"  [WARN] 샘플 없음 — 스킵")
            continue

        w, occ = train_one(args, device, train_loader, val_loader, tag)
        completed.append({**job, "weights": w, "occupancy": occ})

    return completed


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  seed: {args.seed}")

    if args.by_lane_level:
        # filter_by_lane_level은 lane_level == lvl (정확히 0/1/2) 인 샘플만
        # 통과시키므로, lane_level=-2 (history 안에 LC 포함) 샘플은 자동 제외됨.
        if args.data not in ("highD",):
            raise ValueError("--by_lane_level은 현재 highD 단독만 지원합니다.")

        jobs = [
            {"lane_level": lvl, "label": f"{LANE_LEVEL_NAMES[lvl]} (LC-in-history 제외)",
             "tag": f"{args.data}_ll{lvl}"}
            for lvl in [0, 1, 2]
        ]
        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for r in _run_jobs(args, device, jobs):
            results[r["lane_level"]] = (r["weights"], r["occupancy"])

        if results:
            visualize_by_lane_level(
                results, args.data,
                f"slot_weights_{args.data}_by_lane_level.png",
            )

    elif args.by_lc_group:
        if args.data not in ("highD",):
            raise ValueError("--by_lc_group은 현재 highD 단독만 지원합니다.")

        jobs = [
            {"lc_types": LC_GROUPS[grp], "lc_phase": ph,
             "label": f"[G{grp}] {LC_GROUP_NAMES[grp]} | {LC_PHASE_NAMES[ph]}",
             "tag":   f"{args.data}_lcg{grp}_ph{ph}"}
            for grp in range(4) for ph in [0, 2]
        ]
        group_surr: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for r in _run_jobs(args, device, jobs):
            tag_s = r["tag"]  # e.g. "highD_lcg1_ph0"
            grp = int(tag_s.split("lcg")[1].split("_ph")[0])
            ph  = int(tag_s.split("_ph")[1])
            group_surr.setdefault(grp, {})[ph] = (r["weights"], r["occupancy"])

        if group_surr:
            visualize_lc_group_surroundings(
                group_surr, args.data,
                f"slot_weights_{args.data}_by_lc_group_surroundings.png",
            )

    elif args.by_lc_type:
        if args.data not in ("highD",):
            raise ValueError("--by_lc_type은 현재 highD 단독만 지원합니다.")

        jobs = [
            {"lc_type": lc_tp, "lc_phase": 1,
             "label": f"[{lc_tp}] {LC_TYPE_NAMES[lc_tp]} | during-LC",
             "tag":   f"{args.data}_lct{lc_tp}"}
            for lc_tp in range(6)
        ]
        lc_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for r in _run_jobs(args, device, jobs):
            lc_results[r["lc_type"]] = (r["weights"], r["occupancy"])

        if lc_results:
            surr = {tp: {1: lc_results[tp]} for tp in lc_results}
            visualize_lc_surroundings(
                surr, f"{args.data} during-LC",
                f"slot_weights_{args.data}_by_lc_type.png",
            )

    elif args.lc_surroundings:
        if args.data not in ("highD",):
            raise ValueError("--lc_surroundings은 현재 highD 단독만 지원합니다.")

        jobs = [
            {"lc_type": lc_tp, "lc_phase": ph,
             "label": f"[{lc_tp}] {LC_TYPE_NAMES[lc_tp]} | {LC_PHASE_NAMES[ph]}",
             "tag":   f"{args.data}_lct{lc_tp}_ph{ph}"}
            for lc_tp in range(6) for ph in [0, 2]
        ]
        surr: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for r in _run_jobs(args, device, jobs):
            surr.setdefault(r["lc_type"], {})[r["lc_phase"]] = (r["weights"], r["occupancy"])

        if surr:
            visualize_lc_surroundings(
                surr, args.data,
                f"slot_weights_{args.data}_lc_surroundings.png",
            )

    else:
        train_loader, val_loader = build_loaders(args)
        w, occ = train_one(args, device, train_loader, val_loader, args.data)
        visualize(w, occ, args.data, f"slot_weights_{args.data}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slot weight probe")
    parser.add_argument("--data",         choices=["highD", "exiD", "both"], default="highD")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--pretrain_epochs", type=int,   default=100,
                        help="Stage 1: predictor pretraining epochs (균등 가중치, weight_net 동결)")
    parser.add_argument("--batch_size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--d_enc",        type=int,   default=64,
                        help="temporal encoding dim (slot/ego Transformer 출력 dim)")
    parser.add_argument("--d_hidden",     type=int,   default=64,
                        help="weight_net MLP hidden dim")
    parser.add_argument("--n_heads",      type=int,   default=4,
                        help="Transformer attention heads (d_enc must be divisible)")
    parser.add_argument("--n_layers",     type=int,   default=2,
                        help="Transformer encoder layers")
    parser.add_argument("--dropout",      type=float, default=0.1,
                        help="Transformer dropout")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="random seed for reproducibility")
    parser.add_argument("--stat_samples", type=int,   default=50_000,
                        help="stats 계산에 사용할 max sample 수")
    parser.add_argument("--num_workers",  type=int,   default=64)
    parser.add_argument("--by_lane_level", action="store_true",
                        help="lane-level(leftmost/middle/rightmost)별 분리 학습 (highD only). "
                             "combined PNG만 저장. 사전에 etc/analyze_lane_level.py 실행 필요.")
    parser.add_argument("--by_lc_group", action="store_true",
                        help="LC 방향 4-group별 during-LC 샘플 학습 (highD only). "
                             "G0=leftmost→right, G1=middle→right, G2=middle→left, G3=rightmost→left. "
                             "사전에 etc/analyze_lane_level.py --recompute 실행 필요 (type 6/7 추가).")
    parser.add_argument("--by_lc_type", action="store_true",
                        help="LC 방향 6가지별 during-LC 샘플 학습 (highD only). "
                             "사전에 etc/analyze_lane_level.py 실행 필요.")
    parser.add_argument("--lc_surroundings", action="store_true",
                        help="LC type별 pre/during/post 3단계 비교 학습 (highD only). "
                             "사전에 etc/analyze_lane_level.py 실행 필요.")
    args = parser.parse_args()
    run(args)
