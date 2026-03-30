#!/usr/bin/env python3
"""
slot_weight_probe.py — slot별 가중치를 학습하고 시각화하는 standalone 스크립트.

기존 모델(EncDecFormer)은 건드리지 않음.
- SlotWeightNet + 작은 MLP predictor를 trajectory prediction loss로 end-to-end 학습.
- 학습 후 val set에서 slot별 learned weight를 시각화.

Usage:
    python slot_weight_probe.py --data highD
    python slot_weight_probe.py --data exiD
    python slot_weight_probe.py --data both
"""

from __future__ import annotations

import argparse
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

class SlotWeightNet(nn.Module):
    """
    Per-slot weight 생성기.

    Input:
      nb_avg  (B, K, NB_DIM)  — time-averaged neighbor features (masked)
      ego_avg (B, EGO_DIM)    — time-averaged ego features
    Output:
      logits  (B, K)           — raw scores (softmax는 SlotWeightProbe에서 처리)
    """

    def __init__(self, K: int = 8, nb_dim: int = NB_DIM, ego_dim: int = EGO_DIM, d: int = 32):
        super().__init__()
        self.K        = K
        self.slot_emb = nn.Embedding(K, d)        # slot 위치/역할 인코딩
        self.nb_proj  = nn.Linear(nb_dim, d)
        self.ego_proj = nn.Linear(ego_dim, d)
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1),              # raw logit, no activation
        )

    def forward(self, nb_avg: torch.Tensor, ego_avg: torch.Tensor) -> torch.Tensor:
        B = nb_avg.shape[0]
        slot_idx  = torch.arange(self.K, device=nb_avg.device)
        slot_feat = self.slot_emb(slot_idx).unsqueeze(0).expand(B, -1, -1)         # (B, K, d)
        nb_feat   = self.nb_proj(nb_avg)                                             # (B, K, d)
        ego_feat  = self.ego_proj(ego_avg).unsqueeze(1).expand(-1, self.K, -1)     # (B, K, d)
        feat      = torch.cat([slot_feat, nb_feat, ego_feat], dim=-1)               # (B, K, 3d)
        return self.mlp(feat).squeeze(-1)                                            # (B, K)


class SlotWeightProbe(nn.Module):
    """
    SlotWeightNet + 작은 MLP predictor.

    Predictor: weighted sum of nb kinematics + ego → future positions
    """

    def __init__(
        self,
        K: int = 8,
        nb_kin_dim: int = NB_KIN,
        nb_dim: int = NB_DIM,
        ego_dim: int = EGO_DIM,
        d: int = 32,
        Tf: int = 15,
    ):
        super().__init__()
        self.weight_net = SlotWeightNet(K=K, nb_dim=nb_dim, ego_dim=ego_dim, d=d)
        self.predictor  = nn.Sequential(
            nn.Linear(nb_kin_dim + ego_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, Tf * 2),
        )
        self.nb_kin_dim = nb_kin_dim
        self.Tf         = Tf

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
        # ── time-average (mask 적용) ──
        mask    = nb_mask.unsqueeze(-1).float()           # (B, T, K, 1)
        nb_sum  = (x_nb * mask).sum(dim=1)                # (B, K, NB_DIM)
        nb_cnt  = mask.sum(dim=1).clamp(min=1.0)          # (B, K, 1)
        nb_avg  = nb_sum / nb_cnt                          # (B, K, NB_DIM)
        ego_avg = x_ego.mean(dim=1)                        # (B, EGO_DIM)

        # ── slot weights (masked softmax over occupied slots) ──
        logits   = self.weight_net(nb_avg, ego_avg)        # (B, K) raw logits
        nb_exist = nb_mask.any(dim=1)                      # (B, K) bool
        logits   = logits.masked_fill(~nb_exist, float("-inf"))
        weights  = torch.softmax(logits, dim=-1)           # (B, K), 합=1 (occupied slots)
        weights  = torch.nan_to_num(weights, nan=0.0)      # 모든 slot absent인 edge case 처리

        # ── weighted sum of kinematics ──
        nb_kin = nb_avg[..., : self.nb_kin_dim]            # (B, K, NB_KIN)
        w_nb   = (nb_kin * weights.unsqueeze(-1)).sum(dim=1)  # (B, NB_KIN)

        # ── predict ──
        feat = torch.cat([w_nb, ego_avg], dim=-1)          # (B, NB_KIN + EGO_DIM)
        pred = self.predictor(feat).view(-1, self.Tf, 2)   # (B, Tf, 2)

        return pred, weights


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize(weights: np.ndarray, occupancy: np.ndarray, tag: str, out_path: str) -> None:
    """
    weights   : (N_val, K)  — softmax weights (occupied slots의 합=1)
    occupancy : (N_val, K)  — bool, slot이 존재했는지
    """
    means, stds, occs = [], [], []
    for k in range(K):
        active = weights[occupancy[:, k], k]
        means.append(active.mean() if len(active) > 0 else 0.0)
        stds.append(active.std()   if len(active) > 0 else 0.0)
        occs.append(occupancy[:, k].mean() * 100)   # %

    # uniform baseline = 1 / (mean occupied slot count)
    mean_occupied = occupancy.sum(axis=1).mean()
    uniform = 1.0 / mean_occupied if mean_occupied > 0 else 1.0 / K

    x = np.arange(K)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # ── Weight bar ──
    ax = axes[0]
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.85, label="mean ± std")
    ax.set_xticks(x)
    ax.set_xticklabels(SLOT_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Learned weight (softmax)")
    ax.set_title(f"Per-slot learned weights — {tag}", fontsize=13)
    ax.axhline(uniform, color="gray", linestyle="--", linewidth=1,
               label=f"uniform (1/{mean_occupied:.1f} occupied = {uniform:.3f})")
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
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")

    # ── Console table ──
    print(f"\nUniform baseline: 1 / {mean_occupied:.1f} = {uniform:.4f}")
    print(f"\n{'Slot':<20} {'weight_mean':>12} {'weight_std':>11} {'occupancy':>11} {'vs_uniform':>11}")
    print("─" * 70)
    for k, name in enumerate(SLOT_NAMES):
        ratio = means[k] / uniform if uniform > 0 else float("nan")
        print(f"{name:<20} {means[k]:>12.4f} {stds[k]:>11.4f} {occs[k]:>10.1f}%  {ratio:>10.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
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

        print(f"[{ds_path.name}] stats 계산 중 (max {args.stat_samples:,} samples)...")
        ego_mean, ego_std, nb_mean, nb_std = compute_stats(
            mmap_dir, train_idx, max_samples=args.stat_samples
        )

        train_list.append(SlotProbeDataset(mmap_dir, train_idx, ego_mean, ego_std, nb_mean, nb_std))
        val_list.append(  SlotProbeDataset(mmap_dir, val_idx,   ego_mean, ego_std, nb_mean, nb_std))

    train_ds = ConcatDataset(train_list) if len(train_list) > 1 else train_list[0]
    val_ds   = ConcatDataset(val_list)   if len(val_list)   > 1 else val_list[0]

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(args)

    model     = SlotWeightProbe(d=args.d_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    best_weights   = None
    best_occupancy = None

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for ego, nb, mask, y in train_loader:
            ego, nb, mask, y = (
                ego.to(device), nb.to(device), mask.to(device), y.to(device)
            )
            pred, _ = model(nb, ego, mask)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(ego)
        train_loss /= len(train_loader.dataset)

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        val_loss      = 0.0
        all_weights   = []
        all_occupancy = []
        with torch.no_grad():
            for ego, nb, mask, y in val_loader:
                ego, nb, mask, y = (
                    ego.to(device), nb.to(device), mask.to(device), y.to(device)
                )
                pred, w = model(nb, ego, mask)
                val_loss += criterion(pred, y).item() * len(ego)
                all_weights.append(w.cpu().numpy())
                all_occupancy.append(mask.any(dim=1).cpu().numpy())  # (B, K)
        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = np.concatenate(all_weights,   axis=0)   # (N_val, K)
            best_occupancy = np.concatenate(all_occupancy, axis=0)   # (N_val, K)
            torch.save(model.state_dict(), f"slot_weight_probe_{args.data}.pt")

    # ── Visualize ──────────────────────────────────────────────────────────
    assert best_weights is not None
    out_path = f"slot_weights_{args.data}.png"
    visualize(best_weights, best_occupancy, args.data, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slot weight probe")
    parser.add_argument("--data",         choices=["highD", "exiD", "both"], default="highD")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--d_hidden",     type=int,   default=32,
                        help="hidden dim of SlotWeightNet")
    parser.add_argument("--stat_samples", type=int,   default=50_000,
                        help="stats 계산에 사용할 max sample 수")
    parser.add_argument("--num_workers",  type=int,   default=4)
    args = parser.parse_args()
    run(args)
