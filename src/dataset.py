#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py — PyTorch Dataset for highD mmap-preprocessed data.

preprocess.py가 생성하는 파일 구조
────────────────────────────────────
  x_ego.npy          (N, T, 6)      ego features  [x, y, xV, yV, xA, yA]
  x_nb.npy           (N, T, K, 10)  neighbor features (아래 참조)
  nb_mask.npy        (N, T, K)      bool — True if neighbor exists
  y.npy              (N, Tf, 2)     future position  [x, y]
  y_vel.npy          (N, Tf, 2)     future velocity  [xV, yV]
  y_acc.npy          (N, Tf, 2)     future acceleration [xA, yA]
  x_last_abs.npy     (N, 2)         last absolute (x, y) of ego history
  meta_recordingId.npy / meta_trackId.npy / meta_frame.npy  (N,)
  stats.npz          ego_mean/std, nb_mean/std

x_nb feature index map (preprocess.py 기준)
────────────────────────────────────────────
  0  dx          longitudinal distance
  1  dy          lateral distance
  2  dvx         relative longitudinal velocity
  3  dvy         relative lateral velocity
  4  dax         relative longitudinal acceleration
  5  day         relative lateral acceleration
  6  s_x         longitudinal interaction state (existing LIS)
  7  s_y         sqrt(lc_state^2 + delta_lane^2)
  8  dim         vehicle size bin
  9  I           composite importance

nb_kin_mode 옵션
────────────────
  "p"    [dx, dy]                          2-dim
  "v"    [dvx, dvy]                        2-dim
  "a"    [dax, day]                        2-dim
  "pv"   [dx, dy, dvx, dvy]               4-dim
  "pa"   [dx, dy, dax, day]               4-dim
  "va"   [dvx, dvy, dax, day]             4-dim
  "pva"  [dx, dy, dvx, dvy, dax, day]     6-dim  (default)
  "none" kinematics 없음 (aux feature만 사용 시)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# x_ego feature index constants  (preprocess.py 정의와 동일)
# ──────────────────────────────────────────────────────────────────────────────
_EGO_MODE_SLICES: Dict[str, Any] = {
    "p":   slice(0, 2),   # x, y
    "pv":  slice(0, 4),   # x, y, vx, vy
    "pva": slice(0, 6),   # x, y, vx, vy, ax, ay  (전체)
}

# x_nb feature index constants  (preprocess.py 정의와 동일)
# ──────────────────────────────────────────────────────────────────────────────
_NB_IDX_KIN_END  = 6   # 0..5  kinematics
_NB_IDX_SX       = 6
_NB_IDX_SY       = 7
_NB_IDX_DIM      = 8
_NB_IDX_I        = 9

_NB_KIN_MODE_SLICES: Dict[str, Any] = {
    "p":   slice(0, 2),
    "v":   slice(2, 4),
    "a":   slice(4, 6),
    "pv":  slice(0, 4),
    "va":  slice(2, 6),
    "pva": slice(0, 6),
}
_NB_KIN_MODES_NONCONTIGUOUS = {"pa"}   # 별도 cat 필요


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HighDDataset(Dataset):
    """
    highD mmap 데이터셋.

    Parameters
    ----------
    data_dir : Path
        preprocess.py 가 생성한 mmap 디렉터리 경로.
    split_indices : np.ndarray, optional
        split.py 가 생성한 index 배열 (train/val/test_indices.npy).
        None 이면 전체 샘플 사용.
    stats : dict, optional
        {"ego_mean", "ego_std", "nb_mean", "nb_std"} — torch.Tensor.
        load_stats() 헬퍼로 로드 권장.
        None 이면 정규화 생략.

    Feature 선택 (모두 config/argument로 제어)
    ──────────────────────────────────────────
    ego_mode : str
        "p"   → [x, y]                  ego_dim=2
        "pv"  → [x, y, vx, vy]          ego_dim=4
        "pva" → [x, y, vx, vy, ax, ay]  ego_dim=6  (default)
    nb_kin_mode : str
        "p" | "v" | "a" | "pv" | "pa" | "va" | "pva" | "none"
    use_s_x      : bool   x_nb index 6
    use_s_y      : bool   x_nb index 7
    use_dim      : bool   x_nb index 8
    use_I        : bool   x_nb index 9
    return_meta  : bool   recordingId / trackId / t0_frame 반환 여부
    """

    def __init__(
        self,
        data_dir: Path | str,
        split_indices: Optional[np.ndarray] = None,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        # ── ego kinematics ────────────────────────────────────────────────────
        ego_mode: str = "pva",
        # ── neighbor kinematics ───────────────────────────────────────────────
        nb_kin_mode: str = "pva",
        # ── neighbor aux features ─────────────────────────────────────────────
        use_s_x:      bool = True,
        use_s_y:      bool = True,
        use_dim:      bool = True,
        use_I:        bool = False,
        # ── misc ─────────────────────────────────────────────────────────────
        return_meta: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)

        # ── ego_mode 검증 ─────────────────────────────────────────────────────
        ego_mode = ego_mode.lower().strip()
        if ego_mode not in _EGO_MODE_SLICES:
            raise ValueError(
                f"ego_mode must be one of {sorted(_EGO_MODE_SLICES)}, got: '{ego_mode}'"
            )
        self.ego_mode = ego_mode

        # ── nb_kin_mode 검증 ──────────────────────────────────────────────────
        _valid_kin_modes = set(_NB_KIN_MODE_SLICES) | _NB_KIN_MODES_NONCONTIGUOUS | {"none"}
        nb_kin_mode = nb_kin_mode.lower().strip()
        if nb_kin_mode not in _valid_kin_modes:
            raise ValueError(
                f"nb_kin_mode must be one of {sorted(_valid_kin_modes)}, got: '{nb_kin_mode}'"
            )
        self.nb_kin_mode = nb_kin_mode

        self.use_s_x      = use_s_x
        self.use_s_y      = use_s_y
        self.use_dim      = use_dim
        self.use_I        = use_I
        self.return_meta  = return_meta
        self.stats        = stats

        # neighbor feature가 하나도 없는 경우 방어
        if nb_kin_mode == "none" and not any(
            [use_s_x, use_s_y, use_dim, use_I]
        ):
            raise ValueError(
                "All neighbor features are disabled. "
                "Enable at least one neighbor feature."
            )

        # ── mmap 파일 로드 ────────────────────────────────────────────────────
        self._x_ego    = self._load("x_ego.npy")        # (N, T, 6)
        self._x_nb     = self._load("x_nb.npy")         # (N, T, K, 10)
        self._nb_mask  = self._load("nb_mask.npy")      # (N, T, K)
        self._y        = self._load("y.npy")             # (N, Tf, 2)
        self._x_last   = self._load("x_last_abs.npy")   # (N, 2)

        self._y_vel = self._load_optional("y_vel.npy")  # (N, Tf, 2) or None
        self._y_acc = self._load_optional("y_acc.npy")  # (N, Tf, 2) or None
        self._meta_rec   = None
        self._meta_track = None
        self._meta_frame = None
        if return_meta:
            self._meta_rec   = self._load("meta_recordingId.npy")
            self._meta_track = self._load("meta_trackId.npy")
            self._meta_frame = self._load("meta_frame.npy")

        # ── split indices ─────────────────────────────────────────────────────
        self.indices = (
            split_indices if split_indices is not None
            else np.arange(len(self._x_ego))
        )

        # ── stats 캐시 ────────────────────────────────────────────────────────
        self._ego_mean: Optional[torch.Tensor] = None
        self._ego_std:  Optional[torch.Tensor] = None
        self._nb_mean:  Optional[torch.Tensor] = None
        self._nb_std:   Optional[torch.Tensor] = None
        if stats is not None:
            self._ego_mean = stats.get("ego_mean")
            self._ego_std  = stats.get("ego_std")
            self._nb_mean  = stats.get("nb_mean")
            self._nb_std   = stats.get("nb_std")

    # ── 내부 로드 헬퍼 ────────────────────────────────────────────────────────

    def _load(self, filename: str) -> np.ndarray:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required mmap file not found: {path}")
        return np.load(path, mmap_mode="r")

    def _load_optional(self, filename: str) -> Optional[np.ndarray]:
        path = self.data_dir / filename
        return np.load(path, mmap_mode="r") if path.exists() else None

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = int(self.indices[idx])

        # ── ego history ───────────────────────────────────────────────────────
        raw_ego = torch.from_numpy(self._x_ego[real_idx].copy())   # (T, 6)
        x_ego   = raw_ego[..., _EGO_MODE_SLICES[self.ego_mode]]    # (T, ego_dim)

        # ── neighbor features ─────────────────────────────────────────────────
        raw_nb  = torch.from_numpy(self._x_nb[real_idx].copy())    # (T, K, 10)
        nb_mask = torch.from_numpy(self._nb_mask[real_idx].copy()) # (T, K)

        nb_parts: List[torch.Tensor] = []

        # kinematics
        if self.nb_kin_mode != "none":
            if self.nb_kin_mode == "pa":
                kin = torch.cat(
                    [raw_nb[..., 0:2], raw_nb[..., 4:6]], dim=-1   # dx,dy,dax,day
                )
            else:
                kin = raw_nb[..., _NB_KIN_MODE_SLICES[self.nb_kin_mode]]
            nb_parts.append(kin)

        # aux features
        if self.use_s_x:
            nb_parts.append(raw_nb[..., _NB_IDX_SX   : _NB_IDX_SX   + 1])
        if self.use_s_y:
            nb_parts.append(raw_nb[..., _NB_IDX_SY   : _NB_IDX_SY   + 1])
        if self.use_dim:
            nb_parts.append(raw_nb[..., _NB_IDX_DIM  : _NB_IDX_DIM  + 1])
        if self.use_I:
            nb_parts.append(raw_nb[..., _NB_IDX_I    : _NB_IDX_I    + 1])

        x_nb = torch.cat(nb_parts, dim=-1)  # (T, K, D_nb)

        # ── targets ───────────────────────────────────────────────────────────
        y          = torch.from_numpy(self._y[real_idx].copy())         # (Tf, 2)
        x_last_abs = torch.from_numpy(self._x_last[real_idx].copy())   # (2,)
        y_vel = (
            torch.from_numpy(self._y_vel[real_idx].copy())
            if self._y_vel is not None else torch.zeros_like(y)
        )
        y_acc = (
            torch.from_numpy(self._y_acc[real_idx].copy())
            if self._y_acc is not None else torch.zeros_like(y)
        )

        # ── normalize ─────────────────────────────────────────────────────────
        if self.stats is not None:
            if self._ego_mean is not None:
                x_ego = (x_ego - self._ego_mean) / self._ego_std.clamp_min(1e-2)
            if self._nb_mean is not None:
                x_nb = (x_nb - self._nb_mean) / self._nb_std.clamp_min(1e-2)

        # ── output dict ───────────────────────────────────────────────────────
        out: Dict[str, Any] = {
            "x_ego":      x_ego,
            "x_nb":       x_nb,
            "nb_mask":    nb_mask,
            "y":          y,
            "y_vel":      y_vel,
            "y_acc":      y_acc,
            "x_last_abs": x_last_abs,
        }

        if self.return_meta:
            out["meta"] = {
                "recordingId": int(self._meta_rec[real_idx]),
                "trackId":     int(self._meta_track[real_idx]),
                "t0_frame":    int(self._meta_frame[real_idx]),
            }

        return out

    # ── 편의 property ─────────────────────────────────────────────────────────

    @property
    def ego_feature_dim(self) -> int:
        return _EGO_MODE_SLICES[self.ego_mode].stop

    @property
    def nb_feature_dim(self) -> int:
        """현재 설정에서 x_nb 의 feature 차원 수를 반환합니다."""
        dim = 0
        if self.nb_kin_mode != "none":
            if self.nb_kin_mode in ("p", "v", "a"):
                dim += 2
            elif self.nb_kin_mode in ("pv", "pa", "va"):
                dim += 4
            elif self.nb_kin_mode == "pva":
                dim += 6
        if self.use_s_x:      dim += 1
        if self.use_s_y:      dim += 1
        if self.use_dim:      dim += 1
        if self.use_I:        dim += 1
        return dim

    @property
    def meta_rec(self):
        return self._meta_rec

    @property
    def meta_track(self):
        return self._meta_track

    @property
    def meta_frame(self):
        return self._meta_frame

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        """
        현재 설정에서 실제 사용되는 feature 이름을 반환합니다.
        log.py 등에서 실험 기록용으로 사용합니다.
        """
        _EGO_NAMES = {
            "p":   ["x", "y"],
            "pv":  ["x", "y", "vx", "vy"],
            "pva": ["x", "y", "vx", "vy", "ax", "ay"],
        }
        ego: List[str] = list(_EGO_NAMES[self.ego_mode])

        _KIN_NAMES = {
            "p":    ["dx", "dy"],
            "v":    ["dvx", "dvy"],
            "a":    ["dax", "day"],
            "pv":   ["dx", "dy", "dvx", "dvy"],
            "pa":   ["dx", "dy", "dax", "day"],
            "va":   ["dvx", "dvy", "dax", "day"],
            "pva":  ["dx", "dy", "dvx", "dvy", "dax", "day"],
            "none": [],
        }
        nb: List[str] = list(_KIN_NAMES[self.nb_kin_mode])
        if self.use_s_x:      nb.append("s_x")
        if self.use_s_y:      nb.append("s_y")
        if self.use_dim:      nb.append("dim")
        if self.use_I:        nb.append("I")

        return {
            "ego": ego,
            "nb":  nb,
        }

    def __repr__(self) -> str:
        names = self.feature_names
        return (
            f"HighDDataset(n={len(self)}, "
            f"ego=[{', '.join(names['ego'])}], "
            f"nb=[{', '.join(names['nb'])}])"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Stats 로더 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def load_stats(stats_path: Path | str) -> Dict[str, torch.Tensor]:
    """
    stats.npz 를 로드해 torch.Tensor dict 로 반환합니다.

    Parameters
    ----------
    stats_path : make_stats_filename() 으로 생성한 .npz 파일의 전체 경로

    Returns
    -------
    {"ego_mean", "ego_std", "nb_mean", "nb_std"}
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(
            f"stats file not found: {path}\n"
            "compute_stats_if_needed() 가 자동으로 생성합니다."
        )
    npz = np.load(path)
    return {
        "ego_mean": torch.from_numpy(npz["ego_mean"].copy()),
        "ego_std":  torch.from_numpy(npz["ego_std"].copy()),
        "nb_mean":  torch.from_numpy(npz["nb_mean"].copy()),
        "nb_std":   torch.from_numpy(npz["nb_std"].copy()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    DataLoader collate function.
    선택적으로 포함된 y_vel / y_acc / meta 를 안전하게 처리합니다.
    """
    out: Dict[str, Any] = {
        "x_ego":      torch.stack([b["x_ego"]      for b in batch]),  # (B, T, ego_dim)
        "x_nb":       torch.stack([b["x_nb"]       for b in batch]),  # (B, T, K, D_nb)
        "nb_mask":    torch.stack([b["nb_mask"]     for b in batch]),  # (B, T, K)
        "y":          torch.stack([b["y"]           for b in batch]),  # (B, Tf, 2)
        "x_last_abs": torch.stack([b["x_last_abs"]  for b in batch]),  # (B, 2)
    }

    for optional_key in ("y_vel", "y_acc"):
        if optional_key in batch[0]:
            out[optional_key] = torch.stack([b[optional_key] for b in batch])

    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]

    return out
