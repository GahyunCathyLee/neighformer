#!/usr/bin/env python3
"""
preprocess.py  —  HighD preprocessing pipeline  (raw CSV → mmap)

STAGE raw2mmap :  highD raw CSV  →  memory-mapped arrays

stats 계산은 train.py / evaluate.py 실행 시 src/stats.py 가 자동으로 수행합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature schema
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x_ego  : (N, T, 6)
    [x, y, xV, yV, xA, yA]

x_nb   : (N, T, K, 10)   ego-relative neighbor features
    idx  0  dx        longitudinal distance  (nb_x - ego_x)
    idx  1  dy        lateral distance       (nb_y - ego_y)
    idx  2  dvx       relative longitudinal velocity
    idx  3  dvy       relative lateral velocity
    idx  4  dax       relative longitudinal acceleration
    idx  5  day       relative lateral acceleration
    idx  6  s_x       longitudinal interaction state (existing LIS)
    idx  7  s_y       sqrt(lc_state^2 + delta_lane^2)
    idx  8  dim       vehicle size bin (0~4) based on width*length*height_est
                      0: 소형차 (5~12 m³), 1: 일반 승용차 (12~20 m³)
                      2: 대형 승용/픽업 (20~90 m³), 3: 중형 트럭 (90~150 m³)
                      4: 대형 트럭 (150~220 m³)
    idx  9  I         exp(-lambda_x*|s_x|^alpha - lambda_y*s_y^beta)

y          : (N, Tf, 2)     future [x, y]
y_vel      : (N, Tf, 2)     future [xV, yV]
y_acc      : (N, Tf, 2)     future [xA, yA]
nb_mask    : (N, T, K)      bool - True if neighbor exists
x_last_abs : (N, 2)         last absolute (x, y) of ego history
meta_recordingId / trackId / t0_frame : (N,)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIS (Longitudinal Interaction State) modes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    '3'    : {-1,...,1}      lit 3-bin   [-inf,-7.5696,6.4434,inf]
    '5'    : {-2,...,2}      lit 5-bin   [-inf,-16.3463,-4.3708,3.4246,15.5837,inf]
    '7'    : {-3,...,3}      lit 7-bin   [-inf,-22.0410,-10.2790,-3.1883,2.2974,9.1881,21.6504,inf]
    '9'    : {-4,...,4}      lit 9-bin   [-inf,-26.4327,-14.5688,-7.5696,-2.5658,1.6850,6.4434,13.7152,26.2945,inf]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Importance formula
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    s_x = existing LIS
    s_y = sqrt(lc_state^2 + delta_lane^2)
    I   = exp(-lambda_x*|s_x|^alpha - lambda_y*s_y^beta)

    default: lambda_x=0.1, lambda_y=0.1, alpha=1.5, beta=2.0
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NEIGHBOR_COLS_8 = [
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftAlongsideId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightAlongsideId",
    "rightFollowingId",
]

EGO_DIM = 6    # x, y, xV, yV, xA, yA
NB_DIM  = 10   # dx, dy, dvx, dvy, dax, day, s_x, s_y, dim, I
K       = 8    # neighbor slots

# Slot priority for top-N gate tie-breaking: 0 > 2 > 5 > 1 > 4 > 7 > 3 > 6
_TOPN_SLOT_PRIORITY = {s: r for r, s in enumerate([0, 2, 5, 1, 4, 7, 3, 6])}

# Empirical slot weights (mean I per slot, from dataset analysis)
# Order: preceding, following, leftPreceding, leftAlongside, leftFollowing,
#        rightPreceding, rightAlongside, rightFollowing
SLOT_WEIGHTS = [0.4944, 0.0411, 0.0935, 0.0074, 0.0002, 0.5559, 0.0000, 0.1179]

# Conditional slot weights derived from SlotWeightProbe models (mean softmax per slot).
# Used when Config.slot_importance_conditional = True.

# No-LC case: weights by ego lane level  (0=leftmost/fast, 1=middle, 2=rightmost/slow)
SLOT_WEIGHTS_BY_LANE_LEVEL = [
    [0.4255, 0.0336, 0.0000, 0.0000, 0.0000, 0.4574, 0.0119, 0.1190],  # ll0 leftmost
    [0.4805, 0.0002, 0.0000, 0.0000, 0.0000, 0.3803, 0.0234, 0.1839],  # ll1 middle
    [0.4784, 0.0373, 0.3344, 0.0343, 0.2050, 0.0000, 0.0000, 0.0000],  # ll2 rightmost
]

# LC-in-history case: pre-LC weights per LC group (G0-G3)
# Order: preceding, following, leftPreceding, leftAlongside, leftFollowing,
#        rightPreceding, rightAlongside, rightFollowing
# G0: leftmost→right  (lct0 leftmost→middle,    lct1 leftmost→rightmost)
# G1: middle→right    (lct3 middle→rightmost,    lct6 middle→middle(right))
# G2: middle→left     (lct2 middle→leftmost,     lct7 middle→middle(left))
# G3: rightmost→left  (lct4 rightmost→leftmost,  lct5 rightmost→middle)
SLOT_WEIGHTS_PRE_LC = [
    [0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.6253, 0.2663, 0.3117],  # G0 leftmost→right
    [0.0072, 0.0263, 0.0006, 0.0000, 0.0000, 0.3970, 0.3776, 0.5494],  # G1 middle→right
    [0.0183, 0.1326, 0.6745, 0.5179, 0.2365, 0.0000, 0.0000, 0.0000],  # G2 middle→left
    [0.0381, 0.0233, 0.5755, 0.3548, 0.4799, 0.0000, 0.0000, 0.0000],  # G3 rightmost→left
]

# LC-in-history case: post-LC weights per LC group (G0-G3)
SLOT_WEIGHTS_POST_LC = [
    [0.0460, 0.3983, 0.0000, 0.0023, 0.0762, 0.2338, 0.2022, 0.3281],  # G0 leftmost→right
    [0.1036, 0.0851, 0.4832, 0.0540, 0.3810, 0.0013, 0.0000, 0.0002],  # G1 middle→right
    [0.6018, 0.3591, 0.0115, 0.0013, 0.0099, 0.1709, 0.0069, 0.0014],  # G2 middle→left
    [0.2618, 0.0000, 0.0036, 0.0000, 0.0000, 0.6545, 0.2032, 0.1449],  # G3 rightmost→left
]

# lc_type → LC group index  (G0=0, G1=1, G2=2, G3=3)
# lc_type: 0=leftmost→middle, 1=leftmost→rightmost, 2=middle→leftmost,
#           3=middle→rightmost, 4=rightmost→leftmost, 5=rightmost→middle,
#           6=middle→middle(right), 7=middle→middle(left)
_LC_TYPE_TO_GROUP: Dict[int, int] = {
    0: 0, 1: 0,  # G0 leftmost→right
    3: 1, 6: 1,  # G1 middle→right
    2: 2, 7: 2,  # G2 middle→left
    4: 3, 5: 3,  # G3 rightmost→left
}

# (from_level, to_level) → lc_type  (mirrors analyze_lane_level.py)
_LC_TYPE_MAP_LEVEL: Dict[Tuple[int, int], int] = {
    (0, 1): 0, (0, 2): 1,
    (1, 0): 2, (1, 2): 3,
    (2, 0): 4, (2, 1): 5,
}


def _apply_topn_gate(nb_row: np.ndarray, mask_row: np.ndarray, n: int) -> None:
    """Select top-n slots by I (idx 9) and remove the rest (in-place).
    Tie-breaking: slot priority 0>2>5>1>4>7>3>6.
    """
    K_local = nb_row.shape[0]
    valid = [k for k in range(K_local) if mask_row[k]]
    valid.sort(key=lambda k: (-nb_row[k, 9], _TOPN_SLOT_PRIORITY.get(k, K_local)))
    selected = set(valid[:n])
    for k in valid:
        if k not in selected:
            nb_row[k] = 0.0
            mask_row[k] = False


# ─────────────────────────────────────────────────────────────────────────────
# LIS binning
# ─────────────────────────────────────────────────────────────────────────────

# cuts: inner bin boundaries (exclusive upper). vals: one more element than cuts.
# L: max absolute LIS value, used to normalise lis -> lnorm = lis / L before importance calc.
LIS_BINS: Dict[str, Dict] = {
    '3': {'cuts': [-5.8639, 4.9525],
          'vals': [-1.0, 0.0, 1.0],
          'L': 1.0},
    '5': {'cuts': [-13.7033, -3.0238, 2.2735, 13.0957],
          'vals': [-2.0, -1.0, 0.0, 1.0, 2.0],
          'L': 2.0},
    '7': {'cuts': [-18.7902, -8.2922, -1.9963, 1.3381, 7.3744, 18.5267],
          'vals': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
          'L': 3.0},
    '9': {'cuts': [-22.7661, -12.1209, -5.8639, -1.4829, 0.9127, 4.9525, 11.4115, 22.7702],
          'vals': [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
          'L': 4.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle size bin
# ─────────────────────────────────────────────────────────────────────────────

# Bin edges for width * length * height_est (m³): [5, 12, 20, 90, 150, 220]
# Bin values: 0 (소형차) ~ 4 (대형 트럭)
_VOLUME_BIN_EDGES = [12.0, 20.0, 90.0, 150.0]  # 4 inner cuts → 5 bins


def _volume_bin(phys_length: float, phys_width: float, vehicle_class: str) -> Tuple[float, float]:
    """Return (size bin index 0~4, raw volume m³) for a neighbor vehicle.

    height is estimated from vehicle class and physical length:
      Car:   length < 4.5m → 1.45m,  < 5.0m → 1.70m,  >= 5.0m → 1.90m
      Truck: length < 12.0m → 2.75m, >= 12.0m → 3.75m
    """
    if vehicle_class == "Car":
        if phys_length < 4.5:   height = 1.45
        elif phys_length < 5.0: height = 1.70
        else:                   height = 1.90
    else:
        height = 2.75 if phys_length < 12.0 else 3.75
    volume = phys_width * phys_length * height
    for i, edge in enumerate(_VOLUME_BIN_EDGES):
        if volume < edge:
            return float(i), volume
    return 4.0, volume


def _lit_to_lis(lit: float, lis_mode: str) -> float:
    cfg = LIS_BINS[lis_mode]
    return cfg['vals'][bisect.bisect_right(cfg['cuts'], lit)]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # paths
    data_dir: Path = Path("data/highD")
    raw_dir:  Path = Path("raw")
    mmap_dir: Path = Path("mmap")

    @property
    def raw_path(self) -> Path:
        return self.data_dir / self.raw_dir

    @property
    def mmap_path(self) -> Path:
        return self.data_dir / self.mmap_dir

    # recording
    target_hz:          float = 3.0
    history_sec:        float = 2.0
    future_sec:         float = 5.0
    stride_sec:         float = 1.0
    normalize_upper_xy: bool  = True

    # LIS mode
    lis_mode: str = '3'  # '3' | '5' | '7' | '9'

    # importance params: I = exp(-lambda_x*|s_x|^alpha - lambda_y*s_y^beta)
    lambda_x: float = 0.1
    lambda_y: float = 0.1
    alpha:    float = 1.5
    beta:     float = 2.0

    # importance gate
    gate_topn: int = 0     # >0 = keep top-N slots by I; 0 = disabled

    # slot importance: I_new = min(I * (1 + alpha * w_slot), 1.0);  0.0 = disabled
    slot_importance_alpha: float = 0.0

    # conditional slot weights: use lane-level / pre-LC / post-LC weights instead of
    # the global SLOT_WEIGHTS.  Requires slot_importance_alpha > 0.
    slot_importance_conditional: bool = False

    # neighbor feature mode
    non_relative: bool = False  # True → x_nb[0:6] = nb's abs values in globally-shifted frame

    # output / execution
    dry_run:     bool = False
    num_workers: int  = 0   # 0 = os.cpu_count()


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(x: np.ndarray, default: float = 0.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    bad = ~np.isfinite(x)
    if np.any(bad):
        x = x.copy()
        x[bad] = default
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_importance(
    s_x: float,
    s_y: float,
    lambda_x: float,
    lambda_y: float,
    alpha: float,
    beta: float,
) -> float:
    """I = exp(-lambda_x*|s_x|^alpha - lambda_y*s_y^beta)."""
    return float(np.exp(-lambda_x * (abs(s_x) ** alpha) - lambda_y * (s_y ** beta)))


# ─────────────────────────────────────────────────────────────────────────────
# Conditional slot weight helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lane_id_to_level(lid: int, dd: int, sorted_lids: List[int], post_flip: bool) -> int:
    """lane_id → lane_level (0=leftmost/fast, 1=middle, 2=rightmost/slow).

    post_flip=True  : normalize_upper_xy が True の場合（maybe_flip 済み）
      両方向とも ascending sort → idx 0 = leftmost (level 0)
    post_flip=False : normalize_upper_xy が False の場合
      dd=1 ascending → idx 0 = rightmost (level 2), idx -1 = leftmost (level 0)
      dd=2 ascending → idx 0 = leftmost (level 0),  idx -1 = rightmost (level 2)
    """
    n = len(sorted_lids)
    if n == 0 or lid not in sorted_lids:
        return -1
    idx = sorted_lids.index(lid)
    if n == 1:
        return 1
    if post_flip or dd == 2:
        if idx == 0:     return 0  # leftmost
        if idx == n - 1: return 2  # rightmost
        return 1
    else:  # dd=1, no flip
        if idx == 0:     return 2  # rightmost
        if idx == n - 1: return 0  # leftmost
        return 1


def _ego_lc_context(
    ego_lane_arr: np.ndarray,
    dd: int,
    lane_ids_per_dd: Dict[int, List[int]],
    post_flip: bool,
) -> Tuple[int, Optional[int], int]:
    """history window 내 ego LC 상태를 판단한다.

    Returns
    -------
    lane_level  : 0/1/2 (no-LC, ego의 t0 차선), -2 (LC in history), -1 (unknown)
    lc_frame_ti : LC가 처음 일어난 hist frame 인덱스 (None = no LC)
    lc_type     : 0-5  (-1 = no LC or unknown)
    """
    sorted_lids = lane_ids_per_dd.get(dd, [])

    lc_frame_ti: Optional[int] = None
    lc_type = -1

    for ti in range(1, len(ego_lane_arr)):
        if ego_lane_arr[ti] != ego_lane_arr[ti - 1]:
            lc_frame_ti = ti
            from_lvl = _lane_id_to_level(int(ego_lane_arr[ti - 1]), dd, sorted_lids, post_flip)
            to_lvl   = _lane_id_to_level(int(ego_lane_arr[ti]),     dd, sorted_lids, post_flip)
            lc_type  = _LC_TYPE_MAP_LEVEL.get((from_lvl, to_lvl), -1)
            break

    if lc_frame_ti is None:
        lane_level = _lane_id_to_level(int(ego_lane_arr[-1]), dd, sorted_lids, post_flip)
    else:
        lane_level = -2

    return lane_level, lc_frame_ti, lc_type


def _get_slot_weight(
    ki: int,
    ti: int,
    lane_level: int,
    lc_frame_ti: Optional[int],
    lc_type: int,
) -> float:
    """slot ki / time step ti に対応する条件付き slot weight を返す。"""
    if lc_frame_ti is not None and lc_type >= 0:
        lc_group = _LC_TYPE_TO_GROUP.get(lc_type, -1)
        if lc_group < 0:
            return SLOT_WEIGHTS[ki]  # unknown lc_type fallback
        if ti < lc_frame_ti:
            return SLOT_WEIGHTS_PRE_LC[lc_group][ki]
        else:
            return SLOT_WEIGHTS_POST_LC[lc_group][ki]
    elif 0 <= lane_level <= 2:
        return SLOT_WEIGHTS_BY_LANE_LEVEL[lane_level][ki]
    else:
        return SLOT_WEIGHTS[ki]  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Raw CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_semicolon_floats(s: str) -> List[float]:
    if not isinstance(s, str):
        return []
    return [float(p) for p in s.strip().split(";") if p.strip()]


def find_recording_ids(raw_dir: Path) -> List[str]:
    ids = [re.match(r"(\d+)_tracks\.csv$", p.name).group(1)
           for p in raw_dir.glob("*_tracks.csv")
           if re.match(r"(\d+)_tracks\.csv$", p.name)]
    return sorted(set(ids))


def flip_constants(rec_meta: pd.DataFrame) -> Tuple[float, float, np.ndarray, np.ndarray]:
    fr    = float(rec_meta.loc[0, "frameRate"])
    upper = parse_semicolon_floats(str(rec_meta.loc[0, "upperLaneMarkings"])) if "upperLaneMarkings" in rec_meta.columns else []
    lower = parse_semicolon_floats(str(rec_meta.loc[0, "lowerLaneMarkings"])) if "lowerLaneMarkings" in rec_meta.columns else []
    ua, la = np.array(upper, np.float32), np.array(lower, np.float32)
    C_y   = float(ua[-1] + la[0]) if (len(ua) and len(la)) else 0.0
    return C_y, fr, ua, la


def maybe_flip(x, y, xv, yv, xa, ya, lane_id, dd, C_y, x_max, upper_mm):
    mask = dd == 1
    if not np.any(mask):
        return x, y, xv, yv, xa, ya, lane_id
    x2, y2, xv2, yv2, xa2, ya2, l2 = (a.copy() for a in (x, y, xv, yv, xa, ya, lane_id))
    x2[mask]  = x_max - x2[mask]
    y2[mask]  = C_y   - y2[mask]
    xv2[mask] = -xv2[mask];  yv2[mask] = -yv2[mask]
    xa2[mask] = -xa2[mask];  ya2[mask] = -ya2[mask]
    if upper_mm is not None:
        mn, mx = upper_mm
        ok = mask & (l2 > 0)
        l2[ok] = (mn + mx) - l2[ok]
    return x2, y2, xv2, yv2, xa2, ya2, l2


def build_lane_tables(markings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if markings is None or len(markings) < 2:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    left, right = markings[:-1], markings[1:]
    return ((right + left) * 0.5).astype(np.float32), (right - left).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-recording processing  (raw CSV -> list of sample dicts)
# ─────────────────────────────────────────────────────────────────────────────

def _recording_to_buf(cfg: Config, rec_id: str) -> Optional[Dict[str, np.ndarray]]:
    raw_dir  = cfg.raw_path
    rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    trk_meta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")

    C_y, frame_rate, upper_mark, lower_mark = flip_constants(rec_meta)
    step   = max(1, int(round(frame_rate / cfg.target_hz)))
    T      = int(round(cfg.history_sec  * cfg.target_hz))
    Tf     = int(round(cfg.future_sec   * cfg.target_hz))
    stride = max(1, int(round(cfg.stride_sec * cfg.target_hz)))

    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns: tracks[c] = 0
    for c in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
        if c not in tracks.columns: tracks[c] = 0.0
    if "laneId" not in tracks.columns: tracks["laneId"] = 0

    vid_to_dd    = dict(zip(trk_meta["id"].astype(int), trk_meta["drivingDirection"].astype(int)))
    vid_to_w     = dict(zip(trk_meta["id"].astype(int), trk_meta["width"].astype(float)))
    vid_to_l     = dict(zip(trk_meta["id"].astype(int), trk_meta["height"].astype(float)))
    vid_to_class = dict(zip(trk_meta["id"].astype(int), trk_meta["class"].astype(str)))

    upper_for_calc = upper_mark.copy()
    if cfg.normalize_upper_xy and len(upper_for_calc):
        upper_for_calc = np.sort((C_y - upper_for_calc).astype(np.float32))
    upper_center, upper_width = build_lane_tables(upper_for_calc)
    lower_center, lower_width = build_lane_tables(lower_mark)
    upper_mm = (1, int(len(upper_center))) if len(upper_center) else None

    frame   = tracks["frame"].astype(np.int32).to_numpy()
    vid     = tracks["id"].astype(np.int32).to_numpy()
    x       = tracks["x"].astype(np.float32).to_numpy().copy()
    y       = tracks["y"].astype(np.float32).to_numpy().copy()
    w_row   = np.array([vid_to_w.get(int(v), 0.0) for v in vid], np.float32)
    h_row   = np.array([vid_to_l.get(int(v), 0.0) for v in vid], np.float32)
    x      += 0.5 * w_row
    y      += 0.5 * h_row
    xv      = tracks["xVelocity"].astype(np.float32).to_numpy()
    yv      = tracks["yVelocity"].astype(np.float32).to_numpy()
    xa      = tracks["xAcceleration"].astype(np.float32).to_numpy()
    ya      = tracks["yAcceleration"].astype(np.float32).to_numpy()
    lane_id = tracks["laneId"].astype(np.int16).to_numpy()
    dd      = np.array([vid_to_dd.get(int(v), 0) for v in vid], np.int8)
    x_max   = float(np.nanmax(x)) if len(x) else 0.0

    # lat lane center offset (v3 lc_state) – computed in pre-flip coordinates
    # upper_mark[j] = boundary between Lane(j+1) and Lane(j+2)  (j = 0..N_upper-2)
    # lower_mark[j] = boundary between Lane(N_upper+j) and Lane(N_upper+j+1)
    # Lane1 (outermost upper, lid=1) and outermost lower lane are edge lanes → offset=0
    _N_upper = len(upper_mark)
    lat_lane_offset_arr = np.zeros(len(y), np.float32)

    _lid_arr = lane_id.astype(np.int32)

    # lower-direction vehicles (dd==2): j = lid - N_upper - 2
    # (Lane N_upper+1 = central reservation has no track data but consumes one lane ID)
    _mask_lo = (dd == 2)
    _j_lo    = _lid_arr - _N_upper - 2
    _ok_lo   = _mask_lo & (_j_lo >= 0) & (_j_lo < len(lower_mark) - 1)
    lat_lane_offset_arr[_ok_lo] = (
        y[_ok_lo]
        - 0.5 * (lower_mark[_j_lo[_ok_lo]] + lower_mark[_j_lo[_ok_lo] + 1])
    )

    # upper-direction vehicles (dd==1): j = lid - 2  (Lane1 → j=-1 → invalid)
    _mask_up = (dd == 1)
    _j_up    = _lid_arr - 2
    _ok_up   = _mask_up & (_j_up >= 0) & (_j_up < len(upper_mark) - 1)
    lat_lane_offset_arr[_ok_up] = (
        y[_ok_up]
        - 0.5 * (upper_mark[_j_up[_ok_up]] + upper_mark[_j_up[_ok_up] + 1])
    )

    # maybe_flip negates y for upper vehicles → negate lco to match
    lat_lane_offset_arr[dd == 1] *= -1.0

    # lane width array (v4 lc_state용 lco_norm 계산)
    lat_lane_width_arr = np.full(len(y), 3.75, np.float32)
    lat_lane_width_arr[_ok_lo] = np.abs(lower_mark[_j_lo[_ok_lo] + 1] - lower_mark[_j_lo[_ok_lo]])
    lat_lane_width_arr[_ok_up] = np.abs(upper_mark[_j_up[_ok_up] + 1] - upper_mark[_j_up[_ok_up]])

    if cfg.normalize_upper_xy:
        x, y, xv, yv, xa, ya, lane_id = maybe_flip(
            x, y, xv, yv, xa, ya, lane_id, dd, C_y, x_max, upper_mm
        )

    x_min = float(np.nanmin(x)) if x.size else 0.0
    y_min = float(np.nanmin(y)) if y.size else 0.0
    x = (x - x_min).astype(np.float32)
    y = (y - y_min).astype(np.float32)
    if len(upper_center): upper_center = (upper_center - y_min).astype(np.float32)
    if len(lower_center): lower_center = (lower_center - y_min).astype(np.float32)

    per_vid_rows:         Dict[int, np.ndarray]     = {}
    per_vid_frame_to_row: Dict[int, Dict[int, int]] = {}
    for v, idxs in tracks.groupby("id").indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame[idxs])]
        per_vid_rows[int(v)] = idxs
        per_vid_frame_to_row[int(v)] = {int(fr): int(r)
                                        for fr, r in zip(frame[idxs], idxs)}

    lane_change = np.zeros(len(tracks), np.float32)
    for v, idxs in per_vid_rows.items():
        if len(idxs) < 2: continue
        l   = lane_id[idxs].astype(np.int32)
        chg = l[1:] != l[:-1]
        if np.any(chg):
            lane_change[idxs[1:][chg]] = 1.0

    # ── per-dd sorted lane IDs (conditional slot weights 용) ─────────────────
    lane_ids_per_dd_rec: Dict[int, List[int]] = {}
    if cfg.slot_importance_conditional:
        for dd_val in [1, 2]:
            lids = sorted(set(int(x) for x in lane_id[dd == dd_val] if x > 0))
            lane_ids_per_dd_rec[dd_val] = lids

    nb_ids_all = np.stack(
        [tracks[c].astype(np.int32).to_numpy() for c in NEIGHBOR_COLS_8], axis=1
    )

    x_ego_list:      List[np.ndarray] = []
    y_fut_list:      List[np.ndarray] = []
    y_vel_list:      List[np.ndarray] = []
    y_acc_list:      List[np.ndarray] = []
    x_nb_list:       List[np.ndarray] = []
    nb_mask_list:    List[np.ndarray] = []
    x_last_abs_list: List[np.ndarray] = []
    trackid_list:    List[int] = []
    t0_list:         List[int] = []

    for v, idxs in per_vid_rows.items():
        frs = frame[idxs]
        if len(frs) < (T + Tf) * step:
            continue
        fr_set    = set(map(int, frs.tolist()))
        start_min = int(frs[0]  + (T - 1) * step)
        end_max   = int(frs[-1] - Tf       * step)
        if start_min > end_max:
            continue

        t0_frame = start_min
        while t0_frame <= end_max:
            hist_frames = [t0_frame - (T - 1 - i) * step for i in range(T)]
            fut_frames  = [t0_frame + (i + 1)     * step for i in range(Tf)]

            if not all(hf in fr_set for hf in hist_frames) or \
               not all(ff in fr_set for ff in fut_frames):
                t0_frame += stride * step
                continue

            ego_rows = [per_vid_frame_to_row[v][hf] for hf in hist_frames]
            fut_rows = [per_vid_frame_to_row[v][ff] for ff in fut_frames]

            ex  = x[ego_rows];  ey  = y[ego_rows]
            exv = xv[ego_rows]; eyv = yv[ego_rows]
            exa = xa[ego_rows]; eya = ya[ego_rows]

            ego_lane_arr = lane_id[ego_rows].astype(np.int32)

            # ── conditional slot weight context (computed once per sample) ────
            _lc_lane_lv: int       = -1
            _lc_frame_ti: Optional[int] = None
            _lc_type: int          = -1
            if cfg.slot_importance_conditional and cfg.slot_importance_alpha > 0.0:
                ego_dd = vid_to_dd.get(v, 2)
                _lc_lane_lv, _lc_frame_ti, _lc_type = _ego_lc_context(
                    ego_lane_arr, ego_dd, lane_ids_per_dd_rec, cfg.normalize_upper_xy
                )

            # ── ego-centric normalisation: last history frame as origin ───────
            ref_x = float(ex[-1])
            ref_y = float(ey[-1])

            x_ego = np.stack(
                [ex - ref_x, ey - ref_y, exv, eyv, exa, eya], axis=1
            ).astype(np.float32)

            y_fut = np.stack([x[fut_rows] - ref_x, y[fut_rows] - ref_y], axis=1).astype(np.float32)
            y_vel = np.stack([xv[fut_rows], yv[fut_rows]], axis=1).astype(np.float32)
            y_acc = np.stack([xa[fut_rows], ya[fut_rows]], axis=1).astype(np.float32)

            x_nb      = np.zeros((T, K, NB_DIM), np.float32)
            nb_mask   = np.zeros((T, K), bool)
            len_ego   = float(vid_to_w.get(v, 0.0))

            for ti, hf in enumerate(hist_frames):
                ego_vec = np.array([ex[ti], ey[ti], exv[ti], eyv[ti], exa[ti], eya[ti]], np.float32)
                ids8    = nb_ids_all[ego_rows[ti]]

                for ki in range(K):
                    nid = int(ids8[ki])
                    if nid <= 0: continue
                    rm = per_vid_frame_to_row.get(nid)
                    if rm is None: continue
                    r = rm.get(int(hf))
                    if r is None: continue

                    nb_vec = np.array([x[r], y[r], xv[r], yv[r], xa[r], ya[r]], np.float32)
                    rel    = nb_vec - ego_vec
                    if cfg.non_relative:
                        x_nb[ti, ki, 0:6] = np.array(
                            [x[r] - ref_x, y[r] - ref_y, xv[r], yv[r], xa[r], ya[r]], np.float32
                        )
                    else:
                        x_nb[ti, ki, 0:6] = rel
                    nb_mask[ti, ki]   = True

                    # ── lc_state v4: lco_norm 기반 경계 판단 + slot별 방향 결정
                    # lc_state itself is only used to derive s_y; it is not stored.
                    nb_lat_v  = float(yv[r])
                    nb_lco    = float(lat_lane_offset_arr[r])
                    nb_lw     = float(lat_lane_width_arr[r])
                    nb_lco_norm = nb_lco / (nb_lw * 0.5) if nb_lw > 0.5 else 0.0
                    if abs(nb_lco_norm) <= 0.5:
                        lc_state = 1.0
                    elif ki < 2:   # same lane
                        lc_state = 0.0 if nb_lco_norm * nb_lat_v < 0 else 2.0
                    elif ki < 5:   # left lane (slots 2,3,4)
                        lc_state = 0.0 if nb_lat_v < 0 else 2.0
                    else:          # right lane (slots 5,6,7)
                        lc_state = 0.0 if nb_lat_v > 0 else 2.0

                    dx  = float(rel[0])
                    dvx = float(rel[2])
                    len_nb   = float(vid_to_w.get(nid, 0.0))
                    half_sum = 0.5 * (len_ego + len_nb)
                    if dx >= 0:  # nb ahead: gap = x_rear_nb - x_front_ego
                        gap        = abs(dx - half_sum)
                        denom_base = dvx
                    else:        # nb behind: gap = x_rear_ego - x_front_nb
                        gap        = abs(-dx - half_sum)
                        denom_base = -dvx
                    denom = denom_base
                    if abs(denom) < 1e-6:
                        denom = 1e-6 if denom >= 0 else -1e-6
                    lit = gap / denom
                    nb_class   = vid_to_class.get(nid, "Car")
                    nb_phys_l  = vid_to_w.get(nid, 0.0)   # CSV width = physical length
                    nb_phys_w  = vid_to_l.get(nid, 0.0)   # CSV height = physical width
                    size_bin, _ = _volume_bin(nb_phys_l, nb_phys_w, nb_class)
                    s_x        = _lit_to_lis(lit, cfg.lis_mode)
                    delta_lane = float(abs(int(lane_id[r]) - int(ego_lane_arr[ti])))
                    s_y        = float(np.sqrt(lc_state ** 2 + delta_lane ** 2))
                    i_total    = compute_importance(
                        s_x, s_y, cfg.lambda_x, cfg.lambda_y, cfg.alpha, cfg.beta
                    )

                    # ── slot importance boost: I_new = I * (1 + alpha * w_slot) ──
                    if cfg.slot_importance_alpha > 0.0:
                        if cfg.slot_importance_conditional:
                            w_slot = _get_slot_weight(ki, ti, _lc_lane_lv, _lc_frame_ti, _lc_type)
                        else:
                            w_slot = SLOT_WEIGHTS[ki]
                        i_total = min(
                            i_total * (1.0 + cfg.slot_importance_alpha * w_slot),
                            1.0,
                        )

                    x_nb[ti, ki, 6] = s_x
                    x_nb[ti, ki, 7] = s_y
                    x_nb[ti, ki, 8] = size_bin
                    x_nb[ti, ki, 9] = i_total

                # ── top-N gate (applied after all slots are filled) ────
                if cfg.gate_topn > 0:
                    _apply_topn_gate(x_nb[ti], nb_mask[ti], cfg.gate_topn)

            x_ego_list.append(x_ego)
            y_fut_list.append(y_fut)
            y_vel_list.append(y_vel)
            y_acc_list.append(y_acc)
            x_nb_list.append(x_nb)
            nb_mask_list.append(nb_mask)
            x_last_abs_list.append(np.array([ref_x, ref_y], np.float32))
            trackid_list.append(int(v))
            t0_list.append(int(t0_frame))

            t0_frame += stride * step

    if not x_ego_list:
        print(f"  [WARN] {rec_id}: no samples produced.")
        return None

    n_kept = len(x_ego_list)
    return {
        "x_ego":       _safe_float(np.stack(x_ego_list,    0)),
        "y":           _safe_float(np.stack(y_fut_list,     0)),
        "y_vel":       _safe_float(np.stack(y_vel_list,     0)),
        "y_acc":       _safe_float(np.stack(y_acc_list,     0)),
        "x_nb":        _safe_float(np.stack(x_nb_list,      0)),
        "nb_mask":     np.stack(nb_mask_list, 0),
        "x_last_abs":  np.stack(x_last_abs_list, 0),
        "recordingId": np.full(n_kept, int(rec_id), dtype=np.int32),
        "trackId":     np.array(trackid_list, np.int32),
        "t0_frame":    np.array(t0_list,      np.int32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE: raw -> mmap
# ─────────────────────────────────────────────────────────────────────────────

def stage_raw2mmap(cfg: Config) -> None:
    import os

    rec_ids = find_recording_ids(cfg.raw_path)
    if not rec_ids:
        raise FileNotFoundError(f"No recordings found in {cfg.raw_path}")
    n_workers = cfg.num_workers if cfg.num_workers > 0 else os.cpu_count()
    print(f"[Stage] raw -> mmap  |  {len(rec_ids)} recordings  |  "
          f"workers={n_workers}  |  mmap_path={cfg.mmap_path}")
    print(f"  lis_mode        : {cfg.lis_mode}")
    print(f"  importance     : lambda_x={cfg.lambda_x}  lambda_y={cfg.lambda_y}  "
          f"alpha={cfg.alpha}  beta={cfg.beta}")
    if cfg.slot_importance_alpha > 0.0:
        cond_str = "conditional (lane-level / pre-LC / post-LC)" if cfg.slot_importance_conditional else "global SLOT_WEIGHTS"
        print(f"  slotImportance  : alpha={cfg.slot_importance_alpha}  weights={cond_str}  "
              f"I_new = min(I * (1 + {cfg.slot_importance_alpha} * w_slot), 1.0)")
    if cfg.gate_topn > 0:
        print(f"  gate_topn       : keep top {cfg.gate_topn} neighbors by I per history frame")

    # ── pass 1: process all recordings in parallel ────────────────────────────
    bufs: List[Dict[str, np.ndarray]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(_recording_to_buf, cfg, rid): rid for rid in rec_ids}
        for fut in tqdm(concurrent.futures.as_completed(futs),
                        total=len(rec_ids), desc="Processing recordings"):
            result = fut.result()
            if result is not None:
                bufs.append(result)

    if not bufs:
        raise RuntimeError("No samples produced from any recording.")

    total = sum(b["x_ego"].shape[0] for b in bufs)
    print(f"  total samples  : {total:,}")

    if cfg.dry_run:
        print("[DRY RUN] No files written.")
        return

    # ── allocate memmaps ──────────────────────────────────────────────────────
    out = cfg.mmap_path
    out.mkdir(parents=True, exist_ok=True)

    s0 = bufs[0]
    fp = {
        "x_ego":      open_memmap(out / "x_ego.npy",      "w+", "float32", (total, *s0["x_ego"].shape[1:])),
        "y":          open_memmap(out / "y.npy",           "w+", "float32", (total, *s0["y"].shape[1:])),
        "y_vel":      open_memmap(out / "y_vel.npy",       "w+", "float32", (total, *s0["y_vel"].shape[1:])),
        "y_acc":      open_memmap(out / "y_acc.npy",       "w+", "float32", (total, *s0["y_acc"].shape[1:])),
        "x_nb":       open_memmap(out / "x_nb.npy",        "w+", "float32", (total, *s0["x_nb"].shape[1:])),
        "nb_mask":    open_memmap(out / "nb_mask.npy",     "w+", "bool",    (total, *s0["nb_mask"].shape[1:])),
        "x_last_abs": open_memmap(out / "x_last_abs.npy",  "w+", "float32", (total, 2)),
    }
    meta_rec   = np.zeros(total, np.int32)
    meta_track = np.zeros(total, np.int32)
    meta_frame = np.zeros(total, np.int32)

    # ── pass 2: write buffers -> mmap (sequential) ────────────────────────────
    cursor = 0
    for buf in tqdm(bufs, desc="Writing mmap"):
        n   = buf["x_ego"].shape[0]
        end = cursor + n

        for key in ["x_ego", "y", "y_vel", "y_acc", "x_nb", "nb_mask", "x_last_abs"]:
            fp[key][cursor:end] = buf[key]

        meta_rec[cursor:end]   = buf["recordingId"]
        meta_track[cursor:end] = buf["trackId"]
        meta_frame[cursor:end] = buf["t0_frame"]

        cursor = end

    # ── flush + save meta ─────────────────────────────────────────────────────
    for arr in fp.values():
        arr.flush()
    np.save(out / "meta_recordingId.npy", meta_rec)
    np.save(out / "meta_trackId.npy",     meta_track)
    np.save(out / "meta_frame.npy",       meta_frame)

    print(f"  [OK] mmap saved -> {out}")
    print(f"  [INFO] Stats will be computed automatically on first train/evaluate run.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="HighD preprocessing pipeline  (raw CSV -> mmap)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir",    default="data/highD", help="Base data directory")
    ap.add_argument("--raw_dir",     default="raw",        help="Raw CSV subdir under data_dir")
    ap.add_argument("--mmap_dir",    default="mmap",       help="Mmap output subdir under data_dir")
    ap.add_argument("--num_workers", type=int, default=0,  help="Worker processes (0 = os.cpu_count())")

    # recording
    ap.add_argument("--target_hz",          type=float, default=3.0)
    ap.add_argument("--history_sec",        type=float, default=2.0)
    ap.add_argument("--future_sec",         type=float, default=5.0)
    ap.add_argument("--stride_sec",         type=float, default=1.0)
    ap.add_argument("--normalize_upper_xy", action="store_true", default=True)

    # LIS
    ap.add_argument("--lis_mode", default="7",
                    choices=["3", "5", "7", "9"],
                    help=(
                        "LIS binning mode for s_x: "
                        "3=lit 3-bin {-1,0,1} | "
                        "5=lit 5-bin {-2,...,2} | "
                        "7=lit 7-bin {-3,...,3} | "
                        "9=lit 9-bin {-4,...,4}"
                    ))

    # importance
    ap.add_argument("--lambda_x", type=float, default=0.1)
    ap.add_argument("--lambda_y", type=float, default=0.1)
    ap.add_argument("--alpha",    type=float, default=1.5)
    ap.add_argument("--beta",     type=float, default=2.0)

    # gate
    ap.add_argument("--gate_topn", type=int, default=0,
                    help="Top-N gate: keep up to N slots with highest I; "
                         "tie-break by slot priority 0>2>5>1>4>7>3>6. 0 = disabled")
    ap.add_argument("--slotImportance", type=float, default=0.0,
                    dest="slot_importance_alpha",
                    help="Slot importance boost alpha: I_new = min(I * (1 + alpha * w_slot), 1.0). "
                         "w_slot = empirical mean I per slot. 0.0 = disabled (default)")
    ap.add_argument("--slotImportanceConditional", action="store_true", default=False,
                    dest="slot_importance_conditional",
                    help="Use lane-level / pre-LC / post-LC specific slot weights "
                         "instead of the global SLOT_WEIGHTS. Requires --slotImportance > 0.")

    ap.add_argument("--non_relative", action="store_true", default=False,
                    help="x_nb[0:6] = neighbor's abs values in globally-shifted frame "
                         "(instead of ego-relative differences). "
                         "s_x/s_y/I always use relative/context values.")

    ap.add_argument("--dry_run", action="store_true")

    a = ap.parse_args()
    return Config(
        data_dir = Path(a.data_dir),
        raw_dir  = Path(a.raw_dir),
        mmap_dir = Path(a.mmap_dir),
        target_hz          = a.target_hz,
        history_sec        = a.history_sec,
        future_sec         = a.future_sec,
        stride_sec         = a.stride_sec,
        normalize_upper_xy = a.normalize_upper_xy,
        lis_mode = a.lis_mode,
        lambda_x = a.lambda_x,
        lambda_y = a.lambda_y,
        alpha    = a.alpha,
        beta     = a.beta,
        gate_topn = a.gate_topn,
        slot_importance_alpha        = a.slot_importance_alpha,
        slot_importance_conditional  = a.slot_importance_conditional,
        non_relative = a.non_relative,
        dry_run     = a.dry_run,
        num_workers = a.num_workers,
    )


def main() -> None:
    cfg = parse_args()
    stage_raw2mmap(cfg)


if __name__ == "__main__":
    main()
