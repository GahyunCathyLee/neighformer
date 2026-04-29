#!/usr/bin/env python3
"""
preprocess_exid.py  —  exiD preprocessing pipeline  (raw CSV → mmap)

STAGE raw2mmap :  exiD raw CSV  →  memory-mapped arrays

stats 계산은 train.py / evaluate.py 실행 시 src/stats.py 가 자동으로 수행합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature schema  (highD preprocess.py 와 동일한 output format)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x_ego  : (N, T, 6)
    [x, y, xV, yV, xA, yA]  — ego-centric normalised frame
                               (last history frame → origin, heading → +x)

x_nb   : (N, T, K, 10)   ego-relative neighbor features
    idx  0  dx        longitudinal distance  (key-point based, ego local frame)
    idx  1  dy        lateral distance       (key-point based, ego local frame)
    idx  2  dvx       relative longitudinal velocity  (ego local frame)
    idx  3  dvy       relative lateral velocity       (ego local frame)
    idx  4  dax       relative longitudinal acceleration
    idx  5  day       relative lateral acceleration
    idx  6  s_x       longitudinal interaction state (existing LIS)
    idx  7  s_y       sqrt(lc_state^2 + delta_lane^2)
    idx  8  dim       vehicle size bin (0~4) based on width*length*height_est
    idx  9  I         exp(-lambda_x*|s_x|^alpha - lambda_y*s_y^beta)

    LIT is used internally to compute s_x. With --non_relative:
    idx 0-5 hold the neighbor's own values in the
    normalised reference frame instead of ego-relative differences.
    s_x/s_y/I always use relative/context values.

y          : (N, Tf, 2)     future [x, y]  — ego-centric normalised frame
y_vel      : (N, Tf, 2)     future [xV, yV] — normalised frame
y_acc      : (N, Tf, 2)     future [xA, yA] — normalised frame
nb_mask    : (N, T, K)      bool - True if neighbor exists
x_last_abs : (N, 2)         last absolute (x, y) of ego history (pre-normalisation)
meta_recordingId / trackId / t0_frame : (N,)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
highD vs exiD 컬럼 매핑
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  좌표     : x/y (bbox 좌상단, +0.5*size)  → xCenter/yCenter (이미 center)
  속도     : xVelocity/yVelocity           → lonVelocity/latVelocity
  가속도   : xAcceleration/yAcceleration   → lonAcceleration/latAcceleration
  차선     : laneId                        → laneletId
  이웃     : precedingId/followingId/...   → leadId/rearId/...
  차량크기 : tracksMeta width/height       → tracks 행 자체 width/length
  방향정규 : drivingDirection flip 필요    → 불필요 (exiD는 이미 통일된 좌표계)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIS (Longitudinal Interaction State) modes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    '3'    : {-1,...,1}      lit 3-bin
    '5'    : {-2,...,2}      lit 5-bin
    '7'    : {-3,...,3}      lit 7-bin
    '9'    : {-4,...,4}      lit 9-bin

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
import math
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

# exiD neighbor columns (highD의 preceding/following 계열 → exiD의 lead/rear 계열)
NEIGHBOR_COLS_8 = [
    "leadId",            # 0  ← highD: precedingId
    "rearId",            # 1  ← highD: followingId
    "leftLeadId",        # 2  ← highD: leftPrecedingId
    "leftAlongsideId",   # 3  ← highD: leftAlongsideId  (동일)
    "leftRearId",        # 4  ← highD: leftFollowingId
    "rightLeadId",       # 5  ← highD: rightPrecedingId
    "rightAlongsideId",  # 6  ← highD: rightAlongsideId (동일)
    "rightRearId",       # 7  ← highD: rightFollowingId
]

EGO_DIM = 6    # x, y, xV, yV, xA, yA
NB_DIM  = 10   # dx, dy, dvx, dvy, dax, day, s_x, s_y, dim, I
K       = 8    # neighbor slots

# Slot priority for top-N gate tie-breaking: 0 > 2 > 5 > 1 > 4 > 7 > 3 > 6
_TOPN_SLOT_PRIORITY = {s: r for r, s in enumerate([0, 2, 5, 1, 4, 7, 3, 6])}

# Empirical slot weights (mean I per slot, from highD dataset analysis).
# Order: lead, rear, leftLead, leftAlongside, leftRear,
#        rightLead, rightAlongside, rightRear
SLOT_WEIGHTS = [0.4944, 0.0411, 0.0935, 0.0074, 0.0002, 0.5559, 0.0000, 0.1179]

# Conditional slot weights copied from the highD preprocessing contract.
SLOT_WEIGHTS_BY_LANE_LEVEL = [
    [0.4255, 0.0336, 0.0000, 0.0000, 0.0000, 0.4574, 0.0119, 0.1190],
    [0.4805, 0.0002, 0.0000, 0.0000, 0.0000, 0.3803, 0.0234, 0.1839],
    [0.4784, 0.0373, 0.3344, 0.0343, 0.2050, 0.0000, 0.0000, 0.0000],
]

SLOT_WEIGHTS_PRE_LC = [
    [0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.6253, 0.2663, 0.3117],
    [0.0072, 0.0263, 0.0006, 0.0000, 0.0000, 0.3970, 0.3776, 0.5494],
    [0.0183, 0.1326, 0.6745, 0.5179, 0.2365, 0.0000, 0.0000, 0.0000],
    [0.0381, 0.0233, 0.5755, 0.3548, 0.4799, 0.0000, 0.0000, 0.0000],
]

SLOT_WEIGHTS_POST_LC = [
    [0.0460, 0.3983, 0.0000, 0.0023, 0.0762, 0.2338, 0.2022, 0.3281],
    [0.1036, 0.0851, 0.4832, 0.0540, 0.3810, 0.0013, 0.0000, 0.0002],
    [0.6018, 0.3591, 0.0115, 0.0013, 0.0099, 0.1709, 0.0069, 0.0014],
    [0.2618, 0.0000, 0.0036, 0.0000, 0.0000, 0.6545, 0.2032, 0.1449],
]

_LC_TYPE_TO_GROUP: Dict[int, int] = {
    0: 0, 1: 0,
    3: 1, 6: 1,
    2: 2, 7: 2,
    4: 3, 5: 3,
}

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


# exiD 기본 프레임레이트 (recordingMeta에 frameRate 컬럼이 없을 경우 fallback)
EXID_DEFAULT_HZ = 25.0


# ─────────────────────────────────────────────────────────────────────────────
# LIS binning  (highD와 동일)
# ─────────────────────────────────────────────────────────────────────────────

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

_VOLUME_BIN_EDGES = [12.0, 20.0, 90.0, 150.0]


def _volume_bin(phys_length: float, phys_width: float, vehicle_class: str) -> Tuple[float, float]:
    """Return (size bin index 0~4, raw volume m³) for a neighbor vehicle."""
    cls = (vehicle_class or "").strip().lower()
    if cls in {"car", "van"}:
        if phys_length < 4.5:
            height = 1.45
        elif phys_length < 5.0:
            height = 1.70
        else:
            height = 1.90
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
    data_dir: Path = Path("data/exiD")
    raw_dir:  Path = Path("raw")
    mmap_dir: Path = Path("mmap")

    @property
    def raw_path(self) -> Path:
        return self.data_dir / self.raw_dir

    @property
    def mmap_path(self) -> Path:
        return self.data_dir / self.mmap_dir

    # recording
    target_hz:   float = 3.0
    history_sec: float = 2.0
    future_sec:  float = 5.0
    stride_sec:  float = 1.0

    # LIS mode
    lis_mode: str = '3'   # '3' | '5' | '7' | '9'

    # importance params: I = exp(-lambda_x*|s_x|^alpha - lambda_y*s_y^beta)
    lambda_x: float = 0.1
    lambda_y: float = 0.1
    alpha:    float = 1.5
    beta:     float = 2.0

    # importance gate
    gate_topn: int = 0     # >0 = keep top-N slots by I; 0 = disabled

    # slot importance: I_new = min(I * (1 + alpha * w_slot), 1.0); 0.0 = disabled
    slot_importance_alpha: float = 0.0
    slot_importance_conditional: bool = False

    # exiD-specific: VRU 필터링
    drop_vru: bool = True     # VRU (motorcycle/bicycle/pedestrian) 윈도우 제거

    # neighbor feature mode
    non_relative: bool = False  # True → x_nb[0:6] = abs nb values in normalised frame

    # output / execution
    dry_run:     bool = False
    num_workers: int  = 0


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

VRU_CLASSES = {"motorcycle", "bicycle", "pedestrian"}


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

def _lane_id_to_level(lid: int, sorted_lids: List[int]) -> int:
    """lane_id → lane_level (0=leftmost-ish, 1=middle, 2=rightmost-ish).

    exiD lanelet IDs are not as globally regular as highD lane IDs. This helper
    preserves the highD conditional-weight interface using the per-recording
    sorted lanelet IDs as a conservative fallback.
    """
    n = len(sorted_lids)
    if n == 0 or lid not in sorted_lids:
        return -1
    idx = sorted_lids.index(lid)
    if n == 1:
        return 1
    if idx == 0:
        return 0
    if idx == n - 1:
        return 2
    return 1


def _ego_lc_context(
    ego_lane_arr: np.ndarray,
    sorted_lids: List[int],
) -> Tuple[int, Optional[int], int]:
    lc_frame_ti: Optional[int] = None
    lc_type = -1

    for ti in range(1, len(ego_lane_arr)):
        if ego_lane_arr[ti] != ego_lane_arr[ti - 1]:
            lc_frame_ti = ti
            from_lvl = _lane_id_to_level(int(ego_lane_arr[ti - 1]), sorted_lids)
            to_lvl   = _lane_id_to_level(int(ego_lane_arr[ti]),     sorted_lids)
            lc_type  = _LC_TYPE_MAP_LEVEL.get((from_lvl, to_lvl), -1)
            break

    if lc_frame_ti is None:
        lane_level = _lane_id_to_level(int(ego_lane_arr[-1]), sorted_lids)
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
    if lc_frame_ti is not None and lc_type >= 0:
        lc_group = _LC_TYPE_TO_GROUP.get(lc_type, -1)
        if lc_group < 0:
            return SLOT_WEIGHTS[ki]
        if ti < lc_frame_ti:
            return SLOT_WEIGHTS_PRE_LC[lc_group][ki]
        return SLOT_WEIGHTS_POST_LC[lc_group][ki]
    if 0 <= lane_level <= 2:
        return SLOT_WEIGHTS_BY_LANE_LEVEL[lane_level][ki]
    return SLOT_WEIGHTS[ki]


# ─────────────────────────────────────────────────────────────────────────────
# Raw CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_recording_ids(raw_dir: Path) -> List[str]:
    """*_tracks.csv 파일에서 recording ID 목록을 추출합니다."""
    ids = [re.match(r"(\d+)_tracks\.csv$", p.name).group(1)
           for p in raw_dir.glob("*_tracks.csv")
           if re.match(r"(\d+)_tracks\.csv$", p.name)]
    return sorted(set(ids))


def get_frame_rate(rec_meta: pd.DataFrame) -> float:
    """recordingMeta에서 frameRate를 읽습니다. 없으면 EXID_DEFAULT_HZ를 반환합니다."""
    if "frameRate" in rec_meta.columns:
        return float(rec_meta.loc[0, "frameRate"])
    return EXID_DEFAULT_HZ


def get_class_map(trk_meta: pd.DataFrame) -> Dict[int, str]:
    """tracksMeta의 class 컬럼에서 trackId → 정규화된 클래스명 딕셔너리를 반환합니다."""
    class_map: Dict[int, str] = {}
    if "trackId" not in trk_meta.columns or "class" not in trk_meta.columns:
        return class_map
    for tid, cls in zip(trk_meta["trackId"].astype(int).tolist(),
                        trk_meta["class"].astype(str).tolist()):
        s = (cls or "").strip().lower()
        class_map[int(tid)] = s if s not in ("", "nan", "null") else "other"
    return class_map


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rot2d(vx: float, vy: float, h_rad: float) -> Tuple[float, float]:
    """Project (vx, vy) onto a local frame whose heading is h_rad from global +x.
    Returns (longitudinal, lateral) components."""
    c, s = math.cos(h_rad), math.sin(h_rad)
    return c * vx + s * vy, -s * vx + c * vy


def _norm_pos(gx: float, gy: float,
              ref_x: float, ref_y: float, ref_hdg_rad: float) -> Tuple[float, float]:
    """Translate and rotate (gx, gy) into the ego-centric normalised frame.
    ref_hdg_rad is the ego heading at the reference frame (last history step)."""
    return _rot2d(gx - ref_x, gy - ref_y, ref_hdg_rad)


def _local_to_norm_frame(lon: float, lat: float,
                          veh_hdg_rad: float,
                          ref_hdg_rad: float) -> Tuple[float, float]:
    """Rotate a vehicle-local (lon, lat) vector into the normalised reference frame.
    lon/lat are in the vehicle's own heading frame; veh_hdg_rad is the vehicle's
    heading; ref_hdg_rad is the reference heading (ego last history frame)."""
    delta = veh_hdg_rad - ref_hdg_rad
    c, s = math.cos(delta), math.sin(delta)
    return c * lon - s * lat, s * lon + c * lat


def _rel_vel_ego_frame(nb_lon: float, nb_lat: float, nb_hdg: float,
                        ego_lon: float, ego_lat: float, ego_hdg: float,
                        ) -> Tuple[float, float]:
    """Relative velocity (nb - ego) in ego's instantaneous local frame.
    Inputs are vehicle-local lon/lat values and headings in radians."""
    nb_vx  = nb_lon  * math.cos(nb_hdg)  - nb_lat  * math.sin(nb_hdg)
    nb_vy  = nb_lon  * math.sin(nb_hdg)  + nb_lat  * math.cos(nb_hdg)
    ego_vx = ego_lon * math.cos(ego_hdg) - ego_lat * math.sin(ego_hdg)
    ego_vy = ego_lon * math.sin(ego_hdg) + ego_lat * math.cos(ego_hdg)
    return _rot2d(nb_vx - ego_vx, nb_vy - ego_vy, ego_hdg)


def _vehicle_front_rear_pts(
    cx: float, cy: float, hdg_rad: float, w: float, l: float
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Compute front3 and rear3 key points of a vehicle in global frame.
    front3 = [front_left, front_mid, front_right]
    rear3  = [rear_left,  rear_mid,  rear_right ]
    hdg_rad is the heading in radians.
    """
    ux, uy = math.cos(hdg_rad), math.sin(hdg_rad)    # longitudinal unit
    lx, ly = -math.sin(hdg_rad), math.cos(hdg_rad)   # lateral unit
    hl, hw = l / 2.0, w / 2.0
    fm = (cx + hl * ux, cy + hl * uy)   # front_mid
    rm = (cx - hl * ux, cy - hl * uy)   # rear_mid
    front3 = [
        (fm[0] + hw * lx, fm[1] + hw * ly),   # front_left
        fm,                                     # front_mid
        (fm[0] - hw * lx, fm[1] - hw * ly),   # front_right
    ]
    rear3 = [
        (rm[0] + hw * lx, rm[1] + hw * ly),   # rear_left
        rm,                                     # rear_mid
        (rm[0] - hw * lx, rm[1] - hw * ly),   # rear_right
    ]
    return front3, rear3


def _nb_dxdy(
    slot: int,
    ego_cx: float, ego_cy: float, ego_hdg: float, ego_w: float, ego_l: float,
    nb_cx:  float, nb_cy:  float, nb_hdg:  float, nb_w:  float, nb_l:  float,
) -> Tuple[float, float]:
    """Compute (dx, dy) in ego's local frame using per-slot key-point rules.

    alongside (3, 6): center-to-center projection; dy adjusted by
                      -sign(dy) * 0.5 * (ego_w + nb_w)
    lead      (0,2,5): closest pair from ego front3 × nb rear3
    rear      (1,4,7): closest pair from ego rear3  × nb front3

    The returned dx/dy vector is from the selected ego key point to the
    selected neighbor key point, projected onto ego's local frame.
    """
    ego_front, ego_rear = _vehicle_front_rear_pts(ego_cx, ego_cy, ego_hdg, ego_w, ego_l)
    nb_front,  nb_rear  = _vehicle_front_rear_pts(nb_cx,  nb_cy,  nb_hdg,  nb_w,  nb_l)

    if slot in (3, 6):   # alongside
        dx, dy = _rot2d(nb_cx - ego_cx, nb_cy - ego_cy, ego_hdg)
        sign_dy = 1.0 if dy >= 0.0 else -1.0
        dy -= sign_dy * 0.5 * (ego_w + nb_w)
        return dx, dy

    if slot in (0, 2, 5):   # lead: ego front3 × nb rear3
        ego_pts, nb_pts = ego_front, nb_rear
    else:                    # rear: ego rear3 × nb front3  (slots 1, 4, 7)
        ego_pts, nb_pts = ego_rear, nb_front

    best_d  = math.inf
    best_ep = ego_pts[1]
    best_np = nb_pts[1]
    for ep in ego_pts:
        for np_ in nb_pts:
            d = math.hypot(ep[0] - np_[0], ep[1] - np_[1])
            if d < best_d:
                best_d, best_ep, best_np = d, ep, np_
    return _rot2d(best_np[0] - best_ep[0], best_np[1] - best_ep[1], ego_hdg)


# ─────────────────────────────────────────────────────────────────────────────
# Per-recording processing  (raw CSV -> list of sample dicts)
# ─────────────────────────────────────────────────────────────────────────────

def _recording_to_buf(cfg: Config, rec_id: str) -> Optional[Dict[str, np.ndarray]]:
    raw_dir  = cfg.raw_path
    rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    trk_meta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv", low_memory=False)

    # ── 프레임레이트 / 윈도우 파라미터 ────────────────────────────────────────
    frame_rate = get_frame_rate(rec_meta)
    step   = max(1, int(round(frame_rate / cfg.target_hz)))
    T      = int(round(cfg.history_sec  * cfg.target_hz))
    Tf     = int(round(cfg.future_sec   * cfg.target_hz))
    stride = max(1, int(round(cfg.stride_sec * cfg.target_hz)))

    # ── 필수 컬럼 체크 / 기본값 주입 ─────────────────────────────────────────
    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns:
            tracks[c] = -1

    for src in ["lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration"]:
        if src not in tracks.columns:
            tracks[src] = 0.0

    if "laneletId" not in tracks.columns:
        tracks["laneletId"] = -1

    # ── VRU 클래스 맵 ─────────────────────────────────────────────────────────
    class_map = get_class_map(trk_meta)

    # ── 차량 크기 배열 ────────────────────────────────────────────────────────
    has_width  = "width"  in tracks.columns
    has_length = "length" in tracks.columns

    # ── NumPy 배열 추출 ───────────────────────────────────────────────────────
    tracks = tracks.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)

    frame   = tracks["frame"].astype(np.int32).to_numpy()
    vid     = tracks["trackId"].astype(np.int32).to_numpy()

    x       = tracks["xCenter"].astype(np.float32).to_numpy().copy()
    y       = tracks["yCenter"].astype(np.float32).to_numpy().copy()
    xv      = tracks["lonVelocity"].astype(np.float32).to_numpy()
    yv      = tracks["latVelocity"].astype(np.float32).to_numpy()
    xa      = tracks["lonAcceleration"].astype(np.float32).to_numpy()
    ya      = tracks["latAcceleration"].astype(np.float32).to_numpy()
    lane_id = tracks["laneletId"].fillna(-1).astype(np.int32).to_numpy()

    # latLaneCenterOffset (first semicolon-separated value per row)
    if "latLaneCenterOffset" in tracks.columns:
        _lco_s = tracks["latLaneCenterOffset"].astype(str).str.strip().str.split(";").str[0]
        lat_lane_offset_arr = pd.to_numeric(_lco_s, errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
    else:
        lat_lane_offset_arr = np.zeros(len(tracks), np.float32)

    # lane width array (v4 lc_state용 lco_norm 계산)
    if "laneWidth" in tracks.columns:
        _lw_s = tracks["laneWidth"].astype(str).str.strip().str.split(";").str[0]
        lat_lane_width_arr = pd.to_numeric(_lw_s, errors="coerce").fillna(3.5).astype(np.float32).to_numpy()
    else:
        lat_lane_width_arr = np.full(len(tracks), 3.5, np.float32)

    # heading (degrees in CSV → radians for computation)
    heading_deg = (tracks["heading"].astype(np.float32).to_numpy()
                   if "heading" in tracks.columns
                   else np.zeros(len(tracks), np.float32))
    heading_rad = np.deg2rad(heading_deg).astype(np.float32)

    # 차량 크기 배열 (per-row)
    width_arr  = _safe_float(tracks["width"].to_numpy(np.float32),  0.0) if has_width  else np.zeros(len(tracks), np.float32)
    length_arr = _safe_float(tracks["length"].to_numpy(np.float32), 0.0) if has_length else np.zeros(len(tracks), np.float32)

    # ── 원점 이동 (recording 내 min_x, min_y 기준) ───────────────────────────
    x_min = float(np.nanmin(x)) if x.size else 0.0
    y_min = float(np.nanmin(y)) if y.size else 0.0
    x = (x - x_min).astype(np.float32)
    y = (y - y_min).astype(np.float32)

    # ── per-vehicle row/frame 인덱스 구축 ────────────────────────────────────
    per_vid_rows:         Dict[int, np.ndarray]     = {}
    per_vid_frame_to_row: Dict[int, Dict[int, int]] = {}
    for v, idxs in tracks.groupby("trackId").indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame[idxs])]
        per_vid_rows[int(v)] = idxs
        per_vid_frame_to_row[int(v)] = {int(fr): int(r)
                                        for fr, r in zip(frame[idxs], idxs)}

    # ── 차량 크기 딕셔너리 (첫 번째 행 기준) ─────────────────────────────────
    vid_to_w: Dict[int, float] = {}
    vid_to_l: Dict[int, float] = {}
    for v, idxs in per_vid_rows.items():
        r0 = int(idxs[0])
        vid_to_w[int(v)] = float(width_arr[r0])
        vid_to_l[int(v)] = float(length_arr[r0])

    # ── per-vehicle, per-frame heading lookup (radians) ───────────────────────
    per_vid_frame_to_hdg: Dict[int, Dict[int, float]] = {
        int(v): {int(frame[r]): float(heading_rad[r]) for r in idxs}
        for v, idxs in per_vid_rows.items()
    }

    # ── VRU 판별 함수 ─────────────────────────────────────────────────────────
    def is_vru(tid: int) -> bool:
        return class_map.get(int(tid), "other") in VRU_CLASSES

    # ── neighbor ID 배열 (N, 8) ───────────────────────────────────────────────
    nb_id_cols = []
    for c in NEIGHBOR_COLS_8:
        s = tracks[c].astype(str).str.strip().str.split(";").str[0]
        nb_id_cols.append(pd.to_numeric(s, errors="coerce").fillna(-1).astype(np.int32).to_numpy())
    nb_ids_all = np.stack(nb_id_cols, axis=1)   # (rows, 8)

    lane_ids_rec: List[int] = []
    if cfg.slot_importance_conditional:
        lane_ids_rec = sorted(set(int(lid) for lid in lane_id if int(lid) >= 0))

    # ── sample 수집 루프 ──────────────────────────────────────────────────────
    x_ego_list:       List[np.ndarray] = []
    y_fut_list:       List[np.ndarray] = []
    y_vel_list:       List[np.ndarray] = []
    y_acc_list:       List[np.ndarray] = []
    x_nb_list:        List[np.ndarray] = []
    nb_mask_list:     List[np.ndarray] = []
    x_last_abs_list:  List[np.ndarray] = []
    trackid_list:     List[int]        = []
    t0_list:          List[int]        = []

    for v, idxs in per_vid_rows.items():
        frs = frame[idxs]
        if len(frs) < (T + Tf) * step:
            continue

        if cfg.drop_vru and is_vru(int(v)):
            continue

        fr_set    = set(map(int, frs.tolist()))
        start_min = int(frs[0]  + (T - 1) * step)
        end_max   = int(frs[-1] - Tf       * step)
        if start_min > end_max:
            continue

        ego_w = float(vid_to_w.get(v, 0.0))
        ego_l = float(vid_to_l.get(v, 0.0))
        hdg_map_ego = per_vid_frame_to_hdg.get(int(v), {})

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

            _lc_lane_lv: int = -1
            _lc_frame_ti: Optional[int] = None
            _lc_type: int = -1
            if cfg.slot_importance_conditional and cfg.slot_importance_alpha > 0.0:
                _lc_lane_lv, _lc_frame_ti, _lc_type = _ego_lc_context(
                    ego_lane_arr, lane_ids_rec
                )

            # ── ego heading per history frame ─────────────────────────────
            ego_hdg_arr = np.array(
                [hdg_map_ego.get(int(hf), 0.0) for hf in hist_frames], np.float32
            )

            # ── normalisation reference: last history frame ───────────────
            ref_x   = float(ex[-1])
            ref_y   = float(ey[-1])
            ref_hdg = float(ego_hdg_arr[-1])   # radians

            # ── ego history: normalise positions + rotate vel/acc ─────────
            ex_n  = np.zeros(T, np.float32)
            ey_n  = np.zeros(T, np.float32)
            exv_n = np.zeros(T, np.float32)
            eyv_n = np.zeros(T, np.float32)
            exa_n = np.zeros(T, np.float32)
            eya_n = np.zeros(T, np.float32)
            for ti in range(T):
                ex_n[ti],  ey_n[ti]  = _norm_pos(
                    float(ex[ti]), float(ey[ti]), ref_x, ref_y, ref_hdg)
                exv_n[ti], eyv_n[ti] = _local_to_norm_frame(
                    float(exv[ti]), float(eyv[ti]), float(ego_hdg_arr[ti]), ref_hdg)
                exa_n[ti], eya_n[ti] = _local_to_norm_frame(
                    float(exa[ti]), float(eya[ti]), float(ego_hdg_arr[ti]), ref_hdg)
            x_ego = np.stack([ex_n, ey_n, exv_n, eyv_n, exa_n, eya_n],
                             axis=1).astype(np.float32)

            # ── future: normalise positions + rotate vel/acc ──────────────
            y_fut_x = np.zeros(Tf, np.float32)
            y_fut_y = np.zeros(Tf, np.float32)
            y_vel_x = np.zeros(Tf, np.float32)
            y_vel_y = np.zeros(Tf, np.float32)
            y_acc_x = np.zeros(Tf, np.float32)
            y_acc_y = np.zeros(Tf, np.float32)
            for fi in range(Tf):
                fr  = fut_rows[fi]
                ff  = fut_frames[fi]
                hdg_fi = float(hdg_map_ego.get(int(ff), ref_hdg))
                y_fut_x[fi], y_fut_y[fi] = _norm_pos(
                    float(x[fr]), float(y[fr]), ref_x, ref_y, ref_hdg)
                y_vel_x[fi], y_vel_y[fi] = _local_to_norm_frame(
                    float(xv[fr]), float(yv[fr]), hdg_fi, ref_hdg)
                y_acc_x[fi], y_acc_y[fi] = _local_to_norm_frame(
                    float(xa[fr]), float(ya[fr]), hdg_fi, ref_hdg)
            y_fut = np.stack([y_fut_x, y_fut_y], axis=1).astype(np.float32)
            y_vel = np.stack([y_vel_x, y_vel_y], axis=1).astype(np.float32)
            y_acc = np.stack([y_acc_x, y_acc_y], axis=1).astype(np.float32)

            x_nb    = np.zeros((T, K, NB_DIM), np.float32)
            nb_mask = np.zeros((T, K), bool)

            for ti, hf in enumerate(hist_frames):
                ego_hdg_ti = float(ego_hdg_arr[ti])
                ids8       = nb_ids_all[ego_rows[ti]]

                for ki in range(K):
                    nid = int(ids8[ki])
                    if nid <= 0:
                        continue
                    rm = per_vid_frame_to_row.get(nid)
                    if rm is None:
                        continue
                    r = rm.get(int(hf))
                    if r is None:
                        continue

                    if cfg.drop_vru and is_vru(nid):
                        continue

                    nb_w   = float(vid_to_w.get(nid, 0.0))
                    nb_l   = float(vid_to_l.get(nid, 0.0))
                    nb_hdg = float(per_vid_frame_to_hdg.get(nid, {}).get(int(hf), ego_hdg_ti))

                    # ── dx, dy: key-point based, ego local frame ──────────
                    dx_key, dy_key = _nb_dxdy(
                        ki,
                        float(ex[ti]), float(ey[ti]), ego_hdg_ti, ego_w, ego_l,
                        float(x[r]),   float(y[r]),   nb_hdg,     nb_w,  nb_l,
                    )

                    # ── relative vel/acc in ego's instantaneous local frame ─
                    dvx_rel, dvy_rel = _rel_vel_ego_frame(
                        float(xv[r]), float(yv[r]), nb_hdg,
                        float(exv[ti]), float(eyv[ti]), ego_hdg_ti,
                    )
                    dax_rel, day_rel = _rel_vel_ego_frame(
                        float(xa[r]), float(ya[r]), nb_hdg,
                        float(exa[ti]), float(eya[ti]), ego_hdg_ti,
                    )

                    # ── feature values (may be replaced in non_relative mode) ─
                    if cfg.non_relative:
                        # neighbor's absolute values in normalised reference frame
                        f_dx,  f_dy  = _norm_pos(
                            float(x[r]),  float(y[r]),  ref_x, ref_y, ref_hdg)
                        f_dvx, f_dvy = _local_to_norm_frame(
                            float(xv[r]), float(yv[r]), nb_hdg, ref_hdg)
                        f_dax, f_day = _local_to_norm_frame(
                            float(xa[r]), float(ya[r]), nb_hdg, ref_hdg)
                    else:
                        f_dx,  f_dy  = dx_key,  dy_key
                        f_dvx, f_dvy = dvx_rel, dvy_rel
                        f_dax, f_day = dax_rel, day_rel

                    x_nb[ti, ki, 0] = f_dx
                    x_nb[ti, ki, 1] = f_dy
                    x_nb[ti, ki, 2] = f_dvx
                    x_nb[ti, ki, 3] = f_dvy
                    x_nb[ti, ki, 4] = f_dax
                    x_nb[ti, ki, 5] = f_day
                    nb_mask[ti, ki] = True

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

                    # ── LIT: key-point dx, relative dvx in ego frame ──────
                    # dx_key is already edge-to-edge for lead/rear slots
                    gap = abs(dx_key)
                    denom_base = dvx_rel if dx_key >= 0 else -dvx_rel
                    denom = denom_base
                    if abs(denom) < 1e-6:
                        denom = 1e-6 if denom >= 0 else -1e-6
                    lit = gap / denom
                    s_x = _lit_to_lis(lit, cfg.lis_mode)

                    # ── delta_lane ────────────────────────────────────────
                    ego_lid = int(ego_lane_arr[ti])
                    nb_lid  = int(lane_id[r])
                    delta_lane = float(abs(nb_lid - ego_lid)) if (ego_lid >= 0 and nb_lid >= 0) else 0.0
                    s_y = float(np.sqrt(lc_state ** 2 + delta_lane ** 2))

                    # ── importance ────────────────────────────────────────
                    i_total = compute_importance(
                        s_x, s_y, cfg.lambda_x, cfg.lambda_y, cfg.alpha, cfg.beta
                    )

                    # ── slot importance boost: I_new = I * (1 + alpha * w_slot)
                    if cfg.slot_importance_alpha > 0.0:
                        if cfg.slot_importance_conditional:
                            w_slot = _get_slot_weight(ki, ti, _lc_lane_lv, _lc_frame_ti, _lc_type)
                        else:
                            w_slot = SLOT_WEIGHTS[ki]
                        i_total = min(
                            i_total * (1.0 + cfg.slot_importance_alpha * w_slot),
                            1.0,
                        )

                    nb_class = class_map.get(nid, "car")
                    size_bin, _ = _volume_bin(nb_l, nb_w, nb_class)

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
        "x_ego":       _safe_float(np.stack(x_ego_list,   0)),
        "y":           _safe_float(np.stack(y_fut_list,    0)),
        "y_vel":       _safe_float(np.stack(y_vel_list,    0)),
        "y_acc":       _safe_float(np.stack(y_acc_list,    0)),
        "x_nb":        _safe_float(np.stack(x_nb_list,     0)),
        "nb_mask":     np.stack(nb_mask_list, 0),
        "x_last_abs":  np.stack(x_last_abs_list, 0),
        "recordingId": np.full(n_kept, int(rec_id), dtype=np.int32),
        "trackId":     np.array(trackid_list, np.int32),
        "t0_frame":    np.array(t0_list,      np.int32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE: raw -> mmap  (highD preprocess.py와 동일한 출력 구조)
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
    print(f"  drop_vru        : {cfg.drop_vru}")
    print(f"  non_relative    : {cfg.non_relative}")

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
        "x_ego":       open_memmap(out / "x_ego.npy",       "w+", "float32", (total, *s0["x_ego"].shape[1:])),
        "y":           open_memmap(out / "y.npy",            "w+", "float32", (total, *s0["y"].shape[1:])),
        "y_vel":       open_memmap(out / "y_vel.npy",        "w+", "float32", (total, *s0["y_vel"].shape[1:])),
        "y_acc":       open_memmap(out / "y_acc.npy",        "w+", "float32", (total, *s0["y_acc"].shape[1:])),
        "x_nb":        open_memmap(out / "x_nb.npy",         "w+", "float32", (total, *s0["x_nb"].shape[1:])),
        "nb_mask":     open_memmap(out / "nb_mask.npy",      "w+", "bool",    (total, *s0["nb_mask"].shape[1:])),
        "x_last_abs":  open_memmap(out / "x_last_abs.npy",   "w+", "float32", (total, 2)),
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
        description="exiD preprocessing pipeline  (raw CSV -> mmap)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir",    default="data/exiD", help="Base data directory")
    ap.add_argument("--raw_dir",     default="raw",       help="Raw CSV subdir under data_dir")
    ap.add_argument("--mmap_dir",    default="mmap",      help="Mmap output subdir under data_dir")
    ap.add_argument("--num_workers", type=int, default=0, help="Worker processes (0 = os.cpu_count())")

    # recording
    ap.add_argument("--target_hz",   type=float, default=3.0)
    ap.add_argument("--history_sec", type=float, default=2.0)
    ap.add_argument("--future_sec",  type=float, default=5.0)
    ap.add_argument("--stride_sec",  type=float, default=1.0)

    # LIS
    ap.add_argument("--lis_mode", default="7",
                    choices=["3", "5", "7", "9"],
                    help="LIS binning mode for s_x: 3={-1,0,1} | 5={-2,...,2} | 7={-3,...,3} | 9={-4,...,4}")

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

    # exiD-specific
    ap.add_argument("--drop_vru", action="store_true", default=True,
                    help="VRU (motorcycle/bicycle/pedestrian) 차량 관련 윈도우 제거")
    ap.add_argument("--keep_vru", action="store_true", default=False,
                    help="--drop_vru 를 무효화하고 VRU 윈도우를 포함")

    # neighbor feature mode
    ap.add_argument("--non_relative", action="store_true", default=False,
                    help="x_nb[0:6] = neighbor's abs values in normalised frame "
                         "(instead of ego-relative differences). "
                         "s_x/s_y/I always use relative/context values.")

    ap.add_argument("--dry_run", action="store_true")

    a = ap.parse_args()

    drop_vru = a.drop_vru and not a.keep_vru

    return Config(
        data_dir = Path(a.data_dir),
        raw_dir  = Path(a.raw_dir),
        mmap_dir = Path(a.mmap_dir),
        target_hz    = a.target_hz,
        history_sec  = a.history_sec,
        future_sec   = a.future_sec,
        stride_sec   = a.stride_sec,
        lis_mode = a.lis_mode,
        lambda_x = a.lambda_x,
        lambda_y = a.lambda_y,
        alpha    = a.alpha,
        beta     = a.beta,
        gate_topn = a.gate_topn,
        slot_importance_alpha        = a.slot_importance_alpha,
        slot_importance_conditional  = a.slot_importance_conditional,
        drop_vru    = drop_vru,
        non_relative = a.non_relative,
        dry_run     = a.dry_run,
        num_workers = a.num_workers,
    )


def main() -> None:
    cfg = parse_args()
    stage_raw2mmap(cfg)


if __name__ == "__main__":
    main()
