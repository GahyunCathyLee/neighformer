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

x_nb   : (N, T, K, 13)   ego-relative neighbor features
    idx  0  dx        longitudinal distance  (key-point based, ego local frame)
    idx  1  dy        lateral distance       (key-point based, ego local frame)
    idx  2  dvx       relative longitudinal velocity  (ego local frame)
    idx  3  dvy       relative lateral velocity       (ego local frame)
    idx  4  dax       relative longitudinal acceleration
    idx  5  day       relative lateral acceleration
    idx  6  lc_state  lane-change state  {0: closing in, 1: stay, 2: moving out}
    idx  7  lit       Longitudinal Interaction Time  dx / (dvx ± eps)
    idx  8  lis       Longitudinal Interaction State (binned lit)
    idx  9  gate      1 if (I_x >= theta_x) OR (I_y >= theta_y) else 0  [theta = P85]
    idx 10  I_x       longitudinal importance
    idx 11  I_y       lateral importance
    idx 12  I         composite importance  sqrt((I_x^2 + I_y^2) / 2)

    With --non_relative: idx 0-5 hold the neighbor's own values in the
    normalised reference frame instead of ego-relative differences.
    lc_state/LIT/importance (idx 6-12) always use relative values.

y          : (N, Tf, 2)     future [x, y]  — ego-centric normalised frame
y_vel      : (N, Tf, 2)     future [xV, yV] — normalised frame
y_acc      : (N, Tf, 2)     future [xA, yA] — normalised frame
nb_mask    : (N, T, K)      bool - True if neighbor exists
x_last_abs : (N, 2)         last absolute (x, y) of ego history (pre-normalisation)
ref_heading: (N,)            ego heading [rad] at last history frame (for inverse transform)
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
  [importance_mode='lis']  — default
    I_x = exp(-(lis^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
    I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lis|^py) * exp(-by * delta_lane)
    I   = sqrt((I_x^2 + I_y^2) / 2)

  [importance_mode='lit']
    I_x = exp(-(lit^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
    I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lit|^1.5) * exp(-by * delta_lane)
    I   = sqrt((I_x^2 + I_y^2) / 2)
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
NB_DIM  = 13   # dx, dy, dvx, dvy, dax, day, lc_state, lit, lis, gate, I_x, I_y, I
K       = 8    # neighbor slots

# exiD 기본 프레임레이트 (recordingMeta에 frameRate 컬럼이 없을 경우 fallback)
EXID_DEFAULT_HZ = 25.0


# ─────────────────────────────────────────────────────────────────────────────
# LIS binning  (highD와 동일)
# ─────────────────────────────────────────────────────────────────────────────

LIS_BINS: Dict[str, Dict] = {
    '3': {'cuts': [-3.7458, 4.9133],
          'vals': [-1.0, 0.0, 1.0],
          'L': 1.0},
    '5': {'cuts': [-10.0898, -1.5466, 2.5303, 11.4707],
          'vals': [-2.0, -1.0, 0.0, 1.0, 2.0],
          'L': 2.0},
    '7': {'cuts': [-14.5871, -5.6417, -0.8287, 1.6619, 6.8972, 16.0590],
          'vals': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
          'L': 3.0},
    '9': {'cuts': [-18.2465, -8.7666, -3.7458, -0.5275, 1.2365, 4.9133, 10.1101, 19.7939],
          'vals': [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
          'L': 4.0},
}


def _lit_to_lis(lit: float, lis_mode: str) -> float:
    cfg = LIS_BINS[lis_mode]
    return cfg['vals'][bisect.bisect_right(cfg['cuts'], lit)]


# ─────────────────────────────────────────────────────────────────────────────
# Importance parameters  (highD와 동일)
# ─────────────────────────────────────────────────────────────────────────────

IMPORTANCE_PARAMS_LIS: Dict[str, float] = {
    'sx': 1.0, 'ax': 0.15, 'bx': 0.2,
    'sy': 2.0, 'ay': 0.1,  'by': 0.1, 'py': 1.5,
}

IMPORTANCE_PARAMS_LIT: Dict[str, float] = {
    'sx': 15.0, 'ax': 0.2, 'bx': 0.25,
    'sy':  2.0, 'ay': 0.01, 'by': 0.1,
}


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

    # lc / gating
    t_front:  float = 3.0
    t_back:   float = 5.0
    eps_gate: float = 1.0

    # lc_state v2 (dvy-based) thresholds
    dvy_eps_cross: float = 0.26
    dvy_eps_same:  float = 1.03
    dy_same:       float = 1.5

    # LIS mode
    lis_mode: str = '7'   # '3' | '5' | '7' | '9'

    # importance mode
    importance_mode: str = 'lis'  # 'lis' | 'lit'

    # importance gate
    gate_theta: float = 0.0   # 0.0 = legacy time-window gate

    # lc_state version
    lc_version: str = "v2"    # "v1" | "v2"

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
# Importance  (highD와 동일한 함수)
# ─────────────────────────────────────────────────────────────────────────────

def compute_importance_lis(
    lis: float,
    delta_lane: float,
    lc_state: float,
) -> Tuple[float, float, float]:
    p  = IMPORTANCE_PARAMS_LIS
    ix = float(np.exp(-(lis ** 2) / (2.0 * p["sx"] ** 2))
               * np.exp(-p["ax"] * lc_state)
               * np.exp(-p["bx"] * delta_lane))
    iy = float(np.exp(-(lc_state ** 2) / (2.0 * p["sy"] ** 2))
               * np.exp(-p["ay"] * (abs(lis) ** p["py"]))
               * np.exp(-p["by"] * delta_lane))
    i_total = float(np.sqrt((ix ** 2 + iy ** 2) / 2.0))
    return ix, iy, i_total


def compute_importance_lit(
    lit: float,
    delta_lane: float,
    lc_state: float,
) -> Tuple[float, float, float]:
    p  = IMPORTANCE_PARAMS_LIT
    ix = float(np.exp(-(lit ** 2) / (2.0 * p["sx"] ** 2))
               * np.exp(-p["ax"] * lc_state)
               * np.exp(-p["bx"] * delta_lane))
    iy = float(np.exp(-(lc_state ** 2) / (2.0 * p["sy"] ** 2))
               * np.exp(-p["ay"] * (abs(lit) ** 1.5))
               * np.exp(-p["by"] * delta_lane))
    i_total = float(np.sqrt((ix ** 2 + iy ** 2) / 2.0))
    return ix, iy, i_total


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

    # ── sample 수집 루프 ──────────────────────────────────────────────────────
    x_ego_list:       List[np.ndarray] = []
    y_fut_list:       List[np.ndarray] = []
    y_vel_list:       List[np.ndarray] = []
    y_acc_list:       List[np.ndarray] = []
    x_nb_list:        List[np.ndarray] = []
    nb_mask_list:     List[np.ndarray] = []
    x_last_abs_list:  List[np.ndarray] = []
    ref_heading_list: List[float]      = []
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

                    # ── lc_state: center-to-center dy/dvy in ego frame ────
                    _, dy_cc  = _rot2d(
                        float(x[r]) - float(ex[ti]),
                        float(y[r]) - float(ey[ti]),
                        ego_hdg_ti,
                    )
                    # dvy in ego frame = lateral component of dvx_rel, dvy_rel
                    # (already in ego frame, so dvy_rel is the lateral relative vel)
                    dvy_cc = dvy_rel

                    if cfg.lc_version == "v1":
                        vyn = float(yv[r])
                        if ki < 2:
                            lc_state = 0.0
                        elif ki < 5:   # left group
                            if   vyn >  0.27: lc_state = -1.0
                            elif vyn < -0.27: lc_state = -3.0
                            else:             lc_state = -2.0
                        else:          # right group
                            if   vyn < -0.27: lc_state =  1.0
                            elif vyn >  0.27: lc_state =  3.0
                            else:             lc_state =  2.0
                    else:   # v2 (default)
                        abs_dvy_cc = abs(dvy_cc)
                        if ki < 2 and abs(dy_cc) < cfg.dy_same:
                            lc_state = 2.0 if abs_dvy_cc > cfg.dvy_eps_same else 1.0
                        elif ki >= 2:
                            if abs_dvy_cc > cfg.dvy_eps_cross:
                                lc_state = 0.0 if dy_cc * dvy_cc < 0 else 2.0
                            else:
                                lc_state = 1.0
                        else:
                            lc_state = 0.0 if dy_cc * dvy_cc < 0 else 2.0

                    # ── LIT: key-point dx, relative dvx in ego frame ──────
                    # dx_key is already edge-to-edge for lead/rear slots
                    gap = abs(dx_key)
                    denom_base = dvx_rel if dx_key >= 0 else -dvx_rel
                    lit = gap / (denom_base + (cfg.eps_gate if denom_base >= 0 else -cfg.eps_gate))
                    lis = _lit_to_lis(lit, cfg.lis_mode)

                    # ── delta_lane ────────────────────────────────────────
                    ego_lid = int(ego_lane_arr[ti])
                    nb_lid  = int(lane_id[r])
                    delta_lane = float(abs(nb_lid - ego_lid)) if (ego_lid >= 0 and nb_lid >= 0) else 0.0

                    # ── importance ────────────────────────────────────────
                    if cfg.importance_mode == 'lit':
                        ix, iy, i_total = compute_importance_lit(lit, delta_lane, lc_state)
                    else:
                        ix, iy, i_total = compute_importance_lis(lis, delta_lane, lc_state)

                    # ── gate ──────────────────────────────────────────────
                    if cfg.gate_theta > 0.0:
                        gate = 1.0 if i_total >= cfg.gate_theta else 0.0
                    else:
                        gate = 1.0 if (-cfg.t_back < lit < cfg.t_front) else 0.0

                    x_nb[ti, ki, 6]  = lc_state
                    x_nb[ti, ki, 7]  = lit
                    x_nb[ti, ki, 8]  = lis
                    i_total *= gate
                    x_nb[ti, ki, 9]  = gate
                    x_nb[ti, ki, 10] = ix
                    x_nb[ti, ki, 11] = iy
                    x_nb[ti, ki, 12] = i_total

            x_ego_list.append(x_ego)
            y_fut_list.append(y_fut)
            y_vel_list.append(y_vel)
            y_acc_list.append(y_acc)
            x_nb_list.append(x_nb)
            nb_mask_list.append(nb_mask)
            x_last_abs_list.append(np.array([ref_x, ref_y], np.float32))
            ref_heading_list.append(ref_hdg)
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
        "x_last_abs":  np.stack(x_last_abs_list, 0),   # pre-normalisation position
        "ref_heading": np.array(ref_heading_list, np.float32),  # radians
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
    print(f"  importance_mode : {cfg.importance_mode}"
          + (f"  lis_mode : {cfg.lis_mode}" if cfg.importance_mode == 'lis' else
             f"  params   : {IMPORTANCE_PARAMS_LIT}"))
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
        "ref_heading": open_memmap(out / "ref_heading.npy",  "w+", "float32", (total,)),
    }
    meta_rec   = np.zeros(total, np.int32)
    meta_track = np.zeros(total, np.int32)
    meta_frame = np.zeros(total, np.int32)

    # ── pass 2: write buffers -> mmap (sequential) ────────────────────────────
    cursor = 0
    for buf in tqdm(bufs, desc="Writing mmap"):
        n   = buf["x_ego"].shape[0]
        end = cursor + n

        for key in ["x_ego", "y", "y_vel", "y_acc", "x_nb", "nb_mask",
                    "x_last_abs", "ref_heading"]:
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

    # lc / gating
    ap.add_argument("--t_front",  type=float, default=3.0)
    ap.add_argument("--t_back",   type=float, default=5.0)
    ap.add_argument("--eps_gate", type=float, default=1.0,
                    help="eps for lit denominator clamp")
    ap.add_argument("--dvy_eps_cross", type=float, default=0.26,
                    help="lc_state v2: |dvy| threshold for cross-lane slot neighbors")
    ap.add_argument("--dvy_eps_same",  type=float, default=1.03,
                    help="lc_state v2: |dvy| threshold for same-lane slot (0/1) neighbors")
    ap.add_argument("--dy_same",       type=float, default=1.5,
                    help="lc_state v2: |dy| < dy_same means same-lane for slot 0/1")

    # LIS
    ap.add_argument("--lis_mode", default="9",
                    choices=["3", "5", "7", "9"],
                    help="LIS binning mode: 3={-1,0,1} | 5={-2,...,2} | 7={-3,...,3} | 9={-4,...,4}")

    # importance mode
    ap.add_argument("--importance_mode", default="lis", choices=["lis", "lit"])

    # gate
    ap.add_argument("--gate_theta", type=float, default=0.0,
                    help="I threshold for single-mode gate. 0.0 = legacy time-window gate")
    ap.add_argument("--lc_version", default="v2", choices=["v1", "v2"],
                    help="lc_state 계산 방식: v1=slot기반 절대yV | v2=dvy기반+slot/dy조합 (default)")

    # exiD-specific
    ap.add_argument("--drop_vru", action="store_true", default=True,
                    help="VRU (motorcycle/bicycle/pedestrian) 차량 관련 윈도우 제거")
    ap.add_argument("--keep_vru", action="store_true", default=False,
                    help="--drop_vru 를 무효화하고 VRU 윈도우를 포함")

    # neighbor feature mode
    ap.add_argument("--non_relative", action="store_true", default=False,
                    help="x_nb[0:6] = neighbor's abs values in normalised frame "
                         "(instead of ego-relative differences). "
                         "lc_state/LIT/importance (x_nb[6:12]) always use relative values.")

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
        t_front      = a.t_front,
        t_back       = a.t_back,
        eps_gate      = a.eps_gate,
        dvy_eps_cross = a.dvy_eps_cross,
        dvy_eps_same  = a.dvy_eps_same,
        dy_same       = a.dy_same,
        lis_mode        = a.lis_mode,
        importance_mode = a.importance_mode,
        gate_theta      = a.gate_theta,
        lc_version    = a.lc_version,
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
