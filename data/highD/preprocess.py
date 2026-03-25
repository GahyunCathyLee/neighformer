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

x_nb   : (N, T, K, 13)   ego-relative neighbor features
    idx  0  dx        longitudinal distance  (nb_x - ego_x)
    idx  1  dy        lateral distance       (nb_y - ego_y)
    idx  2  dvx       relative longitudinal velocity
    idx  3  dvy       relative lateral velocity
    idx  4  dax       relative longitudinal acceleration
    idx  5  day       relative lateral acceleration
    idx  6  lc_state  lane-change state  {0: closing in, 1: stay, 2: moving out}
    idx  7  lit       Longitudinal Interaction Time  dx / (dvx ± eps)
    idx  8  lis       Longitudinal Interaction State (binned lit)
    idx  9  gate      1 if neighbor is active (gate_theta threshold or top-N selection)
    idx 10  I_x       longitudinal importance
    idx 11  I_y       lateral importance
    idx 12  I         composite importance  sqrt((I_x^2 + I_y^2) / 2)

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
  [importance_mode='lis']  — default
    I_x = exp(-(lis^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
    I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lis|^py) * exp(-by * delta_lane)
    I   = sqrt((I_x^2 + I_y^2) / 2)

    lit: Longitudinal Interaction Time — gap / (dvx ± eps)   eps=1.0
         gap = |x_rear_nb - x_front_ego|  if nb ahead (x_nb >= x_ego)
             = |x_rear_ego - x_front_nb|  if nb behind (x_ego > x_nb)
         front = center_x + 0.5*length,  rear = center_x - 0.5*length
    lis: Longitudinal Interaction State (discrete, LIS_BINS[lis_mode])
    delta_lane = |nb_lane_id - ego_lane_id|  in {0, 1, 2, ...}

    Fixed params (applied as-is regardless of bin count):
        sx=1.0, ax=0.15, bx=0.2, sy=2.0, ay=0.1, by=0.1, py=1.5

  [importance_mode='lit']
    I_x = exp(-(lit^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
    I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lit|^1.5) * exp(-by * delta_lane)
    I   = sqrt((I_x^2 + I_y^2) / 2)

    Fixed params (from legacy highd_pipeline.py):
        sx=15.0, ax=0.2, bx=0.25, sy=2.0, ay=0.01, by=0.1

    gate_mode='single': gate = 1  if  I >= theta  (I = sqrt((I_x^2 + I_y^2) / 2))
        theta = P85 of I distribution over dataset
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
NB_DIM  = 13   # dx, dy, dvx, dvy, dax, day, lc_state, lit, lis, gate, I_x, I_y, I
K       = 8    # neighbor slots

# Slot priority for top-N gate tie-breaking: 0 > 2 > 5 > 1 > 4 > 7 > 3 > 6
_TOPN_SLOT_PRIORITY = {s: r for r, s in enumerate([0, 2, 5, 1, 4, 7, 3, 6])}


def _apply_topn_gate(nb_row: np.ndarray, mask_row: np.ndarray, n: int) -> None:
    """Select top-n slots by I (idx 12) and zero-gate the rest (in-place).
    Tie-breaking: slot priority 0>2>5>1>4>7>3>6.
    """
    K_local = nb_row.shape[0]
    valid = [k for k in range(K_local) if mask_row[k]]
    valid.sort(key=lambda k: (-nb_row[k, 12], _TOPN_SLOT_PRIORITY.get(k, K_local)))
    selected = set(valid[:n])
    for k in valid:
        if k not in selected:
            nb_row[k, 9]  = 0.0
            nb_row[k, 12] = 0.0


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


def _lit_to_lis(lit: float, lis_mode: str) -> float:
    cfg = LIS_BINS[lis_mode]
    return cfg['vals'][bisect.bisect_right(cfg['cuts'], lit)]


# ─────────────────────────────────────────────────────────────────────────────
# Importance parameters
# ─────────────────────────────────────────────────────────────────────────────

# [importance_mode='lis']
# lis is used directly (no normalization). Fixed params apply regardless of bin count.
#
# I_x = exp(-(lis^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
# I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lis|^py) * exp(-by * delta_lane)
# I   = sqrt((I_x^2 + I_y^2) / 2)
IMPORTANCE_PARAMS_LIS: Dict[str, float] = {
    'sx': 1.0, 'ax': 0.15, 'bx': 0.2,
    'sy': 2.0, 'ay': 0.1,  'by': 0.1, 'py': 1.5,
}

# [importance_mode='lit']
# I_x = exp(-(lit^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
# I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lit|^1.5) * exp(-by * delta_lane)
# Fixed legacy params (from highd_pipeline.py)
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

    # lc / gating
    t_front:  float = 3.0
    t_back:   float = 5.0
    vy_eps:   float = 0.27   # used only by lc_version='v1'
    eps_gate: float = 1.0    # raised from 0.1: improves lit sensitivity over dvx

    # lc_state v2 (dvy-based) thresholds
    dvy_eps_cross: float = 0.26   # |dvy| threshold for cross-lane slot neighbors
    dvy_eps_same:  float = 1.03   # |dvy| threshold for same-lane slot (0/1) neighbors
    dy_same:       float = 1.5    # |dy| < dy_same → treat as same-lane

    # LIS mode  (used when importance_mode='lis')
    lis_mode: str = '3'  # '3' | '5' | '7' | '9'

    # importance mode
    importance_mode: str = 'lis'  # 'lis' | 'lit'

    # importance gate
    gate_theta: float = 0.0   # 0.0 = no threshold (all gates=1)
    gate_topn:  int   = 0     # >0 = keep top-N slots by I; 0 = disabled

    # lc_state version
    lc_version: str = "v3"           # "v1" (slot-based) | "v2" (dy-sign-based) | "v3" (latV+lco-based) | "v4" (lco_norm-based)

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

def compute_importance_lis(
    lis: float,
    delta_lane: float,
    lc_state: float,
) -> Tuple[float, float, float]:
    """
    importance_mode='lis'  —  lis used directly (no normalization).
    Fixed params applied regardless of bin count (5 / 7 / 9 / abs variants).

    I_x = exp(-(lis^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
    I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lis|^py) * exp(-by * delta_lane)
    I   = sqrt((I_x^2 + I_y^2) / 2)

    Params: sx=1.0, ax=0.15, bx=0.2, sy=2.0, ay=0.1, by=0.1, py=1.5
    """
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
    """
    importance_mode='lit':
        I_x = exp(-(lit^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
        I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lit|^1.5) * exp(-by * delta_lane)
        I   = sqrt((I_x^2 + I_y^2) / 2)
    Fixed params: sx=15.0, ax=0.2, bx=0.25, sy=2.0, ay=0.01, by=0.1
    """
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

    vid_to_dd  = dict(zip(trk_meta["id"].astype(int), trk_meta["drivingDirection"].astype(int)))
    vid_to_w   = dict(zip(trk_meta["id"].astype(int), trk_meta["width"].astype(float)))
    vid_to_l   = dict(zip(trk_meta["id"].astype(int), trk_meta["height"].astype(float)))

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

                    # ── lc_state ─────────────────────────────────────────────
                    if cfg.lc_version == "v1":
                        vyn = float(yv[r])
                        if ki < 2:                          # same-lane lead / rear
                            lc_state = 0.0
                        elif ki < 5:                        # left group (slots 2,3,4)
                            if   vyn >  cfg.vy_eps: lc_state = -1.0  # toward ego lane
                            elif vyn < -cfg.vy_eps: lc_state = -3.0  # away from ego lane
                            else:                   lc_state = -2.0  # staying
                        else:                               # right group (slots 5,6,7)
                            if   vyn < -cfg.vy_eps: lc_state =  1.0  # toward ego lane
                            elif vyn >  cfg.vy_eps: lc_state =  3.0  # away from ego lane
                            else:                   lc_state =  2.0  # staying
                    elif cfg.lc_version == "v2":
                        dy      = float(rel[1])
                        dvy     = float(rel[3])
                        abs_dvy = abs(dvy)

                        if ki < 2 and abs(dy) < cfg.dy_same:
                            # Case 1: same-lane slot (0/1) AND |dy|<dy_same → same lane
                            if abs_dvy > cfg.dvy_eps_same:
                                lc_state = 2.0
                            else:
                                lc_state = 1.0
                        elif ki >= 2:
                            # Case 2: cross-lane slot → closing/staying/moving
                            if abs_dvy > cfg.dvy_eps_cross:
                                lc_state = 0.0 if dy * dvy < 0 else 2.0
                            else:
                                lc_state = 1.0
                        else:
                            # Case 3: same-lane slot (0/1) but |dy|>=dy_same → transitioning
                            lc_state = 0.0 if dy * dvy < 0 else 2.0
                    elif cfg.lc_version == "v3":
                        nb_lat_v = float(yv[r])
                        nb_lco   = float(lat_lane_offset_arr[r])
                        if ki < 2:   # same lane (lead / rear)
                            if (nb_lco < -1.0 and nb_lat_v > 0.0) or \
                               (nb_lco >  1.0 and nb_lat_v < 0.0):
                                lc_state = 0.0
                            elif (nb_lco < -1.0 and nb_lat_v < 0.0) or \
                                 (nb_lco >  1.0 and nb_lat_v > 0.0) or \
                                 abs(nb_lat_v) > 0.029:
                                lc_state = 2.0
                            else:
                                lc_state = 1.0
                        elif ki < 5:  # left lane (slots 2,3,4)
                            if   nb_lat_v < -0.029: lc_state = 0.0
                            elif nb_lat_v >  0.029: lc_state = 2.0
                            else:                   lc_state = 1.0
                        else:         # right lane (slots 5,6,7)
                            if   nb_lat_v < -0.029: lc_state = 2.0
                            elif nb_lat_v >  0.029: lc_state = 0.0
                            else:                   lc_state = 1.0
                    else:  # v4: lco_norm 기반 경계 판단 + slot별 방향 결정
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
                    lit = gap / (denom_base + (cfg.eps_gate if denom_base >= 0 else -cfg.eps_gate))
                    lis        = _lit_to_lis(lit, cfg.lis_mode)
                    delta_lane = float(abs(int(lane_id[r]) - int(ego_lane_arr[ti])))

                    # ── importance: 'lis' mode or 'lit' mode ──────────────────
                    if cfg.importance_mode == 'lit':
                        ix, iy, i_total = compute_importance_lit(
                            lit, delta_lane, lc_state
                        )
                    else:  # 'lis' (default)
                        ix, iy, i_total = compute_importance_lis(
                            lis, delta_lane, lc_state
                        )

                    # ── gate ──────────────────────────────────────────────
                    if cfg.gate_theta > 0.0:
                        gate = 1.0 if i_total >= cfg.gate_theta else 0.0
                    else:
                        gate = 1.0  # gate_topn post-processing or all-active default

                    x_nb[ti, ki, 6]  = lc_state
                    x_nb[ti, ki, 7]  = lit
                    x_nb[ti, ki, 8]  = lis
                    i_total *= gate
                    x_nb[ti, ki, 9]  = gate
                    x_nb[ti, ki, 10] = ix
                    x_nb[ti, ki, 11] = iy
                    x_nb[ti, ki, 12] = i_total

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
    print(f"  importance_mode : {cfg.importance_mode}"
          + (f"  lis_mode : {cfg.lis_mode}" if cfg.importance_mode == 'lis' else
             f"  params   : {IMPORTANCE_PARAMS_LIT}"))

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
    ap.add_argument("--history_sec",        type=float, default=3.0)
    ap.add_argument("--future_sec",         type=float, default=5.0)
    ap.add_argument("--stride_sec",         type=float, default=1.0)
    ap.add_argument("--normalize_upper_xy", action="store_true", default=True)

    # lc / gating
    ap.add_argument("--t_front",  type=float, default=3.0)
    ap.add_argument("--t_back",   type=float, default=5.0)
    ap.add_argument("--vy_eps",   type=float, default=0.27,
                    help="yV threshold used only by lc_version=v1")
    ap.add_argument("--eps_gate", type=float, default=1.0,
                    help="eps for lit denominator clamp (raised to 1.0)")
    ap.add_argument("--dvy_eps_cross", type=float, default=0.26,
                    help="lc_state v2: |dvy| threshold for cross-lane slot neighbors")
    ap.add_argument("--dvy_eps_same",  type=float, default=1.03,
                    help="lc_state v2: |dvy| threshold for same-lane slot (0/1) neighbors")
    ap.add_argument("--dy_same",       type=float, default=1.5,
                    help="lc_state v2: |dy| < dy_same means same-lane for slot 0/1")

    # LIS
    ap.add_argument("--lis_mode", default="7",
                    choices=["3", "5", "7", "9"],
                    help=(
                        "LIS binning mode (used when importance_mode=lis): "
                        "3=lit 3-bin {-1,0,1} | "
                        "5=lit 5-bin {-2,...,2} | "
                        "7=lit 7-bin {-3,...,3} | "
                        "9=lit 9-bin {-4,...,4}"
                    ))

    # importance mode
    ap.add_argument("--importance_mode", default="lis", choices=["lis", "lit"],
                    help=(
                        "How to compute I_x / I_y / I: "
                        "lis = use discrete LIS as interaction proxy (default, params per lis_mode) | "
                        "lit = use continuous LIT and lc_state directly "
                        "(legacy highd_pipeline.py params: sx=15,ax=0.2,bx=0.25,sy=2,ay=0.01,by=0.1)"
                    ))

    # gate
    ap.add_argument("--gate_theta", type=float, default=0.0,
                    help="I threshold gate: gate=1 if I>=theta. 0.0 = all gates active (default)")
    ap.add_argument("--gate_topn", type=int, default=0,
                    help="Top-N gate: keep up to N slots with highest I; "
                         "tie-break by slot priority 0>2>5>1>4>7>3>6. 0 = disabled")
    ap.add_argument("--lc_version", default="v3", choices=["v1", "v2", "v3", "v4"],
                    help="lc_state 계산 방식: "
                         "v1=slot기반 절대yV (tf_trajPred 방식, 값범주 {-3,-2,-1,0,1,2,3}), "
                         "v2=dvy기반+slot/dy조합 (neighformer 방식, 값범주 {0,1,2}), "
                         "v3=latVelocity+latLaneCenterOffset기반 (default, 값범주 {0,1,2}), "
                         "v4=lco_norm(X=0.5) 기반 경계 판단+slot방향 결정 (값범주 {0,1,2})")

    ap.add_argument("--non_relative", action="store_true", default=False,
                    help="x_nb[0:6] = neighbor's abs values in globally-shifted frame "
                         "(instead of ego-relative differences). "
                         "lc_state/LIT/importance (x_nb[6:12]) always use relative values.")

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
        t_front  = a.t_front,
        t_back   = a.t_back,
        vy_eps   = a.vy_eps,
        eps_gate      = a.eps_gate,
        dvy_eps_cross = a.dvy_eps_cross,
        dvy_eps_same  = a.dvy_eps_same,
        dy_same       = a.dy_same,
        lis_mode        = a.lis_mode,
        importance_mode = a.importance_mode,
        gate_theta      = a.gate_theta,
        gate_topn       = a.gate_topn,
        lc_version    = a.lc_version,
        non_relative = a.non_relative,
        dry_run     = a.dry_run,
        num_workers = a.num_workers,
    )


def main() -> None:
    cfg = parse_args()
    stage_raw2mmap(cfg)


if __name__ == "__main__":
    main()