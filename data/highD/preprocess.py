#!/usr/bin/env python3
"""
highd_pipeline.py  —  HighD preprocessing pipeline  (raw CSV → mmap)

STAGE raw2mmap :  highD raw CSV  →  memory-mapped arrays

stats 계산은 train.py / evaluate.py 실행 시 src/stats.py 가 자동으로 수행합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature schema
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x_ego  : (N, T, 9)
    [x, y, xV, yV, xA, yA,
     latLaneCenterOffset,   idx 6
     laneChange,            idx 7
     norm_off]              idx 8

x_nb   : (N, T, K, 12)   ego-relative neighbor features
    idx  0  dx        longitudinal distance  (nb_x - ego_x)
    idx  1  dy        lateral distance       (nb_y - ego_y)
    idx  2  dvx       relative longitudinal velocity
    idx  3  dvy       relative lateral velocity
    idx  4  dax       relative longitudinal acceleration
    idx  5  day       relative lateral acceleration
    idx  6  lc_state  lane-change state  {0: closing in, 1: stay, 2: moving out}
    idx  7  dx_time   dx / (dvx +- eps)
    idx  8  gate      1 if (I_x >= theta_x) OR (I_y >= theta_y) else 0  [theta = P85]
    idx  9  I_x       longitudinal importance
    idx 10  I_y       lateral importance
    idx 11  I         composite importance  sqrt(I_x * I_y)

y          : (N, Tf, 2)     future [x, y]
y_vel      : (N, Tf, 2)     future [xV, yV]
y_acc      : (N, Tf, 2)     future [xA, yA]
nb_mask    : (N, T, K)      bool - True if neighbor exists
x_last_abs : (N, 2)         last absolute (x, y) of ego history
meta_recordingId / trackId / t0_frame : (N,)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Importance formula
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    I_x = exp(-(dx_time^2 / (2*sx^2)) - ax * delta_lane^2)
    I_y = exp(-(lc_state^2 / (2*sigma^2)) - alpha * |dx_time|^1.5)
    I   = sqrt(I_x * I_y)

    I_x: longitudinal importance
        dx_time = dx / (dvx +- eps)   eps=1.0
        sx=3.0, ax=0.5
        delta_lane = |nb_lane_id - ego_lane_id|  in {0, 1, 2, ...}

    I_y: lateral importance  (always uses lc_state as d_val)
        lc_state: {0: closing in, 1: stay, 2: moving out}
        sigma=1.0  ->  I_y(stay=1)=0.61, I_y(moving_out=2)=0.14
        alpha=0.01 ->  I_y(dx_time=5s) *= 0.84  (lc_state-dominant)

    gate_mode='or':     gate = 1  if  (I_x >= theta_x) OR (I_y >= theta_y)
    gate_mode='single': gate = 1  if  I >= theta  (I = sqrt(I_x * I_y))
        theta_x, theta_y, theta = P85 of respective distribution over dataset

    gate_apply='feature': store gate as 0.0/1.0 in x_nb[:,:,:,8]  (default)
    gate_apply='mask':    gate=0 neighbor -> nb_mask set to False (treated as absent)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python highd_pipeline.py \\
      --raw_dir  data/highD/raw \\
      --mmap_dir data/highD/mmap
"""

from __future__ import annotations

import argparse
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

EGO_DIM = 9    # x, y, xV, yV, xA, yA, latLaneCenterOffset, laneChange, norm_off
NB_DIM  = 12   # dx, dy, dvx, dvy, dax, day, lc_state, dx_time, gate, I_x, I_y, I
K       = 8    # neighbor slots


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
    vy_eps:   float = 0.27
    eps_gate: float = 1.0   # raised from 0.1: improves dx sensitivity over dvx

    # importance gate (set thresholds after first mmap pass)
    gate_mode:    str   = 'or'      # 'or' = (I_x>=theta_x OR I_y>=theta_y) | 'single' = I>=theta
    gate_apply:   str   = 'feature' # 'feature' = store 0/1 in x_nb | 'mask' = gate=0 -> nb_mask=False
    gate_theta_x: float = 0.0       # I_x threshold (or-mode); 0.0 = legacy time-window gate
    gate_theta_y: float = 0.0       # I_y threshold (or-mode)
    gate_theta:   float = 0.0       # I  threshold (single-mode)

    # lc_state version
    lc_version: str = "v2"           # "v1" (slot-based, tf_trajPred) | "v2" (dy-sign-based, neighformer)

    # importance
    importance_dl_mode: str = "dy"   # "dy" | "lane_id" | "lc_state"

    # output / execution
    dry_run:     bool = False
    num_workers: int  = 0   # 0 = os.cpu_count()


# Importance parameters
# I_x: exp(-(dx_time^2/2sx^2) - ax*dy^2)
# I_y: exp(-(lc_state^2/2sigma^2) - alpha*|dx_time|^1.5)   [always lc_state-based]
IMPORTANCE_PARAMS: Dict[str, Dict[str, float]] = {
    "dy":       {"sx": 3.0, "ax": 0.28, "sigma": 1.0, "alpha": 0.01},
    "lane_id":  {"sx": 3.0, "ax": 0.28, "sigma": 1.0, "alpha": 0.01},
    "lc_state": {"sx": 3.0, "ax": 0.28, "sigma": 1.0, "alpha": 0.01},
}


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
    dx_time: float,
    delta_lane: float,
    lc_state: float,
    mode: str,
) -> Tuple[float, float, float]:
    """
    I_x = exp(-(dx_time^2 / 2*sx^2) - ax * delta_lane^2)
        delta_lane = |nb_lane_id - ego_lane_id|
    I_y = exp(-(lc_state^2 / 2*sigma^2) - alpha * |dx_time|^1.5)
        lc_state: {0: closing in -> highest I_y, 1: stay, 2: moving out -> lowest}
    I   = sqrt(I_x * I_y)
    """
    p  = IMPORTANCE_PARAMS[mode]
    ix = float(np.exp(
        -(dx_time ** 2) / (2.0 * p["sx"] ** 2)
        - p["ax"] * delta_lane ** 2
    ))
    iy = float(np.exp(
        -(lc_state ** 2) / (2.0 * p["sigma"] ** 2)
        - p["alpha"] * (abs(dx_time) ** 1.5)
    ))
    i_total = float(np.sqrt(ix * iy))
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

            # ── ego aux: latLaneCenterOffset / laneChange / norm_off ─────────
            ego_dd_v = vid_to_dd.get(v, 0)   # 원본 주행 방향 (flip 전)
            lat_off  = np.zeros(T, np.float32)
            lane_w   = np.zeros(T, np.float32)
            for ti in range(T):
                lid = int(ego_lane_arr[ti])
                if lid <= 0:
                    continue
                if ego_dd_v == 1 and len(upper_center) >= lid:
                    lat_off[ti] = float(ey[ti] - upper_center[lid - 1])
                    lane_w[ti]  = float(upper_width[lid - 1])
                elif ego_dd_v == 2 and len(lower_center) >= lid:
                    lat_off[ti] = float(ey[ti] - lower_center[lid - 1])
                    lane_w[ti]  = float(lower_width[lid - 1])
            half_w   = lane_w * 0.5
            norm_off = np.zeros(T, np.float32)
            ok       = half_w > 1e-6
            norm_off[ok] = lat_off[ok] / half_w[ok]
            ego_lc   = lane_change[ego_rows].astype(np.float32)
            # ─────────────────────────────────────────────────────────────────

            x_ego = np.stack(
                [ex, ey, exv, eyv, exa, eya, lat_off, ego_lc, norm_off], axis=1
            ).astype(np.float32)

            y_fut = np.stack([x[fut_rows],  y[fut_rows]],  axis=1).astype(np.float32)
            y_vel = np.stack([xv[fut_rows], yv[fut_rows]], axis=1).astype(np.float32)
            y_acc = np.stack([xa[fut_rows], ya[fut_rows]], axis=1).astype(np.float32)

            x_nb      = np.zeros((T, K, NB_DIM), np.float32)
            nb_mask   = np.zeros((T, K), bool)

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
                    x_nb[ti, ki, 0:6] = rel
                    nb_mask[ti, ki]   = True

                    # ── lc_state v1: slot-based, 현재 프레임 절대 yVelocity (tf_trajPred 방식) ──
                    # ── lc_state v2: dy부호 + 윈도우 평균 yVelocity             (neighformer 방식) ──
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
                    else:                                   # v2 (default)
                        win_frames = [hf - w * step for w in range(int(round(cfg.target_hz)))]
                        yv_vals = []
                        for wf in win_frames:
                            wr = per_vid_frame_to_row.get(nid, {}).get(int(wf))
                            if wr is not None:
                                yv_vals.append(float(yv[wr]))
                        vyn = float(np.mean(yv_vals)) if yv_vals else float(yv[r])

                        dy_sign = float(rel[1])
                        if abs(vyn) < cfg.vy_eps:
                            lc_state = 1.0
                        elif dy_sign * vyn > 0:
                            lc_state = 2.0
                        else:
                            lc_state = 2.0 if ki < 2 else 0.0

                    dx  = float(rel[0])
                    dvx = float(rel[2])
                    # eps_gate raised to 1.0: keeps dx dominant over dvx in dx_time
                    dx_time    = dx / (dvx + (cfg.eps_gate if dvx >= 0 else -cfg.eps_gate))
                    delta_lane = float(abs(int(lane_id[r]) - int(ego_lane_arr[ti])))

                    ix, iy, i_total = compute_importance(
                        dx_time, delta_lane, lc_state, cfg.importance_dl_mode
                    )

                    # gate: importance-based
                    if cfg.gate_mode == 'single' and cfg.gate_theta > 0.0:
                        gate = 1.0 if i_total >= cfg.gate_theta else 0.0
                    elif cfg.gate_mode == 'or' and cfg.gate_theta_x > 0.0 and cfg.gate_theta_y > 0.0:
                        gate = 1.0 if (ix >= cfg.gate_theta_x or iy >= cfg.gate_theta_y) else 0.0
                    else:
                        # fallback: legacy time-window gate (use until thresholds are calibrated)
                        gate = 1.0 if (-cfg.t_back < dx_time < cfg.t_front) else 0.0

                    x_nb[ti, ki, 6]  = lc_state
                    x_nb[ti, ki, 7]  = dx_time
                    x_nb[ti, ki, 9]  = ix
                    x_nb[ti, ki, 10] = iy
                    x_nb[ti, ki, 11] = i_total

                    if cfg.gate_apply == 'mask':
                        # gate=0: treat neighbor as absent
                        if gate == 0.0:
                            nb_mask[ti, ki]  = False
                            x_nb[ti, ki, :]  = 0.0
                        x_nb[ti, ki, 8] = 1.0   # all remaining are gate=1
                    else:
                        # 'feature': store raw gate value (default)
                        x_nb[ti, ki, 8] = gate

            x_ego_list.append(x_ego)
            y_fut_list.append(y_fut)
            y_vel_list.append(y_vel)
            y_acc_list.append(y_acc)
            x_nb_list.append(x_nb)
            nb_mask_list.append(nb_mask)
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

        for key in ["x_ego", "y", "y_vel", "y_acc", "x_nb", "nb_mask"]:
            fp[key][cursor:end] = buf[key]

        fp["x_last_abs"][cursor:end] = buf["x_ego"][:, -1, 0:2]

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

    # lc / gating
    ap.add_argument("--t_front",  type=float, default=3.0)
    ap.add_argument("--t_back",   type=float, default=5.0)
    ap.add_argument("--vy_eps",   type=float, default=0.27)
    ap.add_argument("--eps_gate",      type=float, default=1.0,
                    help="eps for dx_time denominator clamp (raised to 1.0)")
    ap.add_argument("--gate_mode",     default="or", choices=["or", "single"],
                    help="or: (I_x>=theta_x OR I_y>=theta_y) | single: I>=theta")
    ap.add_argument("--gate_apply",    default="feature", choices=["feature", "mask"],
                    help="feature: store gate 0/1 in x_nb | mask: gate=0 removes neighbor")
    ap.add_argument("--gate_theta_x",  type=float, default=0.0,
                    help="I_x threshold for or-mode (P85). 0.0 = legacy gate")
    ap.add_argument("--gate_theta_y",  type=float, default=0.0,
                    help="I_y threshold for or-mode (P85). 0.0 = legacy gate")
    ap.add_argument("--gate_theta",    type=float, default=0.0,
                    help="I threshold for single-mode (P85). 0.0 = legacy gate")
    ap.add_argument("--lc_version", default="v2", choices=["v1", "v2"],
                    help="lc_state 계산 방식: "
                         "v1=slot기반 절대yV (tf_trajPred 방식, 값범주 {-3,-2,-1,0,1,2,3}), "
                         "v2=dy부호+윈도우평균yV (neighformer 방식, 값범주 {0,1,2}, default)")

    # importance
    ap.add_argument("--importance_dl_mode", default="dy",
                    choices=["dy", "lane_id", "lc_state"])

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
        gate_mode     = a.gate_mode,
        gate_apply    = a.gate_apply,
        gate_theta_x  = a.gate_theta_x,
        gate_theta_y  = a.gate_theta_y,
        gate_theta    = a.gate_theta,
        lc_version    = a.lc_version,
        importance_dl_mode = a.importance_dl_mode,
        dry_run     = a.dry_run,
        num_workers = a.num_workers,
    )


def main() -> None:
    cfg = parse_args()
    stage_raw2mmap(cfg)


if __name__ == "__main__":
    main()