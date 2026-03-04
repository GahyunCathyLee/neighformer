#!/usr/bin/env python3
"""
highd_pipeline.py  —  HighD preprocessing pipeline  (raw CSV → mmap)

STAGE raw2mmap :  highD raw CSV  →  memory-mapped arrays
STAGE stats    :  mmap arrays   →  normalization stats  (optional standalone)

stats는 --calc_stats 플래그로 raw2mmap 직후 자동 계산하거나,
--stage stats 로 기존 mmap에서 별도 실행할 수 있습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature schema
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x_ego  : (N, T, 6)
    [x, y, xV, yV, xA, yA]

x_nb   : (N, T, K, 12)   ego-relative neighbor features
    idx  0  dx        longitudinal distance  (nb_x - ego_x)
    idx  1  dy        lateral distance       (nb_y - ego_y)
    idx  2  dvx       relative longitudinal velocity
    idx  3  dvy       relative lateral velocity
    idx  4  dax       relative longitudinal acceleration
    idx  5  day       relative lateral acceleration
    idx  6  lc_state  lane-change state  {-3,-2,-1,0,1,2,3}
    idx  7  dx_time   dx / (dvx +- eps)
    idx  8  gate      1 if -t_back < dx_time < t_front else 0
    idx  9  I_x       longitudinal importance
    idx 10  I_y       lateral importance
    idx 11  I         composite importance  sqrt(I_x * I_y)

y          : (N, Tf, 2)     future [x, y]
y_vel      : (N, Tf, 2)     future [xV, yV]
y_acc      : (N, Tf, 2)     future [xA, yA]
nb_mask    : (N, T, K)      bool - True if neighbor exists
ego_static : (N, 2+C)       [width, length] + class one-hot (C=8)
nb_static  : (N, T, K, 2+C)
ego_safety : (N, T, 5)      [leadDHW, leadDV, leadTHW, leadTTC, lead_exists]
x_last_abs : (N, 2)         last absolute (x, y) of ego history
meta_recordingId / trackId / t0_frame : (N,)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Importance formula
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    I_x = exp(-(dx_time^2 / (2*sigma_x^2)) - beta_x * dl^2)
    I_y = exp(-(dl^2      / (2*sigma_y^2)) - beta_y * dx_time^2)
    I   = sqrt(I_x * I_y)

    dl mode (--importance_dl_mode):
        "dy"      ->  dl = nb_y - ego_y            (continuous, default)
        "lane_id" ->  dl = |nb_lane_id - ego_lane_id|  (integer lane diff)

    Default params:  sigma_x=5.0, beta_x=0.7, sigma_y=5.0, beta_y=0.2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 전체 실행 (권장)
  python highd_pipeline.py \\
      --raw_dir  data/highD/raw \\
      --mmap_dir data/highD/mmap \\
      --calc_stats

  # stats만 별도 계산
  python highd_pipeline.py --stage stats --mmap_dir data/highD/mmap
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

CLASS_VOCAB = ["car", "truck", "van", "bus", "motorcycle", "bicycle", "pedestrian", "other"]

EGO_DIM = 6    # x, y, xV, yV, xA, yA
NB_DIM  = 12   # dx, dy, dvx, dvy, dax, day, lc_state, dx_time, gate, I_x, I_y, I
K       = 8    # neighbor slots
C       = len(CLASS_VOCAB)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # paths
    raw_dir:  Path = Path("data/highD/raw")
    mmap_dir: Path = Path("data/highD/mmap")

    # recording
    target_hz:          float = 3.0
    history_sec:        float = 2.0
    future_sec:         float = 5.0
    stride_sec:         float = 1.0
    normalize_upper_xy: bool  = True
    recording_offset:   int   = 100
    min_speed_mps:      float = 0.0

    # lc / gating
    t_front:  float = 3.0
    t_back:   float = 5.0
    vy_eps:   float = 0.27
    eps_gate: float = 0.1

    # importance
    sigma_x: float = 5.0
    beta_x:  float = 0.7
    sigma_y: float = 5.0
    beta_y:  float = 0.2
    importance_dl_mode: str = "dy"   # "dy" | "lane_id"

    # output
    calc_stats: bool = False
    dry_run:    bool = False

    # stage
    stage: str = "all"   # "all" | "raw2mmap" | "stats"


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


def update_welford(
    count: int, mean: np.ndarray, m2: np.ndarray, data: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    n = data.shape[0]
    if n == 0:
        return count, mean, m2
    data       = data.astype(np.float64)
    batch_mean = data.mean(axis=0)
    batch_m2   = ((data - batch_mean) ** 2).sum(axis=0)
    delta      = batch_mean - mean
    new_count  = count + n
    new_mean   = mean + delta * (n / new_count)
    new_m2     = m2 + batch_m2 + (delta ** 2) * (count * n / new_count)
    return new_count, new_mean, new_m2


def finalize_stats(
    count: int, mean: np.ndarray, m2: np.ndarray, threshold: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    if count < 2:
        return mean.astype(np.float32), np.ones_like(mean, dtype=np.float32)
    var = m2 / (count - 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    low = np.where(std < threshold)[0]
    if len(low):
        print(f"    [Fix] {len(low)} low-variance features -> mean=0, std=1.")
        mean[low] = 0.0
        std[low]  = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _save_stats(
    out_dir: Path,
    ego_cnt: int, ego_mean: np.ndarray, ego_m2: np.ndarray,
    nb_cnt:  int, nb_mean:  np.ndarray, nb_m2:  np.ndarray,
) -> None:
    m_ego, s_ego = finalize_stats(ego_cnt, ego_mean, ego_m2)
    m_nb,  s_nb  = finalize_stats(nb_cnt,  nb_mean,  nb_m2)
    path = out_dir / "stats.npz"
    np.savez(path, ego_mean=m_ego, ego_std=s_ego, nb_mean=m_nb, nb_std=s_nb)
    print(f"  [stats] saved -> {path}")
    print(f"          ego_dim={len(m_ego)}, nb_dim={len(m_nb)}, "
          f"ego_count={ego_cnt}, nb_count={nb_cnt}")


# ─────────────────────────────────────────────────────────────────────────────
# Importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_importance(dx_time: float, dl: float, cfg: Config) -> Tuple[float, float, float]:
    """
    Parameters
    ----------
    dx_time : longitudinal time-to-reach  (dx / (dvx +- eps))
    dl      : lateral distance proxy
                cfg.importance_dl_mode == "dy"      -> dl = nb_y - ego_y
                cfg.importance_dl_mode == "lane_id" -> dl = |nb_lane_id - ego_lane_id|

    Returns
    -------
    I_x, I_y, I_total
    """
    ix      = float(np.exp(-(dx_time ** 2) / (2.0 * cfg.sigma_x ** 2) - cfg.beta_x * dl ** 2))
    iy      = float(np.exp(-(dl ** 2)      / (2.0 * cfg.sigma_y ** 2) - cfg.beta_y * dx_time ** 2))
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


def class_onehot(name) -> np.ndarray:
    out = np.zeros(C, dtype=np.float32)
    s   = name.strip().lower() if isinstance(name, str) else "other"
    s   = s if s in {"car", "truck"} else "other"
    out[CLASS_VOCAB.index(s)] = 1.0
    return out


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
    """
    Process one recording and return a dict of stacked arrays ready to write
    into the mmap.  Returns None if no samples could be produced.
    """
    raw_dir  = cfg.raw_dir
    rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    trk_meta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")

    C_y, frame_rate, upper_mark, lower_mark = flip_constants(rec_meta)
    step   = max(1, int(round(frame_rate / cfg.target_hz)))
    T      = int(round(cfg.history_sec  * cfg.target_hz))
    Tf     = int(round(cfg.future_sec   * cfg.target_hz))
    stride = max(1, int(round(cfg.stride_sec * cfg.target_hz)))

    # ── ensure required columns exist ──────────────────────────────────────
    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns: tracks[c] = 0
    for c in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
        if c not in tracks.columns: tracks[c] = 0.0
    if "laneId" not in tracks.columns: tracks["laneId"] = 0
    for c in ["dhw", "thw", "ttc"]:
        if c not in tracks.columns: tracks[c] = 0.0

    # ── per-vehicle lookups ─────────────────────────────────────────────────
    vid_to_dd  = dict(zip(trk_meta["id"].astype(int), trk_meta["drivingDirection"].astype(int)))
    vid_to_w   = dict(zip(trk_meta["id"].astype(int), trk_meta["width"].astype(float)))
    vid_to_l   = dict(zip(trk_meta["id"].astype(int), trk_meta["height"].astype(float)))
    vid_to_cls = dict(zip(trk_meta["id"].astype(int), trk_meta["class"].astype(str)))

    # ── lane tables ─────────────────────────────────────────────────────────
    upper_for_calc = upper_mark.copy()
    if cfg.normalize_upper_xy and len(upper_for_calc):
        upper_for_calc = np.sort((C_y - upper_for_calc).astype(np.float32))
    upper_center, upper_width = build_lane_tables(upper_for_calc)
    lower_center, lower_width = build_lane_tables(lower_mark)
    upper_mm = (1, int(len(upper_center))) if len(upper_center) else None

    # ── convert tracks to numpy ─────────────────────────────────────────────
    frame   = tracks["frame"].astype(np.int32).to_numpy()
    vid     = tracks["id"].astype(np.int32).to_numpy()
    x       = tracks["x"].astype(np.float32).to_numpy()
    y       = tracks["y"].astype(np.float32).to_numpy()
    # top-left bbox -> center
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

    # ── per-vehicle frame maps ───────────────────────────────────────────────
    per_vid_rows:         Dict[int, np.ndarray]     = {}
    per_vid_frame_to_row: Dict[int, Dict[int, int]] = {}
    for v, idxs in tracks.groupby("id").indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame[idxs])]
        per_vid_rows[int(v)] = idxs
        per_vid_frame_to_row[int(v)] = {int(fr): int(r)
                                        for fr, r in zip(frame[idxs], idxs)}

    # ── lane-change flags ───────────────────────────────────────────────────
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

    # ── sanitise safety metrics ──────────────────────────────────────────────
    dhw = tracks["dhw"].astype(np.float32).to_numpy()
    thw = tracks["thw"].astype(np.float32).to_numpy()
    ttc = tracks["ttc"].astype(np.float32).to_numpy()
    ttc = np.clip(np.where((~np.isfinite(ttc)) | (ttc < 0.0), -1.0, ttc), -1.0,  90.0)
    thw = np.clip(np.where((~np.isfinite(thw)) | (thw <= 0.5), -1.0, thw), -1.0,  20.0)
    dhw = np.clip(np.where((~np.isfinite(dhw)) | (dhw <= 10.0), -1.0, dhw), -1.0, 150.0)

    # ── sample loop ─────────────────────────────────────────────────────────
    x_ego_list:      List[np.ndarray] = []
    ego_safety_list: List[np.ndarray] = []
    y_fut_list:      List[np.ndarray] = []
    y_vel_list:      List[np.ndarray] = []
    y_acc_list:      List[np.ndarray] = []
    x_nb_list:       List[np.ndarray] = []
    nb_mask_list:    List[np.ndarray] = []
    ego_static_list: List[np.ndarray] = []
    nb_static_list:  List[np.ndarray] = []
    recid_list:      List[int] = []
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

            # speed filter
            if float(np.nanmean(np.sqrt(exv**2 + eyv**2))) < cfg.min_speed_mps:
                t0_frame += stride * step
                continue

            # ── ego_safety ──────────────────────────────────────────────────
            lead_ids    = nb_ids_all[ego_rows, 0]
            lead_exists = (lead_ids > 0).astype(np.float32)
            l_dhw = np.where(lead_exists > 0, dhw[ego_rows], 0.0).astype(np.float32)
            l_thw = np.where(lead_exists > 0, thw[ego_rows], 0.0).astype(np.float32)
            l_ttc = np.where(lead_exists > 0, ttc[ego_rows], 0.0).astype(np.float32)
            l_dv  = np.zeros(T, np.float32)
            for i, (hf, lid) in enumerate(zip(hist_frames, lead_ids.tolist())):
                if lid <= 0: continue
                rm = per_vid_frame_to_row.get(int(lid))
                if rm is None: continue
                r = rm.get(int(hf))
                if r is None: continue
                l_dv[i] = float(xv[r] - exv[i])
            l_dv = np.where(lead_exists > 0, l_dv, 0.0).astype(np.float32)
            ego_safety = np.stack([l_dhw, l_dv, l_thw, l_ttc, lead_exists], axis=1)

            # ── x_ego : (T, EGO_DIM=6) ─────────────────────────────────────
            x_ego = np.stack([ex, ey, exv, eyv, exa, eya], axis=1).astype(np.float32)

            # ── future ──────────────────────────────────────────────────────
            y_fut = np.stack([x[fut_rows],  y[fut_rows]],  axis=1).astype(np.float32)
            y_vel = np.stack([xv[fut_rows], yv[fut_rows]], axis=1).astype(np.float32)
            y_acc = np.stack([xa[fut_rows], ya[fut_rows]], axis=1).astype(np.float32)

            # ── neighbor history : (T, K, NB_DIM=12) ───────────────────────
            x_nb      = np.zeros((T, K, NB_DIM), np.float32)
            nb_mask   = np.zeros((T, K), bool)
            nb_static = np.zeros((T, K, 2 + C), np.float32)
            ego_lane_arr = lane_id[ego_rows].astype(np.int32)

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
                    rel    = nb_vec - ego_vec   # [dx, dy, dvx, dvy, dax, day]
                    x_nb[ti, ki, 0:6] = rel
                    nb_mask[ti, ki]   = True

                    # lc_state
                    if ki < 2:
                        lc_state = 0.0
                    else:
                        vyn = float(yv[r])
                        if ki < 5:  # left group
                            lc_state = -1.0 if vyn >  cfg.vy_eps else (-3.0 if vyn < -cfg.vy_eps else -2.0)
                        else:       # right group
                            lc_state =  1.0 if vyn < -cfg.vy_eps else ( 3.0 if vyn >  cfg.vy_eps else  2.0)

                    # dx_time + gate
                    dx  = float(rel[0])
                    dvx = float(rel[2])
                    dx_time = dx / (dvx + (cfg.eps_gate if dvx >= 0 else -cfg.eps_gate))
                    gate    = 1.0 if (-cfg.t_back < dx_time < cfg.t_front) else 0.0

                    # importance
                    # dl mode:
                    #   "dy"      -> dl = nb_y - ego_y  (continuous lateral offset, default)
                    #   "lane_id" -> dl = |nb_lane_id - ego_lane_id|  (integer lane diff)
                    #   실험 전환 시 --importance_dl_mode 인자만 바꾸면 됩니다.
                    if cfg.importance_dl_mode == "lane_id":
                        dl = float(abs(int(lane_id[r]) - int(ego_lane_arr[ti])))
                    else:  # "dy"
                        dl = float(rel[1])

                    ix, iy, i_total = compute_importance(dx_time, dl, cfg)

                    x_nb[ti, ki, 6]  = lc_state
                    x_nb[ti, ki, 7]  = dx_time
                    x_nb[ti, ki, 8]  = gate
                    x_nb[ti, ki, 9]  = ix
                    x_nb[ti, ki, 10] = iy
                    x_nb[ti, ki, 11] = i_total

                    # nb_static
                    nb_static[ti, ki, 0]  = float(vid_to_w.get(nid, 0.0))
                    nb_static[ti, ki, 1]  = float(vid_to_l.get(nid, 0.0))
                    nb_static[ti, ki, 2:] = class_onehot(vid_to_cls.get(nid, "other"))

            # ego_static
            ego_static = np.zeros(2 + C, np.float32)
            ego_static[0]  = float(vid_to_w.get(v, 0.0))
            ego_static[1]  = float(vid_to_l.get(v, 0.0))
            ego_static[2:] = class_onehot(vid_to_cls.get(v, "other"))

            x_ego_list.append(x_ego)
            ego_safety_list.append(ego_safety)
            y_fut_list.append(y_fut)
            y_vel_list.append(y_vel)
            y_acc_list.append(y_acc)
            x_nb_list.append(x_nb)
            nb_mask_list.append(nb_mask)
            ego_static_list.append(ego_static)
            nb_static_list.append(nb_static)
            recid_list.append(int(rec_id) + cfg.recording_offset)
            trackid_list.append(int(v))
            t0_list.append(int(t0_frame))

            t0_frame += stride * step

    if not x_ego_list:
        print(f"  [WARN] {rec_id}: no samples produced.")
        return None

    print(f"  [{rec_id}] {len(x_ego_list)} samples")
    return {
        "x_ego":       _safe_float(np.stack(x_ego_list,      0)),
        "ego_safety":  _safe_float(np.stack(ego_safety_list, 0)),
        "y":           _safe_float(np.stack(y_fut_list,       0)),
        "y_vel":       _safe_float(np.stack(y_vel_list,       0)),
        "y_acc":       _safe_float(np.stack(y_acc_list,       0)),
        "x_nb":        _safe_float(np.stack(x_nb_list,        0)),
        "nb_mask":     np.stack(nb_mask_list,    0),
        "ego_static":  _safe_float(np.stack(ego_static_list, 0)),
        "nb_static":   _safe_float(np.stack(nb_static_list,  0)),
        "recordingId": np.array(recid_list,   np.int32),
        "trackId":     np.array(trackid_list, np.int32),
        "t0_frame":    np.array(t0_list,      np.int32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE: raw -> mmap
# ─────────────────────────────────────────────────────────────────────────────

def stage_raw2mmap(cfg: Config) -> None:
    rec_ids = find_recording_ids(cfg.raw_dir)
    if not rec_ids:
        raise FileNotFoundError(f"No recordings found in {cfg.raw_dir}")
    print(f"[Stage] raw -> mmap  |  {len(rec_ids)} recordings  |  mmap_dir={cfg.mmap_dir}")

    # ── pass 1: collect all buffers (parallel per recording) ────────────────
    bufs: List[Dict[str, np.ndarray]] = []
    with concurrent.futures.ProcessPoolExecutor() as exe:
        futs = {exe.submit(_recording_to_buf, cfg, rid): rid for rid in rec_ids}
        for fut in tqdm(concurrent.futures.as_completed(futs),
                        total=len(rec_ids), desc="Processing recordings"):
            result = fut.result()
            if result is not None:
                bufs.append(result)

    if not bufs:
        raise RuntimeError("No samples produced from any recording.")

    total = sum(b["x_ego"].shape[0] for b in bufs)
    print(f"  total samples: {total}")

    if cfg.dry_run:
        print("[DRY RUN] No files written.")
        return

    # ── allocate memmaps ─────────────────────────────────────────────────────
    out = cfg.mmap_dir
    out.mkdir(parents=True, exist_ok=True)

    s0 = bufs[0]   # shape reference
    fp = {
        "x_ego":      open_memmap(out / "x_ego.npy",      "w+", "float32", (total, *s0["x_ego"].shape[1:])),
        "y":          open_memmap(out / "y.npy",           "w+", "float32", (total, *s0["y"].shape[1:])),
        "y_vel":      open_memmap(out / "y_vel.npy",       "w+", "float32", (total, *s0["y_vel"].shape[1:])),
        "y_acc":      open_memmap(out / "y_acc.npy",       "w+", "float32", (total, *s0["y_acc"].shape[1:])),
        "x_nb":       open_memmap(out / "x_nb.npy",        "w+", "float32", (total, *s0["x_nb"].shape[1:])),
        "nb_mask":    open_memmap(out / "nb_mask.npy",     "w+", "bool",    (total, *s0["nb_mask"].shape[1:])),
        "ego_static": open_memmap(out / "ego_static.npy",  "w+", "float32", (total, *s0["ego_static"].shape[1:])),
        "nb_static":  open_memmap(out / "nb_static.npy",   "w+", "float32", (total, *s0["nb_static"].shape[1:])),
        "ego_safety": open_memmap(out / "ego_safety.npy",  "w+", "float32", (total, *s0["ego_safety"].shape[1:])),
        "x_last_abs": open_memmap(out / "x_last_abs.npy",  "w+", "float32", (total, 2)),
    }
    meta_rec   = np.zeros(total, np.int32)
    meta_track = np.zeros(total, np.int32)
    meta_frame = np.zeros(total, np.int32)

    ego_dim = s0["x_ego"].shape[-1]
    nb_dim  = s0["x_nb"].shape[-1]
    s_ego_mean, s_ego_m2, s_ego_cnt = np.zeros(ego_dim, np.float64), np.zeros(ego_dim, np.float64), 0
    s_nb_mean,  s_nb_m2,  s_nb_cnt  = np.zeros(nb_dim,  np.float64), np.zeros(nb_dim,  np.float64), 0

    # ── pass 2: write buffers to mmap ────────────────────────────────────────
    cursor = 0
    for buf in tqdm(bufs, desc="Writing mmap"):
        n   = buf["x_ego"].shape[0]
        end = cursor + n

        for key in ["x_ego", "y", "y_vel", "y_acc", "x_nb", "nb_mask",
                    "ego_static", "nb_static", "ego_safety"]:
            fp[key][cursor:end] = buf[key]

        fp["x_last_abs"][cursor:end] = buf["x_ego"][:, -1, 0:2]

        meta_rec[cursor:end]   = buf["recordingId"]
        meta_track[cursor:end] = buf["trackId"]
        meta_frame[cursor:end] = buf["t0_frame"]

        if cfg.calc_stats:
            s_ego_cnt, s_ego_mean, s_ego_m2 = update_welford(
                s_ego_cnt, s_ego_mean, s_ego_m2,
                buf["x_ego"].reshape(-1, ego_dim)
            )
            nb_flat   = buf["x_nb"].reshape(-1, nb_dim)
            mask_flat = buf["nb_mask"].reshape(-1)
            valid_nb  = nb_flat[mask_flat]
            if valid_nb.shape[0]:
                s_nb_cnt, s_nb_mean, s_nb_m2 = update_welford(
                    s_nb_cnt, s_nb_mean, s_nb_m2, valid_nb.astype(np.float64)
                )

        cursor = end

    # ── flush + save meta ────────────────────────────────────────────────────
    for arr in fp.values():
        arr.flush()
    np.save(out / "meta_recordingId.npy", meta_rec)
    np.save(out / "meta_trackId.npy",     meta_track)
    np.save(out / "meta_frame.npy",       meta_frame)
    print(f"  [OK] mmap_dir={out}")

    if cfg.calc_stats:
        _save_stats(out, s_ego_cnt, s_ego_mean, s_ego_m2,
                        s_nb_cnt,  s_nb_mean,  s_nb_m2)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE: stats (standalone)
# ─────────────────────────────────────────────────────────────────────────────

def stage_stats(cfg: Config) -> None:
    out = cfg.mmap_dir
    print(f"[Stage] stats  |  mmap_dir={out}")
    fp_ego  = np.load(out / "x_ego.npy",   mmap_mode="r")
    fp_nb   = np.load(out / "x_nb.npy",    mmap_mode="r")
    fp_mask = np.load(out / "nb_mask.npy", mmap_mode="r")

    N, T, ego_dim         = fp_ego.shape
    _N, _T, _K, nb_dim    = fp_nb.shape
    BS = 1024

    s_ego_mean, s_ego_m2, s_ego_cnt = np.zeros(ego_dim, np.float64), np.zeros(ego_dim, np.float64), 0
    s_nb_mean,  s_nb_m2,  s_nb_cnt  = np.zeros(nb_dim,  np.float64), np.zeros(nb_dim,  np.float64), 0

    for start in tqdm(range(0, N, BS), desc="Computing stats"):
        end  = min(start + BS, N)
        b    = end - start
        s_ego_cnt, s_ego_mean, s_ego_m2 = update_welford(
            s_ego_cnt, s_ego_mean, s_ego_m2,
            fp_ego[start:end].reshape(-1, ego_dim)
        )
        nb_flat   = fp_nb[start:end].reshape(b * _T * _K, nb_dim)
        mask_flat = fp_mask[start:end].reshape(b * _T * _K)
        valid_nb  = nb_flat[mask_flat].astype(np.float64)
        if valid_nb.shape[0]:
            s_nb_cnt, s_nb_mean, s_nb_m2 = update_welford(
                s_nb_cnt, s_nb_mean, s_nb_m2, valid_nb
            )

    _save_stats(out, s_ego_cnt, s_ego_mean, s_ego_m2,
                    s_nb_cnt,  s_nb_mean,  s_nb_m2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="HighD preprocessing pipeline  (raw CSV -> mmap, no intermediate npz)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--stage",    default="all",  choices=["all", "raw2mmap", "stats"])
    ap.add_argument("--raw_dir",  default="data/highD/raw")
    ap.add_argument("--mmap_dir", default="data/highD/mmap")

    # recording
    ap.add_argument("--target_hz",          type=float, default=3.0)
    ap.add_argument("--history_sec",        type=float, default=2.0)
    ap.add_argument("--future_sec",         type=float, default=5.0)
    ap.add_argument("--stride_sec",         type=float, default=1.0)
    ap.add_argument("--normalize_upper_xy", action="store_true", default=True)
    ap.add_argument("--recording_offset",   type=int,   default=100)
    ap.add_argument("--min_speed_mps",      type=float, default=0.0)

    # lc / gating
    ap.add_argument("--t_front",  type=float, default=3.0)
    ap.add_argument("--t_back",   type=float, default=5.0)
    ap.add_argument("--vy_eps",   type=float, default=0.27)
    ap.add_argument("--eps_gate", type=float, default=0.1)

    # importance
    ap.add_argument("--sigma_x", type=float, default=5.0)
    ap.add_argument("--beta_x",  type=float, default=0.7)
    ap.add_argument("--sigma_y", type=float, default=5.0)
    ap.add_argument("--beta_y",  type=float, default=0.2)
    ap.add_argument("--importance_dl_mode", default="dy", choices=["dy", "lane_id"],
                    help="'dy': dl=nb_y-ego_y (continuous)  |  'lane_id': dl=|nb_lane_id-ego_lane_id|")

    # output
    ap.add_argument("--calc_stats", action="store_true",
                    help="Compute and save stats.npz during raw2mmap stage.")
    ap.add_argument("--dry_run", action="store_true")

    a = ap.parse_args()
    return Config(
        raw_dir  = Path(a.raw_dir),
        mmap_dir = Path(a.mmap_dir),
        target_hz          = a.target_hz,
        history_sec        = a.history_sec,
        future_sec         = a.future_sec,
        stride_sec         = a.stride_sec,
        normalize_upper_xy = a.normalize_upper_xy,
        recording_offset   = a.recording_offset,
        min_speed_mps      = a.min_speed_mps,
        t_front  = a.t_front,
        t_back   = a.t_back,
        vy_eps   = a.vy_eps,
        eps_gate = a.eps_gate,
        sigma_x  = a.sigma_x,
        beta_x   = a.beta_x,
        sigma_y  = a.sigma_y,
        beta_y   = a.beta_y,
        importance_dl_mode = a.importance_dl_mode,
        calc_stats = a.calc_stats,
        dry_run    = a.dry_run,
        stage      = a.stage,
    )


def main() -> None:
    cfg = parse_args()
    if cfg.stage in ("all", "raw2mmap"):
        stage_raw2mmap(cfg)
    if cfg.stage == "stats":
        stage_stats(cfg)
    if cfg.stage == "all" and not cfg.calc_stats:
        print("\n[INFO] Stats not computed. Re-run with --calc_stats or --stage stats.")


if __name__ == "__main__":
    main()