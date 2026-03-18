#!/usr/bin/env python3
"""
preprocess_exid.py  —  exiD preprocessing pipeline  (raw CSV → mmap)

STAGE raw2mmap :  exiD raw CSV  →  memory-mapped arrays

stats 계산은 train.py / evaluate.py 실행 시 src/stats.py 가 자동으로 수행합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature schema  (highD preprocess.py 와 동일한 output format)
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
    idx  9  gate      1 if (I_x >= theta_x) OR (I_y >= theta_y) else 0  [theta = P85]
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
    lis_mode: str = '3'   # '3' | '5' | '7' | '9'

    # importance mode
    importance_mode: str = 'lis'  # 'lis' | 'lit'

    # importance gate
    gate_theta: float = 0.0   # 0.0 = legacy time-window gate

    # lc_state version
    lc_version: str = "v2"    # "v1" | "v2"

    # exiD-specific: VRU 필터링
    drop_vru: bool = True     # VRU (motorcycle/bicycle/pedestrian) 윈도우 제거

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
# Per-recording processing  (raw CSV -> list of sample dicts)
# ─────────────────────────────────────────────────────────────────────────────

def _recording_to_buf(cfg: Config, rec_id: str) -> Optional[Dict[str, np.ndarray]]:
    raw_dir  = cfg.raw_path
    rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    trk_meta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")

    # ── 프레임레이트 / 윈도우 파라미터 ────────────────────────────────────────
    frame_rate = get_frame_rate(rec_meta)
    step   = max(1, int(round(frame_rate / cfg.target_hz)))
    T      = int(round(cfg.history_sec  * cfg.target_hz))
    Tf     = int(round(cfg.future_sec   * cfg.target_hz))
    stride = max(1, int(round(cfg.stride_sec * cfg.target_hz)))

    # ── 필수 컬럼 체크 / 기본값 주입 ─────────────────────────────────────────
    # exiD neighbor 컬럼
    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns:
            tracks[c] = -1   # exiD는 0 대신 -1을 missing 값으로 사용

    # exiD 속도/가속도 컬럼 (lonVelocity/latVelocity/lonAcceleration/latAcceleration)
    for src, dst in [
        ("lonVelocity",     "lonVelocity"),
        ("latVelocity",     "latVelocity"),
        ("lonAcceleration", "lonAcceleration"),
        ("latAcceleration", "latAcceleration"),
    ]:
        if src not in tracks.columns:
            tracks[dst] = 0.0

    if "laneletId" not in tracks.columns:
        tracks["laneletId"] = -1

    # ── VRU 클래스 맵 ─────────────────────────────────────────────────────────
    class_map = get_class_map(trk_meta)

    # ── 차량 크기 맵 (exiD: tracks 행 자체에 width/length가 있음) ──────────────
    # highD는 tracksMeta의 width(=차량길이)/height(=차량너비)를 사용했지만,
    # exiD는 tracks CSV의 각 행에 width/length가 직접 기록됨.
    # 여기서는 첫 번째 등장 행의 값을 차량 고유 크기로 사용합니다.
    has_width  = "width"  in tracks.columns
    has_length = "length" in tracks.columns

    # ── NumPy 배열 추출 ───────────────────────────────────────────────────────
    # exiD는 xCenter/yCenter (중심 좌표) → highD처럼 +0.5*size 보정 불필요
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

    # 차량 크기 배열 (per-row)
    width_arr  = _safe_float(tracks["width"].to_numpy(np.float32),  0.0) if has_width  else np.zeros(len(tracks), np.float32)
    length_arr = _safe_float(tracks["length"].to_numpy(np.float32), 0.0) if has_length else np.zeros(len(tracks), np.float32)

    # ── exiD는 drivingDirection flip 없음 ─────────────────────────────────────
    # highD에서는 방향 1(왼쪽) 차량들을 flip하여 모든 차량이 오른쪽 방향으로
    # 주행하도록 정규화했지만, exiD는 좌표계가 이미 통일되어 있습니다.

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
    vid_to_w: Dict[int, float] = {}   # width  (exiD: 차량 폭)
    vid_to_l: Dict[int, float] = {}   # length (exiD: 차량 길이, lit 계산에 사용)
    for v, idxs in per_vid_rows.items():
        r0 = int(idxs[0])
        vid_to_w[int(v)] = float(width_arr[r0])
        vid_to_l[int(v)] = float(length_arr[r0])

    # ── VRU 판별 함수 ─────────────────────────────────────────────────────────
    def is_vru(tid: int) -> bool:
        return class_map.get(int(tid), "other") in VRU_CLASSES

    # ── neighbor ID 배열 (N, 8) ───────────────────────────────────────────────
    # exiD neighbor 컬럼은 세미콜론으로 여러 값이 들어올 수 있으므로 첫 값만 사용
    nb_id_cols = []
    for c in NEIGHBOR_COLS_8:
        s = tracks[c].astype(str).str.strip().str.split(";").str[0]
        nb_id_cols.append(pd.to_numeric(s, errors="coerce").fillna(-1).astype(np.int32).to_numpy())
    nb_ids_all = np.stack(nb_id_cols, axis=1)   # (N, 8)

    # ── sample 수집 루프 ──────────────────────────────────────────────────────
    x_ego_list:   List[np.ndarray] = []
    y_fut_list:   List[np.ndarray] = []
    y_vel_list:   List[np.ndarray] = []
    y_acc_list:   List[np.ndarray] = []
    x_nb_list:    List[np.ndarray] = []
    nb_mask_list: List[np.ndarray] = []
    trackid_list: List[int] = []
    t0_list:      List[int] = []

    for v, idxs in per_vid_rows.items():
        frs = frame[idxs]
        if len(frs) < (T + Tf) * step:
            continue

        # exiD: VRU ego 차량 필터링
        if cfg.drop_vru and is_vru(int(v)):
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

            x_ego = np.stack(
                [ex, ey, exv, eyv, exa, eya], axis=1
            ).astype(np.float32)

            y_fut = np.stack([x[fut_rows],  y[fut_rows]],  axis=1).astype(np.float32)
            y_vel = np.stack([xv[fut_rows], yv[fut_rows]], axis=1).astype(np.float32)
            y_acc = np.stack([xa[fut_rows], ya[fut_rows]], axis=1).astype(np.float32)

            x_nb    = np.zeros((T, K, NB_DIM), np.float32)
            nb_mask = np.zeros((T, K), bool)

            # exiD: LIT 계산에 차량 길이(length)를 사용 (highD에서 width를 썼던 이유:
            # highD는 주행 방향이 x축이고 width가 차량 '길이'였음.
            # exiD는 lonVelocity가 x축이므로 length가 종방향 크기에 해당함)
            len_ego = float(vid_to_l.get(v, 0.0))

            for ti, hf in enumerate(hist_frames):
                ego_vec = np.array([ex[ti], ey[ti], exv[ti], eyv[ti], exa[ti], eya[ti]], np.float32)
                ids8    = nb_ids_all[ego_rows[ti]]

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

                    # exiD: neighbor VRU 필터링 (ego가 아닌 neighbor에 적용)
                    if cfg.drop_vru and is_vru(nid):
                        continue

                    nb_vec = np.array([x[r], y[r], xv[r], yv[r], xa[r], ya[r]], np.float32)
                    rel    = nb_vec - ego_vec
                    x_nb[ti, ki, 0:6] = rel
                    nb_mask[ti, ki]   = True

                    # ── lc_state (highD와 동일한 v1/v2 로직) ───────────────────
                    # exiD slot 배열은 highD와 동일하게 구성됨:
                    #   0: leadId (same-lane front)
                    #   1: rearId (same-lane rear)
                    #   2~4: left group
                    #   5~7: right group
                    if cfg.lc_version == "v1":
                        vyn = float(yv[r])
                        if ki < 2:
                            lc_state = 0.0
                        elif ki < 5:   # left group
                            if   vyn >  cfg.vy_eps if hasattr(cfg, 'vy_eps') else 0.27: lc_state = -1.0
                            elif vyn < -(cfg.vy_eps if hasattr(cfg, 'vy_eps') else 0.27): lc_state = -3.0
                            else:                   lc_state = -2.0
                        else:          # right group
                            if   vyn < -(cfg.vy_eps if hasattr(cfg, 'vy_eps') else 0.27): lc_state =  1.0
                            elif vyn >  (cfg.vy_eps if hasattr(cfg, 'vy_eps') else 0.27): lc_state =  3.0
                            else:                   lc_state =  2.0
                    else:   # v2 (default, highD와 동일)
                        dy      = float(rel[1])
                        dvy     = float(rel[3])
                        abs_dvy = abs(dvy)

                        if ki < 2 and abs(dy) < cfg.dy_same:
                            if abs_dvy > cfg.dvy_eps_same:
                                lc_state = 2.0
                            else:
                                lc_state = 1.0
                        elif ki >= 2:
                            if abs_dvy > cfg.dvy_eps_cross:
                                lc_state = 0.0 if dy * dvy < 0 else 2.0
                            else:
                                lc_state = 1.0
                        else:
                            lc_state = 0.0 if dy * dvy < 0 else 2.0

                    # ── LIT 계산 ────────────────────────────────────────────────
                    dx  = float(rel[0])
                    dvx = float(rel[2])
                    len_nb   = float(vid_to_l.get(nid, 0.0))
                    half_sum = 0.5 * (len_ego + len_nb)
                    if dx >= 0:
                        gap        = abs(dx - half_sum)
                        denom_base = dvx
                    else:
                        gap        = abs(-dx - half_sum)
                        denom_base = -dvx
                    lit = gap / (denom_base + (cfg.eps_gate if denom_base >= 0 else -cfg.eps_gate))
                    lis = _lit_to_lis(lit, cfg.lis_mode)

                    # ── delta_lane: exiD는 laneletId가 정수이므로 단순 차이 사용 ──
                    # laneletId가 -1인 경우(미상) delta_lane=0으로 처리
                    ego_lid = int(ego_lane_arr[ti])
                    nb_lid  = int(lane_id[r])
                    if ego_lid >= 0 and nb_lid >= 0:
                        delta_lane = float(abs(nb_lid - ego_lid))
                    else:
                        delta_lane = 0.0

                    # ── importance 계산 (highD와 동일) ──────────────────────────
                    if cfg.importance_mode == 'lit':
                        ix, iy, i_total = compute_importance_lit(
                            lit, delta_lane, lc_state
                        )
                    else:
                        ix, iy, i_total = compute_importance_lis(
                            lis, delta_lane, lc_state
                        )

                    # ── gate ────────────────────────────────────────────────────
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
        dry_run     = a.dry_run,
        num_workers = a.num_workers,
    )


def main() -> None:
    cfg = parse_args()
    stage_raw2mmap(cfg)


if __name__ == "__main__":
    main()