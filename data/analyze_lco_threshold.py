#!/usr/bin/env python3
"""
analyze_lco_threshold.py — lco_norm & latVelocity 분포 분석 (v4 lc_state threshold X 결정용)

ego 차량의 LC 이벤트 기준 시간 창별로 아래 값의 분포를 비교한다.
  - lco_norm  = lco / (lane_width / 2)   (정규화 오프셋, 범위 약 ±1)
  - |lco_norm|
  - latVelocity  (부호: +→오른쪽, 단 exiD는 좌표계 반전 주의 — 아래 주석 참조)

시간 창 (다음 LC 이벤트까지 남은 시간):
  "-1~0"     : 0   < t_to_lc ≤ 1s
  "-2~-1"    : 1   < t_to_lc ≤ 2s
  "-3~-2"    : 2   < t_to_lc ≤ 3s
  "-4~-3"    : 3   < t_to_lc ≤ 4s
  "-5~-4"    : 4   < t_to_lc ≤ 5s
  "-4.5~-2.5": 2.5 < t_to_lc ≤ 4.5s
  "no_lc"    : 다음 5s 내 LC 없음 (baseline)

부호 규약:
  highD  : maybe_flip 이후 좌표 기준. +y = 우측, +latV = 우측 이동.
  exiD   : 원본 좌표 기준. +y = 좌측(y 작을수록 오른쪽).
           → latVelocity 부호는 exiD 에서 +가 좌측 이동을 의미할 수 있음.
           분포 형태(분산, 피크 위치) 비교에 집중할 것.

Usage:
    python data/analyze_lco_threshold.py --dataset exid
    python data/analyze_lco_threshold.py --dataset highd
    python data/analyze_lco_threshold.py --dataset both
    python data/analyze_lco_threshold.py --dataset both --out lco_analysis --num_workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_HZ         = 5.0   # 분석용 다운샘플 Hz
MAX_T_LC          = 5.0   # LC 예측 최대 horizon (s)
NO_LC_EXCLUDE_PRE = 5.0   # no_lc: 직전 LC 이후 이 시간이 지난 프레임만 포함

# (window_key, t_lo, t_hi)  — t_lo < t_to_lc ≤ t_hi
TIME_WINDOWS: List[Tuple[str, float, float]] = [
    ("-1~0",      0.0, 1.0),
    ("-2~-1",     1.0, 2.0),
    ("-3~-2",     2.0, 3.0),
    ("-4~-3",     3.0, 4.0),
    ("-5~-4",     4.0, 5.0),
    ("-4.5~-2.5", 2.5, 4.5),
]
ALL_WINDOW_KEYS = [w[0] for w in TIME_WINDOWS] + ["no_lc"]

COLORS = {
    "-1~0":       "#d62728",
    "-2~-1":      "#ff7f0e",
    "-3~-2":      "#e7c200",
    "-4~-3":      "#2ca02c",
    "-5~-4":      "#1f77b4",
    "-4.5~-2.5":  "#9467bd",
    "no_lc":      "#7f7f7f",
}
LINESTYLES = {
    "-1~0":       "-",
    "-2~-1":      "-",
    "-3~-2":      "-",
    "-4~-3":      "-",
    "-5~-4":      "-",
    "-4.5~-2.5":  "--",
    "no_lc":      ":",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def parse_semicolon_floats(s: str) -> List[float]:
    return [float(v) for v in str(s).strip().split(";") if v.strip()]


def find_recording_ids(raw_dir: Path) -> List[str]:
    return sorted(
        re.match(r"(\d+)_tracks\.csv$", p.name).group(1)
        for p in raw_dir.glob("*_tracks.csv")
        if re.match(r"(\d+)_tracks\.csv$", p.name)
    )


def empty_result() -> Dict[str, Dict[str, List[float]]]:
    return {k: {"lco_norm": [], "lat_v": []} for k in ALL_WINDOW_KEYS}


def merge_results(results: List[Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, List[float]]]:
    merged = empty_result()
    for res in results:
        for wkey in ALL_WINDOW_KEYS:
            merged[wkey]["lco_norm"].extend(res[wkey]["lco_norm"])
            merged[wkey]["lat_v"].extend(res[wkey]["lat_v"])
    return merged


def assign_window(t_to_lc: Optional[float]) -> Optional[str]:
    if t_to_lc is None:
        return "no_lc"
    for key, lo, hi in TIME_WINDOWS:
        if lo < t_to_lc <= hi:
            return key
    return None


def kde_safe(arr: np.ndarray, x_grid: np.ndarray, bw: float = 0.08) -> Optional[np.ndarray]:
    if len(arr) < 20:
        return None
    try:
        return gaussian_kde(arr, bw_method=bw)(x_grid)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# exiD — per-recording (top-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _process_exid_rec(args: Tuple[str, str]) -> Tuple[str, Dict[str, Dict[str, List[float]]], int]:
    """args = (raw_dir_str, rec_id)  →  (rec_id, result, n_samples)"""
    raw_dir = Path(args[0])
    rec_id  = args[1]

    rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv", low_memory=False)

    frame_rate = float(rec_meta.loc[0, "frameRate"]) if "frameRate" in rec_meta.columns else 25.0
    step = max(1, int(round(frame_rate / SAMPLE_HZ)))

    tracks = tracks.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)

    frame_arr = tracks["frame"].to_numpy(np.int32)

    # laneletId: 여러 lanelet에 걸쳐 있을 때 "123;456" 형태 → 첫 번째 값 사용
    # (int 변환 실패로 -1이 되면 LC 이벤트 검출이 누락됨)
    if "laneletId" in tracks.columns:
        _s = tracks["laneletId"].astype(str).str.strip().str.split(";").str[0]
        lane_arr = pd.to_numeric(_s, errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    else:
        lane_arr = np.full(len(tracks), -1, np.int32)

    # latLaneCenterOffset & laneWidth: 여러 값이 있을 때 |lco|가 최대인 lanelet의 값 사용
    # (lco와 laneWidth는 같은 인덱스 lanelet 것을 사용해야 올바르게 normalize됨)
    if "latLaneCenterOffset" in tracks.columns and "laneWidth" in tracks.columns:
        lco_raw = tracks["latLaneCenterOffset"].astype(str).tolist()
        lw_raw  = tracks["laneWidth"].astype(str).tolist()
        lco_lw_pairs: List[Tuple[float, float]] = []
        for lco_s, lw_s in zip(lco_raw, lw_raw):
            lcos = [float(v) for v in lco_s.strip().split(";") if v.strip() not in ("", "nan")]
            lws  = [float(v) for v in lw_s.strip().split(";")  if v.strip() not in ("", "nan")]
            if not lcos:
                lco_lw_pairs.append((0.0, 3.5))
                continue
            idx  = max(range(len(lcos)), key=lambda i: abs(lcos[i]))
            lco  = lcos[idx]
            lw   = lws[idx] if idx < len(lws) else (lws[0] if lws else 3.5)
            if not (lw > 0):
                lw = 3.5
            lco_lw_pairs.append((lco, lw))
        lco_arr = np.array([p[0] for p in lco_lw_pairs], np.float32)
        lw_arr  = np.array([p[1] for p in lco_lw_pairs], np.float32)
    elif "latLaneCenterOffset" in tracks.columns:
        _s = tracks["latLaneCenterOffset"].astype(str).str.strip().str.split(";").str[0]
        lco_arr = pd.to_numeric(_s, errors="coerce").fillna(0.0).to_numpy(np.float32)
        lw_arr  = np.full(len(tracks), 3.5, np.float32)
    else:
        lco_arr = np.zeros(len(tracks), np.float32)
        lw_arr  = np.full(len(tracks), 3.5, np.float32)

    lat_v_arr = (tracks["latVelocity"].to_numpy(np.float32)
                 if "latVelocity" in tracks.columns
                 else np.zeros(len(tracks), np.float32))

    # laneChange 컬럼: exiD 어노테이션 기반 LC 이벤트 (laneletId 변화보다 훨씬 정확)
    has_lc_col = "laneChange" in tracks.columns
    if has_lc_col:
        lc_col_arr = (tracks["laneChange"].fillna(0).astype(np.int32).to_numpy())

    per_vid_rows: Dict[int, np.ndarray] = {}
    per_vid_f2r:  Dict[int, Dict[int, int]] = {}
    for v, idxs in tracks.groupby("trackId").indices.items():
        idxs = np.array(sorted(idxs, key=lambda i: frame_arr[i]), np.int32)
        per_vid_rows[int(v)] = idxs
        per_vid_f2r[int(v)]  = {int(frame_arr[r]): int(r) for r in idxs}

    lc_frames: Dict[int, np.ndarray] = {}
    for v, idxs in per_vid_rows.items():
        if has_lc_col:
            # laneChange != 0인 연속 구간의 첫 번째 프레임을 LC 이벤트로 사용
            mask   = lc_col_arr[idxs] != 0
            starts = np.where(mask & ~np.concatenate([[False], mask[:-1]]))[0]
            lc_frames[v] = frame_arr[idxs[starts]] if len(starts) else np.array([], np.int32)
        else:
            lids = lane_arr[idxs]
            chg  = np.where((lids[1:] != lids[:-1]) & (lids[1:] >= 0) & (lids[:-1] >= 0))[0] + 1
            lc_frames[v] = frame_arr[idxs[chg]] if len(chg) else np.array([], np.int32)

    result = empty_result()
    n_samples = 0

    for v, idxs in per_vid_rows.items():
        frs   = frame_arr[idxs]
        lc_fs = np.sort(lc_frames.get(v, np.array([], np.int32)))
        f2r_v = per_vid_f2r[v]

        for sf in frs[::step]:
            r = f2r_v.get(sf)
            if r is None:
                continue
            lco = float(lco_arr[r])
            lw  = float(lw_arr[r])
            if lw < 0.5:
                continue
            lco_norm = lco / (lw * 0.5)
            lat_v    = float(lat_v_arr[r])

            future_lc = lc_fs[lc_fs > sf]
            t_to_lc   = ((future_lc[0] - sf) / frame_rate if len(future_lc) else None)
            if t_to_lc is not None and t_to_lc > MAX_T_LC:
                t_to_lc = None

            wkey = assign_window(t_to_lc)
            if wkey is None:
                continue

            if wkey == "no_lc":
                prev_lc = lc_fs[lc_fs <= sf]
                if len(prev_lc) and (sf - prev_lc[-1]) / frame_rate < NO_LC_EXCLUDE_PRE:
                    continue

            result[wkey]["lco_norm"].append(lco_norm)
            result[wkey]["lat_v"].append(lat_v)
            n_samples += 1

    return rec_id, result, n_samples


# ─────────────────────────────────────────────────────────────────────────────
# highD — per-recording (top-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _process_highd_rec(args: Tuple[str, str]) -> Tuple[str, Dict[str, Dict[str, List[float]]], int]:
    """args = (raw_dir_str, rec_id)  →  (rec_id, result, n_samples)"""
    raw_dir = Path(args[0])
    rec_id  = args[1]

    rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    trk_meta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")

    frame_rate = float(rec_meta.loc[0, "frameRate"]) if "frameRate" in rec_meta.columns else 25.0
    step = max(1, int(round(frame_rate / SAMPLE_HZ)))

    upper_mark = np.array(parse_semicolon_floats(str(rec_meta.loc[0, "upperLaneMarkings"])), np.float32)
    lower_mark = np.array(parse_semicolon_floats(str(rec_meta.loc[0, "lowerLaneMarkings"])), np.float32)
    n_upper    = len(upper_mark)

    vid_to_dd = dict(zip(trk_meta["id"].astype(int), trk_meta["drivingDirection"].astype(int)))
    vid_to_w  = dict(zip(trk_meta["id"].astype(int), trk_meta["width"].astype(float)))
    vid_to_h  = dict(zip(trk_meta["id"].astype(int), trk_meta["height"].astype(float)))

    tracks = tracks.sort_values(["id", "frame"], kind="mergesort").reset_index(drop=True)

    frame_arr = tracks["frame"].to_numpy(np.int32)
    vid_arr   = tracks["id"].astype(np.int32).to_numpy()
    y_arr     = tracks["y"].astype(np.float32).to_numpy().copy()
    yv_arr    = tracks["yVelocity"].astype(np.float32).to_numpy().copy()
    lane_arr  = tracks["laneId"].astype(np.int32).to_numpy()

    h_row  = np.array([vid_to_h.get(int(v), 0.0) for v in vid_arr], np.float32)
    y_arr += 0.5 * h_row

    dd_arr = np.array([vid_to_dd.get(int(v), 2) for v in vid_arr], np.int8)

    # lco / laneWidth 계산 (preprocess.py 부호 규약과 동일)
    lco_arr = np.zeros(len(y_arr), np.float32)
    lw_arr  = np.zeros(len(y_arr), np.float32)

    mask_up = (dd_arr == 1)
    j_up    = lane_arr - 2
    ok_up   = mask_up & (j_up >= 0) & (j_up < n_upper - 1)
    if ok_up.any():
        j = j_up[ok_up]
        c = 0.5 * (upper_mark[j] + upper_mark[j + 1])
        lco_arr[ok_up] = y_arr[ok_up] - c
        lw_arr[ok_up]  = np.abs(upper_mark[j + 1] - upper_mark[j])
    lco_arr[mask_up] *= -1.0  # pre-flip → post-flip sign

    mask_lo = (dd_arr == 2)
    j_lo    = lane_arr - n_upper - 2
    ok_lo   = mask_lo & (j_lo >= 0) & (j_lo < len(lower_mark) - 1)
    if ok_lo.any():
        j = j_lo[ok_lo]
        c = 0.5 * (lower_mark[j] + lower_mark[j + 1])
        lco_arr[ok_lo] = y_arr[ok_lo] - c
        lw_arr[ok_lo]  = np.abs(lower_mark[j + 1] - lower_mark[j])

    # latVelocity 부호: dd==1 → negate (post-flip +y = 우측)
    lat_v_arr = yv_arr.copy()
    lat_v_arr[dd_arr == 1] *= -1.0

    per_vid_rows: Dict[int, np.ndarray] = {}
    per_vid_f2r:  Dict[int, Dict[int, int]] = {}
    for v, idxs in tracks.groupby("id").indices.items():
        idxs = np.array(sorted(idxs, key=lambda i: frame_arr[i]), np.int32)
        per_vid_rows[int(v)] = idxs
        per_vid_f2r[int(v)]  = {int(frame_arr[r]): int(r) for r in idxs}

    lc_frames: Dict[int, np.ndarray] = {}
    for v, idxs in per_vid_rows.items():
        lids = lane_arr[idxs]
        chg  = np.where((lids[1:] != lids[:-1]) & (lids[1:] > 0) & (lids[:-1] > 0))[0] + 1
        lc_frames[v] = frame_arr[idxs[chg]] if len(chg) else np.array([], np.int32)

    result = empty_result()
    n_samples = 0

    for v, idxs in per_vid_rows.items():
        frs   = frame_arr[idxs]
        lc_fs = np.sort(lc_frames.get(v, np.array([], np.int32)))
        f2r_v = per_vid_f2r[v]

        for sf in frs[::step]:
            r = f2r_v.get(sf)
            if r is None:
                continue
            lco = float(lco_arr[r])
            lw  = float(lw_arr[r])
            if lw < 0.5:
                continue
            lco_norm = lco / (lw * 0.5)
            lat_v    = float(lat_v_arr[r])

            future_lc = lc_fs[lc_fs > sf]
            t_to_lc   = ((future_lc[0] - sf) / frame_rate if len(future_lc) else None)
            if t_to_lc is not None and t_to_lc > MAX_T_LC:
                t_to_lc = None

            wkey = assign_window(t_to_lc)
            if wkey is None:
                continue

            if wkey == "no_lc":
                prev_lc = lc_fs[lc_fs <= sf]
                if len(prev_lc) and (sf - prev_lc[-1]) / frame_rate < NO_LC_EXCLUDE_PRE:
                    continue

            result[wkey]["lco_norm"].append(lco_norm)
            result[wkey]["lat_v"].append(lat_v)
            n_samples += 1

    return rec_id, result, n_samples


# ─────────────────────────────────────────────────────────────────────────────
# Parallel collectors
# ─────────────────────────────────────────────────────────────────────────────

def collect_exid(raw_dir: Path, num_workers: int = 0) -> Dict[str, Dict[str, List[float]]]:
    rec_ids = find_recording_ids(raw_dir)
    if not rec_ids:
        raise FileNotFoundError(f"No exiD recordings in {raw_dir}")
    nw = num_workers or os.cpu_count() or 1
    print(f"  [exiD] {len(rec_ids)} recordings, {nw} workers")
    task_args = [(str(raw_dir), r) for r in rec_ids]
    with concurrent.futures.ProcessPoolExecutor(max_workers=nw) as ex:
        futures = {ex.submit(_process_exid_rec, a): a[1] for a in task_args}
        results = []
        for fut in concurrent.futures.as_completed(futures):
            rec_id, res, n = fut.result()
            print(f"    {rec_id} → {n:,} samples", flush=True)
            results.append(res)
    return merge_results(results)


def collect_highd(raw_dir: Path, num_workers: int = 0) -> Dict[str, Dict[str, List[float]]]:
    rec_ids = find_recording_ids(raw_dir)
    if not rec_ids:
        raise FileNotFoundError(f"No highD recordings in {raw_dir}")
    nw = num_workers or os.cpu_count() or 1
    print(f"  [highD] {len(rec_ids)} recordings, {nw} workers")
    task_args = [(str(raw_dir), r) for r in rec_ids]
    with concurrent.futures.ProcessPoolExecutor(max_workers=nw) as ex:
        futures = {ex.submit(_process_highd_rec, a): a[1] for a in task_args}
        results = []
        for fut in concurrent.futures.as_completed(futures):
            rec_id, res, n = fut.result()
            print(f"    {rec_id} → {n:,} samples", flush=True)
            results.append(res)
    return merge_results(results)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics summary
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(data: Dict[str, Dict[str, List[float]]], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  lco_norm 통계  (signed)")
    print(f"{'='*60}")
    print(f"  {'window':<14} {'N':>7}  {'min':>7}  {'max':>7}  {'mean':>6}  {'std':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}  {'p99':>6}")
    print(f"  {'-'*84}")
    for wkey in ALL_WINDOW_KEYS:
        arr = np.array(data[wkey]["lco_norm"], np.float32)
        if len(arr) == 0:
            continue
        print(f"  {wkey:<14}  {len(arr):>7,}  {arr.min():>7.3f}  {arr.max():>7.3f}  "
              f"{arr.mean():>6.3f}  {arr.std():>6.3f}  "
              f"{np.percentile(arr,50):>6.3f}  {np.percentile(arr,75):>6.3f}  "
              f"{np.percentile(arr,90):>6.3f}  {np.percentile(arr,99):>6.3f}")

    print(f"\n  {label}  |lco_norm| 통계")
    print(f"  {'window':<14} {'N':>7}  {'min':>7}  {'max':>7}  {'mean':>6}  {'std':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}  {'p99':>6}")
    print(f"  {'-'*84}")
    for wkey in ALL_WINDOW_KEYS:
        arr = np.abs(np.array(data[wkey]["lco_norm"], np.float32))
        if len(arr) == 0:
            continue
        print(f"  {wkey:<14}  {len(arr):>7,}  {arr.min():>7.3f}  {arr.max():>7.3f}  "
              f"{arr.mean():>6.3f}  {arr.std():>6.3f}  "
              f"{np.percentile(arr,50):>6.3f}  {np.percentile(arr,75):>6.3f}  "
              f"{np.percentile(arr,90):>6.3f}  {np.percentile(arr,99):>6.3f}")

    print(f"\n  {label}  |latVelocity| 통계")
    print(f"  {'window':<14} {'N':>7}  {'mean':>6}  {'std':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}  {'p99':>6}")
    print(f"  {'-'*64}")
    for wkey in ALL_WINDOW_KEYS:
        arr = np.abs(np.array(data[wkey]["lat_v"], np.float32))
        if len(arr) == 0:
            continue
        print(f"  {wkey:<14}  {len(arr):>7,}  {arr.mean():>6.3f}  {arr.std():>6.3f}  "
              f"{np.percentile(arr,50):>6.3f}  {np.percentile(arr,75):>6.3f}  "
              f"{np.percentile(arr,90):>6.3f}  {np.percentile(arr,99):>6.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset(
    ax_abs:    plt.Axes,
    ax_signed: plt.Axes,
    ax_latv:   plt.Axes,
    data: Dict[str, Dict[str, List[float]]],
    title: str,
    no_lc_subsample: int = 5,
) -> None:
    x_abs    = np.linspace(0.0,  1.5, 400)
    x_signed = np.linspace(-1.5, 1.5, 400)
    x_latv   = np.linspace(-2.0, 2.0, 400)

    for wkey in ALL_WINDOW_KEYS:
        lco_norm = np.array(data[wkey]["lco_norm"], np.float32)
        lat_v    = np.array(data[wkey]["lat_v"],    np.float32)
        if len(lco_norm) < 20:
            continue

        if wkey == "no_lc" and no_lc_subsample > 1:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(lco_norm), max(20, len(lco_norm) // no_lc_subsample), replace=False)
            lco_norm = lco_norm[idx]
            lat_v    = lat_v[idx]

        color = COLORS[wkey]
        ls    = LINESTYLES[wkey]
        alpha = 0.85 if wkey != "no_lc" else 0.55
        lw    = 2.0  if wkey != "no_lc" else 1.5

        y = kde_safe(np.abs(lco_norm), x_abs)
        if y is not None:
            ax_abs.plot(x_abs, y, color=color, ls=ls, lw=lw, alpha=alpha, label=wkey)

        y = kde_safe(lco_norm, x_signed)
        if y is not None:
            ax_signed.plot(x_signed, y, color=color, ls=ls, lw=lw, alpha=alpha, label=wkey)

        y = kde_safe(lat_v, x_latv)
        if y is not None:
            ax_latv.plot(x_latv, y, color=color, ls=ls, lw=lw, alpha=alpha, label=wkey)

    for ax, xlabel, title_suffix in [
        (ax_abs,    "|lco_norm|",         "|lco_norm| (threshold X 후보)"),
        (ax_signed, "lco_norm (signed)",  "lco_norm (signed)"),
        (ax_latv,   "latVelocity [+→right]  (m/s)", "latVelocity (signed)"),
    ]:
        ax.set_title(f"{title}\n{title_suffix}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.axvline(0 if "sign" in title_suffix or "lat" in title_suffix else -1,
                   color="k", lw=0.8, ls="--", alpha=0.3)
        ax.legend(fontsize=7)

    # threshold 후보 참조선
    for x in [0.5, 0.7, 0.9]:
        ax_abs.axvline(x, color="dimgray", lw=0.8, ls=":", alpha=0.45)
        ax_abs.text(x, ax_abs.get_ylim()[1] * 0.03, f"{x}", fontsize=6,
                    ha="center", color="dimgray")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset",     default="both", choices=["exid", "highd", "both"])
    ap.add_argument("--exid_dir",    default="data/exiD/raw")
    ap.add_argument("--highd_dir",   default="data/highD/raw")
    ap.add_argument("--out",         default="lco_threshold_analysis")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="worker 수 (0 = os.cpu_count())")
    args = ap.parse_args()

    datasets: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    if args.dataset in ("exid", "both"):
        print(f"\n[exiD] {args.exid_dir}")
        datasets["exiD"] = collect_exid(Path(args.exid_dir), args.num_workers)
        print_stats(datasets["exiD"], "exiD")

    if args.dataset in ("highd", "both"):
        print(f"\n[highD] {args.highd_dir}")
        datasets["highD"] = collect_highd(Path(args.highd_dir), args.num_workers)
        print_stats(datasets["highD"], "highD")

    # ── Figure ───────────────────────────────────────────────────────────────
    n_ds  = len(datasets)
    fig, axes = plt.subplots(n_ds, 3, figsize=(15, n_ds * 4))
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    for row, (ds_name, data) in enumerate(datasets.items()):
        plot_dataset(axes[row, 0], axes[row, 1], axes[row, 2], data, ds_name)

    fig.suptitle(
        "lco_norm & latVelocity 분포  (LC 이벤트 기준 시간 창별)\n"
        "v4 threshold X 결정용: |lco_norm| 열에서 LC 창과 no_lc 창이 분리되는 지점을 X 후보로 선택",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{args.out}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")

    # ── threshold X 요약 ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  threshold X 후보 요약  (|lco_norm| 통계)")
    print("="*60)
    for ds_name, data in datasets.items():
        no_lc = np.abs(np.array(data["no_lc"]["lco_norm"], np.float32))
        p90_no = float(np.percentile(no_lc, 90)) if len(no_lc) else float("nan")
        print(f"\n  [{ds_name}]  no_lc p90 = {p90_no:.3f}")
        for wkey, _, _ in TIME_WINDOWS:
            arr = np.abs(np.array(data[wkey]["lco_norm"], np.float32))
            if len(arr) < 10:
                continue
            p50, p70, p90 = np.percentile(arr, [50, 70, 90])
            print(f"    {wkey:<14}: p50={p50:.3f}  p70={p70:.3f}  p90={p90:.3f}  N={len(arr):,}")


if __name__ == "__main__":
    main()
