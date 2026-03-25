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
    python data/analyze_lco_threshold.py --dataset both --out lco_analysis
"""

from __future__ import annotations

import argparse
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

SAMPLE_HZ = 5.0          # 분석용 다운샘플 Hz
MAX_T_LC  = 5.0          # LC 예측 최대 horizon (s)
NO_LC_EXCLUDE_PRE = 5.0  # no_lc: 최근 LC로부터도 이 시간 이후 프레임만 포함 (선택 off 가능)

# (window_key, t_lo, t_hi)  — t_lo < t_to_lc ≤ t_hi
TIME_WINDOWS: List[Tuple[str, float, float]] = [
    ("-1~0",     0.0,  1.0),
    ("-2~-1",    1.0,  2.0),
    ("-3~-2",    2.0,  3.0),
    ("-4~-3",    3.0,  4.0),
    ("-5~-4",    4.0,  5.0),
    ("-4.5~-2.5",2.5,  4.5),
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


def find_recording_ids(raw_dir: Path, prefix_pattern: str = r"(\d+)_tracks\.csv$") -> List[str]:
    return sorted(
        re.match(prefix_pattern, p.name).group(1)
        for p in raw_dir.glob("*_tracks.csv")
        if re.match(prefix_pattern, p.name)
    )


def assign_window(t_to_lc: Optional[float]) -> Optional[str]:
    """t_to_lc: seconds until next LC (None if no LC within MAX_T_LC)."""
    if t_to_lc is None:
        return "no_lc"
    for key, lo, hi in TIME_WINDOWS:
        if lo < t_to_lc <= hi:
            return key
    return None  # outside all windows (e.g. t_to_lc > 5s)


def kde_safe(arr: np.ndarray, x_grid: np.ndarray, bw: float = 0.08) -> Optional[np.ndarray]:
    if len(arr) < 20:
        return None
    try:
        return gaussian_kde(arr, bw_method=bw)(x_grid)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# exiD data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_exid(raw_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns: {window_key: {"lco_norm": [...], "lat_v": [...]}}
    """
    rec_ids = find_recording_ids(raw_dir)
    if not rec_ids:
        raise FileNotFoundError(f"No exiD recordings in {raw_dir}")

    result: Dict[str, Dict[str, List[float]]] = {k: {"lco_norm": [], "lat_v": []} for k in ALL_WINDOW_KEYS}

    for rec_id in rec_ids:
        print(f"  [exiD] {rec_id} ...", end="", flush=True)
        rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
        tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv", low_memory=False)

        frame_rate = float(rec_meta.loc[0, "frameRate"]) if "frameRate" in rec_meta.columns else 25.0
        step = max(1, int(round(frame_rate / SAMPLE_HZ)))

        tracks = tracks.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)

        frame_arr = tracks["frame"].to_numpy(np.int32)
        lane_arr  = tracks["laneletId"].fillna(-1).astype(np.int32).to_numpy() \
                    if "laneletId" in tracks.columns \
                    else np.full(len(tracks), -1, np.int32)

        # latLaneCenterOffset (first semicolon-sep value)
        if "latLaneCenterOffset" in tracks.columns:
            _s = tracks["latLaneCenterOffset"].astype(str).str.strip().str.split(";").str[0]
            lco_arr = pd.to_numeric(_s, errors="coerce").fillna(0.0).to_numpy(np.float32)
        else:
            lco_arr = np.zeros(len(tracks), np.float32)

        # laneWidth (first semicolon-sep value)
        if "laneWidth" in tracks.columns:
            _s = tracks["laneWidth"].astype(str).str.strip().str.split(";").str[0]
            lw_arr = pd.to_numeric(_s, errors="coerce").fillna(3.5).to_numpy(np.float32)
        else:
            lw_arr = np.full(len(tracks), 3.5, np.float32)

        lat_v_arr = tracks["latVelocity"].to_numpy(np.float32) \
                    if "latVelocity" in tracks.columns \
                    else np.zeros(len(tracks), np.float32)

        # per-vehicle 인덱스 구축
        per_vid_rows: Dict[int, np.ndarray] = {}
        per_vid_f2r:  Dict[int, Dict[int, int]] = {}
        for v, idxs in tracks.groupby("trackId").indices.items():
            idxs = np.array(sorted(idxs, key=lambda i: frame_arr[i]), np.int32)
            per_vid_rows[int(v)] = idxs
            per_vid_f2r[int(v)]  = {int(frame_arr[r]): int(r) for r in idxs}

        # LC 이벤트 프레임 (laneletId 변화)
        lc_frames: Dict[int, np.ndarray] = {}
        for v, idxs in per_vid_rows.items():
            lids = lane_arr[idxs]
            chg = np.where(
                (lids[1:] != lids[:-1]) & (lids[1:] >= 0) & (lids[:-1] >= 0)
            )[0] + 1
            lc_frames[v] = frame_arr[idxs[chg]] if len(chg) else np.array([], np.int32)

        n_samples = 0
        for v, idxs in per_vid_rows.items():
            frs     = frame_arr[idxs]
            lc_fs   = np.sort(lc_frames.get(v, np.array([], np.int32)))
            f2r_v   = per_vid_f2r[v]

            for i, sf in enumerate(frs[::step]):
                r = f2r_v.get(sf)
                if r is None:
                    continue

                lco  = float(lco_arr[r])
                lw   = float(lw_arr[r])
                if lw < 0.5:
                    continue
                lco_norm = lco / (lw * 0.5)
                lat_v    = float(lat_v_arr[r])

                # 다음 LC까지 시간 계산
                future_lc = lc_fs[lc_fs > sf]
                if len(future_lc):
                    t_to_lc = (future_lc[0] - sf) / frame_rate
                    t_to_lc = t_to_lc if t_to_lc <= MAX_T_LC else None
                else:
                    t_to_lc = None

                wkey = assign_window(t_to_lc)
                if wkey is None:
                    continue

                # no_lc: 이전 LC로부터 NO_LC_EXCLUDE_PRE s 이후 프레임만 포함
                if wkey == "no_lc":
                    prev_lc = lc_fs[lc_fs <= sf]
                    if len(prev_lc):
                        if (sf - prev_lc[-1]) / frame_rate < NO_LC_EXCLUDE_PRE:
                            continue

                result[wkey]["lco_norm"].append(lco_norm)
                result[wkey]["lat_v"].append(lat_v)
                n_samples += 1

        print(f" {n_samples:,} samples")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# highD data collection
# ─────────────────────────────────────────────────────────────────────────────

def compute_highd_lco_width(
    y_center: np.ndarray,
    lane_id:  np.ndarray,
    dd:       np.ndarray,
    upper_mark: np.ndarray,
    lower_mark: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    preprocess.py 와 동일한 부호 규약:
      - lco 계산은 pre-flip 좌표
      - dd==1 차량: lco *= -1  (maybe_flip 후 일관성 유지)
      → 결과: positive lco = post-flip 기준 우측 이동 방향
    """
    n_upper = len(upper_mark)
    lco = np.zeros(len(y_center), np.float32)
    lw  = np.zeros(len(y_center), np.float32)

    # upper vehicles (dd==1): j = laneId - 2
    mask_up = (dd == 1)
    j_up    = lane_id.astype(np.int32) - 2
    ok_up   = mask_up & (j_up >= 0) & (j_up < n_upper - 1)
    if ok_up.any():
        j = j_up[ok_up]
        c = 0.5 * (upper_mark[j] + upper_mark[j + 1])
        lco[ok_up] = y_center[ok_up] - c
        lw[ok_up]  = np.abs(upper_mark[j + 1] - upper_mark[j])
    lco[mask_up] *= -1.0   # pre-flip → post-flip sign flip

    # lower vehicles (dd==2): j = laneId - n_upper - 2
    mask_lo = (dd == 2)
    j_lo    = lane_id.astype(np.int32) - n_upper - 2
    ok_lo   = mask_lo & (j_lo >= 0) & (j_lo < len(lower_mark) - 1)
    if ok_lo.any():
        j = j_lo[ok_lo]
        c = 0.5 * (lower_mark[j] + lower_mark[j + 1])
        lco[ok_lo] = y_center[ok_lo] - c
        lw[ok_lo]  = np.abs(lower_mark[j + 1] - lower_mark[j])

    return lco, lw


def collect_highd(raw_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    latVelocity 부호: +→우측 (post-flip 기준, dd==1 yVelocity 부호 반전 적용)
    """
    rec_ids = find_recording_ids(raw_dir)
    if not rec_ids:
        raise FileNotFoundError(f"No highD recordings in {raw_dir}")

    result: Dict[str, Dict[str, List[float]]] = {k: {"lco_norm": [], "lat_v": []} for k in ALL_WINDOW_KEYS}

    for rec_id in rec_ids:
        print(f"  [highD] {rec_id} ...", end="", flush=True)
        rec_meta  = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
        trk_meta  = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
        tracks    = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")

        frame_rate = float(rec_meta.loc[0, "frameRate"]) if "frameRate" in rec_meta.columns else 25.0
        step = max(1, int(round(frame_rate / SAMPLE_HZ)))

        upper_mark = np.array(parse_semicolon_floats(str(rec_meta.loc[0, "upperLaneMarkings"])), np.float32)
        lower_mark = np.array(parse_semicolon_floats(str(rec_meta.loc[0, "lowerLaneMarkings"])), np.float32)
        n_upper    = len(upper_mark)

        vid_to_dd = dict(zip(trk_meta["id"].astype(int), trk_meta["drivingDirection"].astype(int)))
        vid_to_w  = dict(zip(trk_meta["id"].astype(int), trk_meta["width"].astype(float)))
        vid_to_h  = dict(zip(trk_meta["id"].astype(int), trk_meta["height"].astype(float)))

        tracks = tracks.sort_values(["id", "frame"], kind="mergesort").reset_index(drop=True)

        frame_arr  = tracks["frame"].to_numpy(np.int32)
        vid_arr    = tracks["id"].astype(np.int32).to_numpy()
        x_arr      = tracks["x"].astype(np.float32).to_numpy().copy()
        y_arr      = tracks["y"].astype(np.float32).to_numpy().copy()
        yv_arr     = tracks["yVelocity"].astype(np.float32).to_numpy().copy()
        lane_arr   = tracks["laneId"].astype(np.int32).to_numpy()

        # bounding box corner → center
        w_row = np.array([vid_to_w.get(int(v), 0.0) for v in vid_arr], np.float32)
        h_row = np.array([vid_to_h.get(int(v), 0.0) for v in vid_arr], np.float32)
        y_arr += 0.5 * h_row
        x_arr += 0.5 * w_row

        dd_arr = np.array([vid_to_dd.get(int(v), 2) for v in vid_arr], np.int8)

        # lco, laneWidth 계산
        lco_arr, lw_arr = compute_highd_lco_width(y_arr, lane_arr, dd_arr, upper_mark, lower_mark)

        # latVelocity 부호: dd==1 → negate (post-flip 기준 +y=우측)
        lat_v_arr = yv_arr.copy()
        lat_v_arr[dd_arr == 1] *= -1.0

        # per-vehicle 인덱스
        per_vid_rows: Dict[int, np.ndarray] = {}
        per_vid_f2r:  Dict[int, Dict[int, int]] = {}
        for v, idxs in tracks.groupby("id").indices.items():
            idxs = np.array(sorted(idxs, key=lambda i: frame_arr[i]), np.int32)
            per_vid_rows[int(v)] = idxs
            per_vid_f2r[int(v)]  = {int(frame_arr[r]): int(r) for r in idxs}

        # LC 이벤트 (laneId 변화)
        lc_frames: Dict[int, np.ndarray] = {}
        for v, idxs in per_vid_rows.items():
            lids = lane_arr[idxs]
            chg = np.where(
                (lids[1:] != lids[:-1]) & (lids[1:] > 0) & (lids[:-1] > 0)
            )[0] + 1
            lc_frames[v] = frame_arr[idxs[chg]] if len(chg) else np.array([], np.int32)

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
                if len(future_lc):
                    t_to_lc = (future_lc[0] - sf) / frame_rate
                    t_to_lc = t_to_lc if t_to_lc <= MAX_T_LC else None
                else:
                    t_to_lc = None

                wkey = assign_window(t_to_lc)
                if wkey is None:
                    continue

                if wkey == "no_lc":
                    prev_lc = lc_fs[lc_fs <= sf]
                    if len(prev_lc):
                        if (sf - prev_lc[-1]) / frame_rate < NO_LC_EXCLUDE_PRE:
                            continue

                result[wkey]["lco_norm"].append(lco_norm)
                result[wkey]["lat_v"].append(lat_v)
                n_samples += 1

        print(f" {n_samples:,} samples")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Statistics summary
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(data: Dict[str, Dict[str, List[float]]], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  |lco_norm| 통계")
    print(f"{'='*60}")
    print(f"  {'window':<14} {'N':>7}  {'mean':>6}  {'std':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}  {'p95':>6}")
    print(f"  {'-'*14}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for wkey in ALL_WINDOW_KEYS:
        arr = np.abs(np.array(data[wkey]["lco_norm"], np.float32))
        if len(arr) == 0:
            continue
        print(f"  {wkey:<14}  {len(arr):>7,}  {arr.mean():>6.3f}  {arr.std():>6.3f}  "
              f"{np.percentile(arr,50):>6.3f}  {np.percentile(arr,75):>6.3f}  "
              f"{np.percentile(arr,90):>6.3f}  {np.percentile(arr,95):>6.3f}")

    print(f"\n  {label}  |latVelocity| 통계")
    print(f"  {'window':<14} {'N':>7}  {'mean':>6}  {'std':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}")
    print(f"  {'-'*14}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for wkey in ALL_WINDOW_KEYS:
        arr = np.abs(np.array(data[wkey]["lat_v"], np.float32))
        if len(arr) == 0:
            continue
        print(f"  {wkey:<14}  {len(arr):>7,}  {arr.mean():>6.3f}  {arr.std():>6.3f}  "
              f"{np.percentile(arr,50):>6.3f}  {np.percentile(arr,75):>6.3f}  "
              f"{np.percentile(arr,90):>6.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset(
    ax_abs:   plt.Axes,
    ax_signed: plt.Axes,
    ax_latv:  plt.Axes,
    data: Dict[str, Dict[str, List[float]]],
    title: str,
    no_lc_subsample: int = 5,   # no_lc 데이터 비율 축소 (KDE 가시성)
) -> None:
    # KDE x grids
    x_abs    = np.linspace(0.0, 1.5, 400)
    x_signed = np.linspace(-1.5, 1.5, 400)
    x_latv   = np.linspace(-2.0, 2.0, 400)

    for wkey in ALL_WINDOW_KEYS:
        lco_norm = np.array(data[wkey]["lco_norm"], np.float32)
        lat_v    = np.array(data[wkey]["lat_v"],    np.float32)
        if len(lco_norm) < 20:
            continue

        # no_lc는 서브샘플 (너무 많아서 KDE가 쪼그라드는 것 방지)
        if wkey == "no_lc" and no_lc_subsample > 1:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(lco_norm), max(20, len(lco_norm) // no_lc_subsample), replace=False)
            lco_norm = lco_norm[idx]
            lat_v    = lat_v[idx]

        color = COLORS[wkey]
        ls    = LINESTYLES[wkey]
        alpha = 0.85 if wkey != "no_lc" else 0.55
        lw    = 2.0  if wkey != "no_lc" else 1.5

        # |lco_norm|
        y = kde_safe(np.abs(lco_norm), x_abs)
        if y is not None:
            ax_abs.plot(x_abs, y, color=color, ls=ls, lw=lw, alpha=alpha, label=wkey)

        # signed lco_norm
        y = kde_safe(lco_norm, x_signed)
        if y is not None:
            ax_signed.plot(x_signed, y, color=color, ls=ls, lw=lw, alpha=alpha, label=wkey)

        # latVelocity
        y = kde_safe(lat_v, x_latv)
        if y is not None:
            ax_latv.plot(x_latv, y, color=color, ls=ls, lw=lw, alpha=alpha, label=wkey)

    ax_abs.set_title(f"{title}\n|lco_norm| (threshold X 후보)", fontsize=9)
    ax_abs.set_xlabel("|lco_norm|", fontsize=8)
    ax_abs.set_ylabel("density", fontsize=8)
    ax_abs.axvline(0.5, color="k", lw=0.8, ls=":", alpha=0.4)
    ax_abs.axvline(0.7, color="k", lw=0.8, ls=":", alpha=0.4)
    ax_abs.axvline(0.9, color="k", lw=0.8, ls=":", alpha=0.4)
    ax_abs.text(0.5, ax_abs.get_ylim()[1]*0.02, "0.5", fontsize=6, ha="center", color="gray")
    ax_abs.text(0.7, ax_abs.get_ylim()[1]*0.02, "0.7", fontsize=6, ha="center", color="gray")
    ax_abs.text(0.9, ax_abs.get_ylim()[1]*0.02, "0.9", fontsize=6, ha="center", color="gray")
    ax_abs.legend(fontsize=7)

    ax_signed.set_title(f"{title}\nlco_norm (signed)", fontsize=9)
    ax_signed.set_xlabel("lco_norm", fontsize=8)
    ax_signed.set_ylabel("density", fontsize=8)
    ax_signed.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax_signed.legend(fontsize=7)

    ax_latv.set_title(f"{title}\nlatVelocity (signed, m/s)", fontsize=9)
    ax_latv.set_xlabel("latVelocity  [+→right]", fontsize=8)
    ax_latv.set_ylabel("density", fontsize=8)
    ax_latv.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax_latv.legend(fontsize=7)


def add_vline_labels(ax: plt.Axes, xs: List[float]) -> None:
    """x 축에 참조선 추가 (threshold 후보 표시용)."""
    ylim = ax.get_ylim()
    for x in xs:
        ax.axvline(x, color="dimgray", lw=0.8, ls=":", alpha=0.5)
        ax.text(x, ylim[1] * 0.04, f"{x}", fontsize=6, ha="center", color="dimgray")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset",    default="both", choices=["exid", "highd", "both"])
    ap.add_argument("--exid_dir",   default="data/exiD/raw")
    ap.add_argument("--highd_dir",  default="data/highD/raw")
    ap.add_argument("--out",        default="lco_threshold_analysis",
                    help="출력 파일 prefix (확장자 제외)")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    datasets = {}
    if args.dataset in ("exid", "both"):
        print(f"\n[exiD] loading from {args.exid_dir}")
        datasets["exiD"] = collect_exid(Path(args.exid_dir))
        print_stats(datasets["exiD"], "exiD")

    if args.dataset in ("highd", "both"):
        print(f"\n[highD] loading from {args.highd_dir}")
        datasets["highD"] = collect_highd(Path(args.highd_dir))
        print_stats(datasets["highD"], "highD")

    # ── Figure ───────────────────────────────────────────────────────────────
    n_ds  = len(datasets)
    n_col = 3  # |lco_norm|, signed, latV
    fig, axes = plt.subplots(n_ds, n_col, figsize=(n_col * 5, n_ds * 4))
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    for row, (ds_name, data) in enumerate(datasets.items()):
        plot_dataset(
            ax_abs=axes[row, 0],
            ax_signed=axes[row, 1],
            ax_latv=axes[row, 2],
            data=data,
            title=ds_name,
        )
        # |lco_norm| 축 참조선 (plot 완료 후 y축 재조정)
        add_vline_labels(axes[row, 0], [0.5, 0.7, 0.9])

    fig.suptitle(
        "lco_norm & latVelocity 분포  (LC 이벤트 기준 시간 창별)\n"
        "v4 threshold X 결정용: |lco_norm| plot에서 LC 창과 no_lc 창의 분리가 뚜렷한 값을 X 후보로 선택",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = f"{args.out}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")

    # ── 후보 X 요약 출력 ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  threshold X 후보 요약  (|lco_norm| p90 기준)")
    print("="*60)
    for ds_name, data in datasets.items():
        no_lc_p90 = np.percentile(np.abs(data["no_lc"]["lco_norm"]), 90) \
                    if data["no_lc"]["lco_norm"] else float("nan")
        print(f"\n  [{ds_name}]  no_lc p90 = {no_lc_p90:.3f}")
        for wkey, _, hi in TIME_WINDOWS:
            arr = np.abs(np.array(data[wkey]["lco_norm"], np.float32))
            if len(arr) < 10:
                continue
            p50, p70, p90 = np.percentile(arr, [50, 70, 90])
            print(f"    {wkey:<14}: p50={p50:.3f}  p70={p70:.3f}  p90={p90:.3f}  N={len(arr):,}")


if __name__ == "__main__":
    main()
