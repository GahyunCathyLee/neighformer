#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_label.py  —  Window-level scenario labeling for NGSIM tracks.

NGSIM vs highD differences
───────────────────────────
  - Single combined CSV (NGSIM_data.csv) instead of per-recording CSV files
  - Fixed frame rate: 10 Hz (no recordingMeta)
  - Recording IDs: 0 = "i-80",  1 = "us-101"  (sorted alphabetically,
    matching the order in data/NGSIM/preprocess.py)
  - Lane_ID 1 = leftmost (fastest lane);  increasing Lane_ID → moving right
  - LC direction inferred from Lane_ID transition (not yVelocity)
  - Adjacent-lane neighbor IDs computed geometrically per frame
  - Freeway locations only (lankershim / peachtree excluded)

Event labels (3-class)
──────────────────────
  cut_in          : lane change + rear/alongside vehicle on target side
  lane_change     : lane change + no such vehicle (clear gap)
  lane_following  : no lane change in the window

State labels (2-class)
──────────────────────
  dense     : occupancy <= 0.40
  free_flow : occupancy >  0.40
  (computed from mmap nb_mask, STATE_SLOTS = [0, 2, 3, 5, 6])

Usage (mmap mode, default):
    python scenario_label.py \\
        --data_dir   data/NGSIM \\
        --ngsim_file NGSIM_data.csv \\
        --mmap_dir   c0 \\
        --out_csv    scenario_labels.csv \\
        --history_sec 2 --future_sec 5 --target_hz 3

Usage (standalone, enumerates windows by stride):
    python scenario_label.py \\
        --data_dir   data/NGSIM \\
        --ngsim_file NGSIM_data.csv \\
        --mmap_dir   "" \\
        --out_csv    scenario_labels.csv
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
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NGSIM_FPS = 10.0
FT2M      = 0.3048

# Recording ID → location name  (preprocess.py: enumerate(sorted(loc_dfs.items())))
LOCATION_MAP: Dict[int, str] = {0: "i-80", 1: "us-101"}

# Max through-lane per location (from preprocess.py)
MAX_LANES: Dict[str, int] = {"i-80": 6, "us-101": 5}

# State-label constants (identical to highD)
STATE_SLOTS     = [0, 2, 3, 5, 6]
STATE_THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# State label
# ─────────────────────────────────────────────────────────────────────────────

def compute_state_labels(
    nb_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    nb_mask : (N, T, K) bool array

    Returns
    -------
    state_labels : (N,) str array  — 'dense' or 'free_flow'
    occ          : (N,) float32
    """
    occ    = nb_mask[:, :, STATE_SLOTS].astype(np.float32).mean(axis=(1, 2))
    labels = np.where(occ <= STATE_THRESHOLD, "dense", "free_flow")
    return labels, occ


# ─────────────────────────────────────────────────────────────────────────────
# Raw data loading
# ─────────────────────────────────────────────────────────────────────────────

def smart_read_csv(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-delimited CSV."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)


def find_recording_ids(raw_dir: Path) -> List[str]:
    """Return sorted XX strings where XX_tracks.csv exists."""
    ids = []
    for p in raw_dir.glob("*_tracks.csv"):
        m = re.match(r"^(\d+)_tracks\.csv$", p.name)
        if m:
            ids.append(m.group(1))
    return sorted(set(ids))


def load_highd_style_recording(raw_dir: Path, xx: str) -> pd.DataFrame:
    """Load one converted highD-style NGSIM recording."""
    path = raw_dir / f"{xx}_tracks.csv"
    tracks = smart_read_csv(path)
    tracks.columns = [c.strip() for c in tracks.columns]

    required = {"frame", "id", "laneId"}
    missing = required - set(tracks.columns)
    if missing:
        raise KeyError(f"{path} missing required columns: {sorted(missing)}")

    df = tracks.copy()
    rec_id = int(xx)
    df["recordingId"] = rec_id
    df["trackId"] = df["id"].astype(np.int32)
    df["frame"] = df["frame"].astype(np.int32)
    df["laneId"] = df["laneId"].astype(np.int16)

    if "v_length" not in df.columns:
        df["v_length"] = pd.to_numeric(df.get("width", 0.0), errors="coerce").fillna(0.0)
    if "xc" not in df.columns:
        df["xc"] = (
            pd.to_numeric(df.get("x", 0.0), errors="coerce").fillna(0.0)
            + 0.5 * pd.to_numeric(df.get("width", 0.0), errors="coerce").fillna(0.0)
        )
    if "yc" not in df.columns:
        df["yc"] = (
            pd.to_numeric(df.get("y", 0.0), errors="coerce").fillna(0.0)
            + 0.5 * pd.to_numeric(df.get("height", 0.0), errors="coerce").fillna(0.0)
        )

    rename = {
        "leftFollowingId": "leftRearId",
        "rightFollowingId": "rightRearId",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    for col in ["leftRearId", "leftAlongsideId", "rightRearId", "rightAlongsideId"]:
        if col not in df.columns:
            df[col] = 0

    return df.sort_values(["trackId", "frame"]).reset_index(drop=True)


def load_highd_style_raw(raw_dir: Path) -> Dict[int, pd.DataFrame]:
    """
    Load highD-style raw CSVs produced by convert_to_highd.py.

    The converted files already contain per-segment recording ids, split track
    ids, lane ids, and highD-style adjacent-neighbor columns. This is the
    preferred input when labels must match highD-preprocessor mmap keys.
    """
    rec_ids = find_recording_ids(raw_dir)
    if not rec_ids:
        raise FileNotFoundError(f"No *_tracks.csv files found in {raw_dir}")

    result: Dict[int, pd.DataFrame] = {}
    for xx in rec_ids:
        rec_id = int(xx)
        df = load_highd_style_recording(raw_dir, xx)
        print(f"  [{xx}] rec_id={rec_id}  {len(df):,} rows  "
              f"{df['trackId'].nunique():,} tracks")
        result[rec_id] = df

    return result


def load_ngsim_freeway(ngsim_path: Path) -> Dict[int, pd.DataFrame]:
    """
    Load NGSIM_data.csv, keep freeway locations only.

    Returns
    -------
    {rec_id: DataFrame}
      rec_id 0 → "i-80",  rec_id 1 → "us-101"

    Columns in output DataFrame (all units already converted to meters):
      trackId, frame, laneId, xc (longitudinal center),
      yc (lateral center), v_length, leftRearId, leftAlongsideId,
      rightRearId, rightAlongsideId  (added by compute_adjacent_neighbors)
    """
    FREEWAY_LOCS = {"i-80", "us-101"}
    needed = [
        "Vehicle_ID", "Frame_ID",
        "Local_X", "Local_Y", "v_length", "v_Width",
        "v_Class", "v_Vel", "v_Acc", "Lane_ID", "Location",
    ]

    print(f"Reading {ngsim_path} …")
    df = pd.read_csv(ngsim_path, usecols=needed, low_memory=False)
    df = df[df["Location"].isin(FREEWAY_LOCS)].copy()
    print(f"  Freeway rows: {len(df):,}")

    result: Dict[int, pd.DataFrame] = {}
    for rec_id, loc in sorted(LOCATION_MAP.items()):
        sub = df[df["Location"] == loc].copy()

        # ── basic filters ──────────────────────────────────────────────────
        before = len(sub)
        sub = sub.drop_duplicates(subset=["Vehicle_ID", "Frame_ID"])
        if (dropped := before - len(sub)):
            print(f"  [{loc}] removed {dropped:,} duplicate rows")

        max_lane = MAX_LANES[loc]
        sub = sub[(sub["Lane_ID"] >= 1) & (sub["Lane_ID"] <= max_lane)].copy()
        sub = sub[sub["v_Class"] != 1].copy()   # no motorcycles

        # ── feet → meters ──────────────────────────────────────────────────
        for col in ["Local_X", "Local_Y", "v_length", "v_Width", "v_Vel", "v_Acc"]:
            sub[col] = sub[col] * FT2M

        # ── center positions ───────────────────────────────────────────────
        # Local_Y = front-center (longitudinal); xc = center = front - length/2
        sub["xc"] = sub["Local_Y"] - 0.5 * sub["v_length"]
        # Local_X ≈ lateral center
        sub["yc"] = sub["Local_X"]

        # ── normalized column names ────────────────────────────────────────
        sub["trackId"]     = sub["Vehicle_ID"].astype(np.int32)
        sub["frame"]       = sub["Frame_ID"].astype(np.int32)
        sub["laneId"]      = sub["Lane_ID"].astype(np.int16)
        sub["recordingId"] = rec_id

        sub = sub.sort_values(["trackId", "frame"]).reset_index(drop=True)
        print(f"  [{loc}] rec_id={rec_id}  {len(sub):,} rows  "
              f"{sub['trackId'].nunique():,} vehicles after filters")
        result[rec_id] = sub

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Adjacent-lane neighbor computation
# ─────────────────────────────────────────────────────────────────────────────

def _adj_neighbors_one_frame(
    vids: np.ndarray,
    lids: np.ndarray,
    xc:   np.ndarray,
    hl:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 4 adjacent-lane neighbor IDs for every vehicle in one frame.

    Slot definitions (NGSIM: Lane_ID 1=leftmost, increasing → rightward):
      leftRearId      : vehicle in (lane-1) behind  ego  (highest xc < ego_xc)
      leftAlongsideId : vehicle in (lane-1) alongside ego (longitudinal overlap)
      rightRearId     : vehicle in (lane+1) behind  ego
      rightAlongsideId: vehicle in (lane+1) alongside ego

    Returns four (N,) int32 arrays.
    """
    N = len(vids)
    left_rear   = np.zeros(N, np.int32)
    left_along  = np.zeros(N, np.int32)
    right_rear  = np.zeros(N, np.int32)
    right_along = np.zeros(N, np.int32)

    for i in range(N):
        xi  = xc[i]
        hli = hl[i]
        xf  = xi + hli
        xr  = xi - hli
        lid = int(lids[i])

        for tlane, r_out, a_out in (
            (lid - 1, left_rear,  left_along),
            (lid + 1, right_rear, right_along),
        ):
            mask = lids == tlane
            if not np.any(mask):
                continue
            idx_n = np.where(mask)[0]
            xc_n  = xc[idx_n]
            hl_n  = hl[idx_n]

            # Alongside: longitudinal overlap (same logic as preprocess.py)
            xf_n = xc_n + hl_n
            xr_n = xc_n - hl_n
            ov   = (xr_n < xf) & (xf_n > xr)
            if np.any(ov):
                c = idx_n[ov]
                a_out[i] = vids[c[np.argmin(np.abs(xc_n[ov] - xi))]]

            # Rear: behind ego — closest vehicle (highest xc that is < ego_xc)
            behind = xc_n < xi
            if np.any(behind):
                c = idx_n[behind]
                r_out[i] = vids[c[np.argmax(xc_n[behind])]]

    return left_rear, left_along, right_rear, right_along


def compute_adjacent_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leftRearId, leftAlongsideId, rightRearId, rightAlongsideId columns.
    Processes frame-by-frame (same approach as preprocess.py compute_all_neighbors).
    """
    df = df.reset_index(drop=True)
    N  = len(df)

    lr_all  = np.zeros(N, np.int32)
    la_all  = np.zeros(N, np.int32)
    rr_all  = np.zeros(N, np.int32)
    ra_all  = np.zeros(N, np.int32)

    vids = df["trackId"].to_numpy(np.int32)
    fids = df["frame"].to_numpy(np.int32)
    lids = df["laneId"].to_numpy(np.int32)
    xc   = df["xc"].to_numpy(np.float64)
    hl   = (df["v_length"] * 0.5).to_numpy(np.float64)

    for fr in tqdm(np.unique(fids), desc="  adj neighbors", leave=False):
        rows = np.where(fids == fr)[0]
        if len(rows) < 2:
            continue
        lr, la, rr, ra = _adj_neighbors_one_frame(
            vids[rows], lids[rows], xc[rows], hl[rows]
        )
        lr_all[rows] = lr
        la_all[rows] = la
        rr_all[rows] = rr
        ra_all[rows] = ra

    df["leftRearId"]       = lr_all
    df["leftAlongsideId"]  = la_all
    df["rightRearId"]      = rr_all
    df["rightAlongsideId"] = ra_all
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Lane lookup
# ─────────────────────────────────────────────────────────────────────────────

def build_lane_lookup(df: pd.DataFrame) -> Dict[int, Dict[int, int]]:
    """Build lookup[trackId][frame] = laneId."""
    lookup: Dict[int, Dict[int, int]] = {}
    for tid, g in df.groupby("trackId", sort=False):
        d: Dict[int, int] = {}
        for f, l in zip(g["frame"].to_numpy(), g["laneId"].to_numpy()):
            try:
                d[int(f)] = int(l)
            except (ValueError, TypeError):
                pass
        lookup[int(tid)] = d
    return lookup


def get_lane_at(
    lookup: Dict[int, Dict[int, int]],
    track_id: int,
    frame: int,
) -> Optional[int]:
    d = lookup.get(int(track_id))
    if d is None:
        return None
    return d.get(int(frame))


def is_adjacent_lane(ego_lane: int, nb_lane: int) -> bool:
    """Adjacent iff Lane_ID differs by exactly 1 (same in NGSIM and highD)."""
    return abs(int(ego_lane) - int(nb_lane)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Lane change detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_lane_change(
    w: pd.DataFrame,
) -> Tuple[bool, int, Optional[int]]:
    """
    Detect lane changes in window w via laneId transitions.

    Returns
    -------
    has_lc   : bool
    lc_count : int
    lc_frame : Optional[int]  (frame of first transition)
    """
    if "laneId" not in w.columns or "frame" not in w.columns:
        return False, 0, None

    df = w[["frame", "laneId"]].copy()
    df["frame"]  = pd.to_numeric(df["frame"],  errors="coerce").astype("Int64")
    df["laneId"] = pd.to_numeric(df["laneId"], errors="coerce")
    df = df.dropna().sort_values("frame")

    if len(df) < 2:
        return False, 0, None

    lane   = df["laneId"].to_numpy()
    frames = df["frame"].to_numpy()

    changed = lane[1:] != lane[:-1]
    count   = int(changed.sum())
    if count == 0:
        return False, 0, None

    first_frame = int(frames[1:][changed][0])
    return True, count, first_frame


# ─────────────────────────────────────────────────────────────────────────────
# LC direction inference  (NGSIM-specific: use laneId transition)
# ─────────────────────────────────────────────────────────────────────────────

def infer_lc_direction(
    w: pd.DataFrame,
    lc_frame: int,
    K: int = 5,   # unused, kept for interface compatibility
) -> Optional[str]:
    """
    NGSIM: infer LC direction from Lane_ID transition around lc_frame.

    Lane_ID 1 = leftmost (fast);  increasing Lane_ID = moving right.

    Returns "left", "right", or None (if indeterminate).
    """
    df = w[["frame", "laneId"]].copy()
    df["frame"]  = pd.to_numeric(df["frame"],  errors="coerce").astype("Int64")
    df["laneId"] = pd.to_numeric(df["laneId"], errors="coerce")
    df = df.dropna().sort_values("frame")

    pre  = df[df["frame"] <  lc_frame]["laneId"]
    post = df[df["frame"] >= lc_frame]["laneId"]

    if pre.empty or post.empty:
        return None

    old_lane = int(pre.iloc[-1])
    new_lane = int(post.iloc[0])

    if new_lane > old_lane:
        return "right"
    if new_lane < old_lane:
        return "left"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Adjacent presence check (cut-in detection)
# ─────────────────────────────────────────────────────────────────────────────

def _to_int_id(x) -> Optional[int]:
    try:
        v = int(float(x))
        return None if v in (-1, 0) else v
    except Exception:
        return None


def check_adjacent_rear_or_alongside(
    w: pd.DataFrame,
    lc_frame: int,
    direction: str,
    lookup: Dict[int, Dict[int, int]],
    W: int = 10,
) -> bool:
    """
    Scan pre-LC window [lc_frame - W, lc_frame - 1].
    Return True if a rear or alongside vehicle in the target adjacent lane
    is confirmed present.

    direction : "left" | "right"
    W         : look-back in native frames (10 Hz); default=10 → 1 s
    """
    if direction == "left":
        rear_col      = "leftRearId"
        alongside_col = "leftAlongsideId"
    else:
        rear_col      = "rightRearId"
        alongside_col = "rightAlongsideId"

    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    pre = df[(df["frame"] >= lc_frame - W) & (df["frame"] < lc_frame)]
    if len(pre) == 0:
        return False

    for _, row in pre.iterrows():
        f        = int(row["frame"])
        ego_lane = row.get("laneId", np.nan)
        if pd.isna(ego_lane):
            continue
        ego_lane = int(ego_lane)

        for col in (rear_col, alongside_col):
            if col not in pre.columns:
                continue
            nb_id = _to_int_id(row.get(col, 0))
            if nb_id is None:
                continue
            nb_lane = get_lane_at(lookup, nb_id, f)
            if nb_lane is None:
                continue
            if is_adjacent_lane(ego_lane, nb_lane):
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Window-level labeling
# ─────────────────────────────────────────────────────────────────────────────

def label_window(
    w: pd.DataFrame,
    lookup: Dict[int, Dict[int, int]],
    W_adj: int = 10,
) -> Dict:
    """
    Label one sample window.

    Event labels
    ────────────
    cut_in          : LC + rear/alongside in adjacent lane on target side
    lane_change     : LC + no such vehicle
    lane_following  : no LC

    Returns dict with keys:
      event_label, lc_direction, lc_frame, lc_count, has_adj_rear_or_alongside
    """
    out: Dict = {}

    has_lc, lc_count, lc_frame = detect_lane_change(w)
    out["lc_count"] = lc_count
    out["lc_frame"] = int(lc_frame) if lc_frame is not None else -1

    if not has_lc or lc_frame is None:
        out["event_label"]               = "lane_following"
        out["lc_direction"]              = "none"
        out["has_adj_rear_or_alongside"] = False
        return out

    direction = infer_lc_direction(w, lc_frame=lc_frame)
    out["lc_direction"] = direction if direction is not None else "unknown"

    if direction is None:
        out["event_label"]               = "lane_change"
        out["has_adj_rear_or_alongside"] = False
        return out

    has_adj = check_adjacent_rear_or_alongside(
        w=w, lc_frame=lc_frame, direction=direction, lookup=lookup, W=W_adj,
    )
    out["has_adj_rear_or_alongside"] = bool(has_adj)
    out["event_label"] = "cut_in" if has_adj else "lane_change"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-location processing
# ─────────────────────────────────────────────────────────────────────────────

def label_location(
    rec_id:      int,
    df:          pd.DataFrame,
    keys:        Optional[List[Tuple[int, int]]],
    history_sec: float,
    future_sec:  float,
    target_hz:   float,
    stride_sec:  float = 1.0,
    W_adj:       int   = 10,
) -> List[Dict]:
    """
    Label all requested windows for one NGSIM location.

    Parameters
    ----------
    rec_id  : recording id (0 = i-80, 1 = us-101)
    df      : pre-processed dataframe with adjacent neighbor columns
    keys    : list of (trackId, t0_frame) pairs from mmap (None = standalone)
    """
    fr         = NGSIM_FPS
    step       = max(1, int(round(fr / target_hz)))
    T          = int(round(history_sec * target_hz))
    Tf         = int(round(future_sec  * target_hz))
    win_native = (T + Tf - 1) * step

    lookup = build_lane_lookup(df)
    by_tid = {int(tid): g for tid, g in df.groupby("trackId", sort=False)}

    rows: List[Dict] = []

    if keys is not None:
        # ── mmap mode: label exactly the requested (tid, t0) keys ─────────
        for tid, t0_frame in sorted(set(keys)):
            g    = by_tid.get(int(tid))
            base = {
                "recordingId": rec_id,
                "trackId":     int(tid),
                "t0_frame":    int(t0_frame),
                "frameRate":   fr,
                "ds_step":     step,
                "history_sec": history_sec,
                "future_sec":  future_sec,
                "target_hz":   target_hz,
            }

            if g is None:
                rows.append({**base, "event_label": "unknown", "lc_frame": -1,
                              "lc_count": 0, "lc_direction": "unknown",
                              "has_adj_rear_or_alongside": False})
                continue

            t1_frame = int(t0_frame) + win_native
            w = g[(g["frame"] >= t0_frame) & (g["frame"] <= t1_frame)]

            if len(w) == 0:
                rows.append({**base, "event_label": "unknown", "lc_frame": -1,
                              "lc_count": 0, "lc_direction": "unknown",
                              "has_adj_rear_or_alongside": False})
                continue

            result = label_window(w, lookup=lookup, W_adj=W_adj)
            rows.append({**base, **result})

    else:
        # ── standalone mode: enumerate windows by stride ───────────────────
        stride_native = max(1, int(round(stride_sec * fr)))

        for tid, g in tqdm(by_tid.items(), desc="  tracks", leave=False):
            frames = g["frame"].dropna().astype(int).to_numpy()
            if len(frames) == 0:
                continue
            f0, f1 = int(frames.min()), int(frames.max())
            t0 = f0
            while t0 + win_native <= f1:
                t1 = t0 + win_native
                w  = g[(g["frame"] >= t0) & (g["frame"] <= t1)]
                if len(w) > 0:
                    result = label_window(w, lookup=lookup, W_adj=W_adj)
                    rows.append({
                        "recordingId": rec_id,
                        "trackId":     int(tid),
                        "t0_frame":    int(t0),
                        "t1_frame":    int(t1),
                        "frameRate":   fr,
                        "ds_step":     step,
                        "history_sec": history_sec,
                        "future_sec":  future_sec,
                        "target_hz":   target_hz,
                        **result,
                    })
                t0 += stride_native

    return rows


def _process_highd_style_recording(
    item: Tuple[int, Optional[List[Tuple[int, int, int]]]],
    raw_dir: Path,
    history_sec: float,
    future_sec: float,
    target_hz: float,
    stride_sec: float,
    W_adj: int,
) -> Tuple[int, List[Dict], Optional[List[Tuple[int, int, int]]]]:
    """Worker for converted highD-style raw recordings."""
    rec_id, raw_keys = item
    xx = f"{rec_id:02d}"
    df = load_highd_style_recording(raw_dir, xx)
    keys_for_rec = [(tid, t0) for tid, t0, _ in raw_keys] if raw_keys is not None else None
    rows = label_location(
        rec_id=rec_id,
        df=df,
        keys=keys_for_rec,
        history_sec=history_sec,
        future_sec=future_sec,
        target_hz=target_hz,
        stride_sec=stride_sec,
        W_adj=W_adj,
    )
    return rec_id, rows, raw_keys


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="NGSIM window-level scenario labeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir",    default="data/NGSIM",      help="Base data directory")
    ap.add_argument("--raw_dir",     default="raw",             help="highD-style raw CSV subdir under data_dir")
    ap.add_argument("--ngsim_file",  default="NGSIM_data.csv",  help="Combined NGSIM CSV filename")
    ap.add_argument("--mmap_dir",    default="c0",              help="Mmap subdir (empty = standalone mode)")
    ap.add_argument("--out_csv",     default="scenario_labels.csv",
                    help="Output filename (saved inside mmap_dir if set, else data_dir)")
    ap.add_argument("--use_combined_raw", action="store_true",
                    help="Use legacy combined NGSIM_data.csv loader instead of highD-style raw files")

    ap.add_argument("--history_sec", type=float, default=2.0)
    ap.add_argument("--future_sec",  type=float, default=5.0)
    ap.add_argument("--target_hz",   type=float, default=3.0)
    ap.add_argument("--stride_sec",  type=float, default=1.0, help="Stride (standalone mode only)")
    ap.add_argument("--W_adj",       type=int,   default=10,
                    help="Pre-LC look-back window (native 10-Hz frames). 10 = 1 s")
    ap.add_argument("--num_workers", type=int,   default=0,
                    help="Worker processes for highD-style raw mode (0 = os.cpu_count())")
    return ap.parse_args()


def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir)
    raw_dir  = data_dir / args.raw_dir
    mmap_dir = data_dir / args.mmap_dir if args.mmap_dir else None

    out_dir  = mmap_dir if mmap_dir else data_dir
    out_csv  = out_dir / args.out_csv
    out_dir.mkdir(parents=True, exist_ok=True)

    ngsim_path = data_dir / args.ngsim_file

    # ── Determine mode ────────────────────────────────────────────────────────
    keys_by_rec: Optional[Dict[int, List[Tuple[int, int, int]]]] = None
    state_labels = None
    state_occ    = None

    if mmap_dir and mmap_dir.exists():
        rec_path = mmap_dir / "meta_recordingId.npy"
        trk_path = mmap_dir / "meta_trackId.npy"
        t0_path  = mmap_dir / "meta_frame.npy"

        if not (rec_path.exists() and trk_path.exists() and t0_path.exists()):
            raise FileNotFoundError(
                f"mmap meta files not found in {mmap_dir}.\n"
                f"Expected: meta_recordingId.npy, meta_trackId.npy, meta_frame.npy"
            )

        rids = np.load(rec_path)
        tids = np.load(trk_path)
        t0s  = np.load(t0_path)
        assert len(rids) == len(tids) == len(t0s), "mmap meta length mismatch"

        # State labels from nb_mask
        nb_mask_path = mmap_dir / "nb_mask.npy"
        if nb_mask_path.exists():
            nb_mask_arr             = np.load(nb_mask_path, mmap_mode="r")
            state_labels, state_occ = compute_state_labels(nb_mask_arr)
            print(f"  state dense={(state_labels=='dense').mean()*100:.1f}%  "
                  f"free_flow={(state_labels=='free_flow').mean()*100:.1f}%")
        else:
            print("  [WARN] nb_mask.npy not found — state_label will be 'unknown'")

        keys_by_rec = {}
        for i, (rid, tid, t0) in enumerate(zip(rids, tids, t0s)):
            keys_by_rec.setdefault(int(rid), []).append((int(tid), int(t0), i))

        n_total = len(rids)
        print(f"[Mode] mmap  |  {n_total:,} samples  |  "
              f"{len(keys_by_rec)} locations  "
              f"(rec IDs: {sorted(keys_by_rec.keys())})")
    else:
        n_total = None
        print(f"[Mode] standalone  |  stride={args.stride_sec}s")

    highd_style_mode = (
        not args.use_combined_raw
        and raw_dir.exists()
        and bool(find_recording_ids(raw_dir))
    )

    # ── Label each recording/location ─────────────────────────────────────────
    all_rows: List[Dict] = []

    if highd_style_mode:
        rec_ids = [int(xx) for xx in find_recording_ids(raw_dir)]
        if keys_by_rec is not None:
            items = [(rec_id, keys_by_rec.get(rec_id, [])) for rec_id in rec_ids]
        else:
            items = [(rec_id, None) for rec_id in rec_ids]

        n_workers = args.num_workers if args.num_workers > 0 else os.cpu_count()
        print(f"[Process] highD-style raw from {raw_dir}  |  "
              f"{len(items)} recordings  |  workers={n_workers}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
            futs = {
                exe.submit(
                    _process_highd_style_recording,
                    item,
                    raw_dir,
                    args.history_sec,
                    args.future_sec,
                    args.target_hz,
                    args.stride_sec,
                    args.W_adj,
                ): item[0]
                for item in items
            }
            for fut in tqdm(concurrent.futures.as_completed(futs),
                            total=len(futs), desc="Labeling recordings"):
                rec_id, rows, raw_keys = fut.result()
                idx_map = {
                    (tid, t0): mi
                    for tid, t0, mi in raw_keys
                } if raw_keys is not None else {}

                for row in rows:
                    if state_labels is not None:
                        mi = idx_map.get((row["trackId"], row["t0_frame"]))
                        if mi is not None:
                            row["state_label"] = state_labels[mi]
                            row["occupancy"] = float(state_occ[mi])
                            continue
                    row["state_label"] = "unknown"
                    row["occupancy"] = float("nan")

                all_rows.extend(rows)
                print(f"  [{rec_id:02d}] → {len(rows):,} windows labeled")

    else:
        loc_dfs = load_ngsim_freeway(ngsim_path)
        for rec_id in loc_dfs:
            loc_name = LOCATION_MAP.get(rec_id, f"{rec_id:02d}")
            print(f"[{loc_name}] computing adjacent neighbors …")
            loc_dfs[rec_id] = compute_adjacent_neighbors(loc_dfs[rec_id])

        mmap_idx_by_key: Dict[Tuple[int,int,int], int] = {}
        for rec_id, df in loc_dfs.items():
            loc_name = LOCATION_MAP.get(rec_id, f"{rec_id:02d}")

            if keys_by_rec is not None:
                raw_keys = keys_by_rec.get(rec_id, [])
                keys_for_loc = [(tid, t0) for tid, t0, _ in raw_keys]
                for tid, t0, mi in raw_keys:
                    mmap_idx_by_key[(rec_id, tid, t0)] = mi
            else:
                keys_for_loc = None

            print(f"[{loc_name}] labeling windows …")
            rows = label_location(
                rec_id      = rec_id,
                df          = df,
                keys        = keys_for_loc,
                history_sec = args.history_sec,
                future_sec  = args.future_sec,
                target_hz   = args.target_hz,
                stride_sec  = args.stride_sec,
                W_adj       = args.W_adj,
            )

            for row in rows:
                if state_labels is not None:
                    mi = mmap_idx_by_key.get((rec_id, row["trackId"], row["t0_frame"]))
                    if mi is not None:
                        row["state_label"] = state_labels[mi]
                        row["occupancy"]   = float(state_occ[mi])
                        continue
                row["state_label"] = "unknown"
                row["occupancy"]   = float("nan")

            all_rows.extend(rows)
            print(f"  → {len(rows):,} windows labeled")

    # ── Save ─────────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    if not df_out.empty:
        df_out = df_out.sort_values(["recordingId", "trackId", "t0_frame"])
    df_out.to_csv(out_csv, index=False)
    print(f"\n[DONE] {len(df_out):,} labels → {out_csv}")

    if n_total is not None and len(df_out) != n_total:
        print(f"[WARN] mmap samples={n_total}, labeled={len(df_out)} "
              f"(delta={n_total - len(df_out)})")

    if "event_label" in df_out.columns:
        print("\nEvent label counts:")
        print(df_out["event_label"].value_counts().to_string())

    if "state_label" in df_out.columns:
        print("\nState label counts:")
        print(df_out["state_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
