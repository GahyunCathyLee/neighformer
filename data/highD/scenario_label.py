#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_label.py  —  Window-level scenario labeling for highD tracks.

Event labels (2-class):
    lane_change     : lane change in the window
    lane_following  : no lane change in the window

State labels (2-class):
    dense     : occupancy >  0.40
    free_flow : occupancy <= 0.40

    occupancy   = selected-neighbor occupancy over history frames
    STATE_SLOTS = [0, 2, 3, 5, 6]  (front + adjacent lead/alongside, no rear slots)
    Threshold 0.40 derived from KDE valley of the occupancy distribution.
    Prefer raw neighbor-id columns so the label is not distorted by top-N gating.

Usage (standalone):
    python scenario_label.py \\
        --data_dir  data/highD \\
        --raw_dir   raw \\
        --out_csv   window_labels.csv \\
        --history_sec 2 --future_sec 5 --target_hz 3

Usage (from mmap metadata):
    python scenario_label.py \\
        --data_dir  data/highD \\
        --raw_dir   raw \\
        --mmap_dir  mmap \\
        --out_csv   window_labels.csv \\
        --history_sec 2 --future_sec 5 --target_hz 3
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
# State label constants
# ─────────────────────────────────────────────────────────────────────────────

STATE_SLOTS     = [0, 2, 3, 5, 6]   # front + adj lead/alongside (no rear)
STATE_RAW_COLS  = ["leadId", "leftLeadId", "leftAlongsideId", "rightLeadId", "rightAlongsideId"]
STATE_THRESHOLD = 0.40               # occupancy > 0.40 → dense, else free_flow


def labels_from_occupancy(occ: np.ndarray) -> np.ndarray:
    """Map neighbor occupancy to traffic state labels."""
    return np.where(occ > STATE_THRESHOLD, "dense", "free_flow")


def compute_state_labels(nb_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute state label for each sample from mmap nb_mask.

    Parameters
    ----------
    nb_mask : (N, T, K) bool array

    Returns
    -------
    state_labels : (N,) str array  —  'dense' or 'free_flow'
    occ          : (N,) float32    —  raw occupancy value
    """
    occ    = nb_mask[:, :, STATE_SLOTS].astype(np.float32).mean(axis=(1, 2))
    labels = labels_from_occupancy(occ)
    return labels, occ


def compute_raw_state_label(history: pd.DataFrame) -> Tuple[str, float]:
    """
    Compute state label from raw highD neighbor-id columns over history frames.

    This intentionally uses raw neighbor presence instead of mmap nb_mask because
    nb_mask may already have top-N gating applied for model input.
    """
    if history.empty:
        return "unknown", float("nan")

    cols = [c for c in STATE_RAW_COLS if c in history.columns]
    if len(cols) != len(STATE_RAW_COLS):
        return "unknown", float("nan")

    vals = history[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    occ = float((vals > 0).mean())
    return str(labels_from_occupancy(np.array([occ]))[0]), occ


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def smart_read_csv(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-delimited CSV."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)


def find_recording_ids(raw_dir: Path) -> List[str]:
    """Return sorted list of XX strings where XX_tracks.csv exists."""
    ids = []
    for p in raw_dir.glob("*_tracks.csv"):
        m = re.match(r"^(\d+)_tracks\.csv$", p.name)
        if m:
            ids.append(m.group(1))
    return sorted(set(ids))


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_tracks(tracks: pd.DataFrame, xx: str) -> pd.DataFrame:
    """
    Normalize highD tracks to the common schema used by labeling logic.

    Input columns (highD format):
        frame, id, x, y, width, height,
        xVelocity, yVelocity, xAcceleration, yAcceleration,
        dhw, thw, ttc,
        precedingId, followingId,
        leftPrecedingId, leftAlongsideId, leftFollowingId,
        rightPrecedingId, rightAlongsideId, rightFollowingId,
        laneId

    Output adds / renames:
        recordingId  (derived from filename XX)
        trackId      (= id)
        latVelocity  (= yVelocity)
        leadId       (= precedingId)
        rearId       (= followingId)
        leftLeadId, leftRearId, leftAlongsideId
        rightLeadId, rightRearId, rightAlongsideId
    """
    df = tracks.copy()
    df.columns = [c.strip() for c in df.columns]

    df["recordingId"] = int(xx)

    if "trackId" not in df.columns:
        df["trackId"] = df["id"]

    # latVelocity: yVelocity (lateral in highD coordinate)
    if "latVelocity" not in df.columns:
        df["latVelocity"] = df.get("yVelocity", np.nan)

    # same-lane lead / rear
    if "leadId" not in df.columns:
        df["leadId"] = df.get("precedingId", 0)
    if "rearId" not in df.columns:
        df["rearId"] = df.get("followingId", 0)

    # adjacent-lane neighbors: rename highD columns to common names
    rename = {
        "leftPrecedingId":  "leftLeadId",
        "leftFollowingId":  "leftRearId",
        "rightPrecedingId": "rightLeadId",
        "rightFollowingId": "rightRearId",
        # typo variant present in some highD files
        "rightAlsongsideId": "rightAlongsideId",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    for col in ["leftLeadId", "leftRearId", "leftAlongsideId",
                "rightLeadId", "rightRearId", "rightAlongsideId"]:
        if col not in df.columns:
            df[col] = 0

    for col in ["frame", "laneId", "trackId", "recordingId"]:
        if col not in df.columns:
            raise KeyError(f"normalize_tracks: missing required column '{col}'")

    return df


def normalize_recmeta(recmeta: pd.DataFrame, xx: str) -> pd.DataFrame:
    """
    Normalize recordingMeta to columns: recordingId, frameRate.
    highD recordingMeta uses 'id' for the recording id.
    """
    rm = recmeta.copy()
    rm.columns = [c.strip() for c in rm.columns]

    if "recordingId" not in rm.columns:
        rm["recordingId"] = rm["id"] if "id" in rm.columns else int(xx)

    rm["recordingId"] = rm["recordingId"].astype(int)

    if "frameRate" not in rm.columns:
        raise KeyError("recordingMeta missing 'frameRate'")

    return rm[["recordingId", "frameRate"]].drop_duplicates()


# ─────────────────────────────────────────────────────────────────────────────
# Lane lookup
# ─────────────────────────────────────────────────────────────────────────────

def build_lane_lookup(df: pd.DataFrame) -> Dict[int, Dict[int, int]]:
    """
    Build lookup[trackId][frame] = laneId from the full recording DataFrame.
    Used for adjacency checks during cut-in detection.
    """
    tmp = df[["trackId", "frame", "laneId"]].copy()
    tmp["trackId"] = pd.to_numeric(tmp["trackId"], errors="coerce").astype(int)
    tmp["frame"]   = pd.to_numeric(tmp["frame"],   errors="coerce").astype(int)
    tmp["laneId"]  = pd.to_numeric(tmp["laneId"],  errors="coerce")

    lookup: Dict[int, Dict[int, int]] = {}
    for tid, g in tmp.groupby("trackId", sort=False):
        d: Dict[int, int] = {}
        for f, l in zip(g["frame"].to_numpy(), g["laneId"].to_numpy()):
            if not np.isnan(l):
                d[int(f)] = int(l)
        lookup[int(tid)] = d
    return lookup


def get_lane_at(lookup: Dict[int, Dict[int, int]], track_id: int, frame: int) -> Optional[int]:
    d = lookup.get(int(track_id))
    if d is None:
        return None
    return d.get(int(frame))


def is_adjacent_lane(ego_lane: int, nb_lane: int) -> bool:
    """highD: adjacent iff laneId differs by exactly 1."""
    return abs(int(ego_lane) - int(nb_lane)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Lane change detection (highD)
# ─────────────────────────────────────────────────────────────────────────────

def detect_lane_change(w: pd.DataFrame) -> Tuple[bool, int, Optional[int]]:
    """
    Detect lane changes in window w via laneId transitions.

    Returns
    -------
    has_lc      : bool
    lc_count    : int   (number of transitions)
    lc_frame    : Optional[int]  (frame of first transition, None if no LC)
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
# LC direction inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_lc_direction(w: pd.DataFrame, lc_frame: int, K: int = 5) -> Optional[str]:
    """
    Infer lane-change direction using yVelocity around lc_frame.
        yVelocity > 0  ->  right
        yVelocity < 0  ->  left
    K: half-window (frames) around lc_frame to average.
    """
    vcol = "yVelocity" if "yVelocity" in w.columns else "latVelocity"
    if vcol not in w.columns:
        return None

    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    win = df[(df["frame"] >= lc_frame - K) & (df["frame"] <= lc_frame + K)]
    v = pd.to_numeric(win[vcol], errors="coerce").dropna()
    if len(v) == 0:
        return None

    mv = float(v.mean())
    if mv > 0:
        return "right"
    if mv < 0:
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
    Scan the pre-LC window [lc_frame - W, lc_frame - 1].
    Return True if any vehicle in the target-side rear or alongside slot
    is confirmed to be in the immediately adjacent lane (|laneId diff| == 1).

    direction : "left" | "right"
    """
    if direction == "left":
        rear_col      = "leftRearId"
        alongside_col = "leftAlongsideId"
        lane_col_ego  = "laneId"
    else:
        rear_col      = "rightRearId"
        alongside_col = "rightAlongsideId"
        lane_col_ego  = "laneId"

    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    pre = df[(df["frame"] >= lc_frame - W) & (df["frame"] < lc_frame)]
    if len(pre) == 0:
        return False

    for _, row in pre.iterrows():
        f        = int(row["frame"])
        ego_lane = row.get(lane_col_ego, np.nan)
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
    lc_direction_K: int = 5,
) -> Dict:
    """
    Label one sample window.

    Event labels
    ────────────
    lane_change     : LC detected
    lane_following  : no LC detected

    Note: cut_in is intentionally folded into lane_change.

    Parameters
    ----------
    w       : DataFrame slice for this window (single trackId, sorted by frame)
    lookup  : lane_lookup built from the full recording
    W_adj   : look-back window (frames at native fps) for adjacency check
    lc_direction_K : half-window for velocity-based direction inference

    Returns
    -------
    dict with keys: event_label, lc_direction, lc_frame, lc_count,
                    has_adj_rear_or_alongside
    """
    out: Dict = {}

    has_lc, lc_count, lc_frame = detect_lane_change(w)
    out["lc_count"]  = lc_count
    out["lc_frame"]  = int(lc_frame) if lc_frame is not None else -1

    if not has_lc or lc_frame is None:
        out["event_label"]              = "lane_following"
        out["lc_direction"]             = "none"
        out["has_adj_rear_or_alongside"] = False
        return out

    # ── LC confirmed: all LC windows are folded into lane_change ─────────────
    direction = infer_lc_direction(w, lc_frame=lc_frame, K=lc_direction_K)
    out["lc_direction"] = direction if direction is not None else "unknown"
    if direction is None:
        out["event_label"]              = "lane_change"
        out["has_adj_rear_or_alongside"] = False
        return out

    has_adj = check_adjacent_rear_or_alongside(
        w=w,
        lc_frame=lc_frame,
        direction=direction,
        lookup=lookup,
        W=W_adj,
    )
    out["has_adj_rear_or_alongside"] = bool(has_adj)
    out["event_label"] = "lane_change"

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-recording processing
# ─────────────────────────────────────────────────────────────────────────────

def label_recording(
    xx: str,
    raw_dir: Path,
    keys: Optional[List[Tuple[int, int]]],   # [(trackId, t0_frame), ...] or None = all windows
    history_sec: float,
    future_sec: float,
    target_hz: float,
    stride_sec: float = 1.0,
    W_adj: int = 10,
) -> List[Dict]:
    """
    Label all requested sample windows for one recording.

    If `keys` is None (standalone mode), enumerate all windows via stride.
    If `keys` is provided (mmap mode), label only those exact windows.
    """
    tracks_path  = raw_dir / f"{xx}_tracks.csv"
    recmeta_path = raw_dir / f"{xx}_recordingMeta.csv"

    if not tracks_path.exists() or not recmeta_path.exists():
        print(f"  [SKIP] {xx}: raw files not found")
        return []

    tracks  = smart_read_csv(tracks_path)
    recmeta = smart_read_csv(recmeta_path)
    tracks.columns  = [c.strip() for c in tracks.columns]
    recmeta.columns = [c.strip() for c in recmeta.columns]

    tracks_n  = normalize_tracks(tracks, xx)
    recmeta_n = normalize_recmeta(recmeta, xx)

    fr = float(recmeta_n["frameRate"].iloc[0])
    if not np.isfinite(fr) or fr <= 0:
        print(f"  [SKIP] {xx}: invalid frameRate={fr}")
        return []

    # merge frameRate into tracks for convenience
    tracks_n = tracks_n.merge(recmeta_n, on="recordingId", how="left")
    tracks_n["frame"] = pd.to_numeric(tracks_n["frame"], errors="coerce").astype("Int64")
    tracks_n = tracks_n.dropna(subset=["frame"]).sort_values(["trackId", "frame"])

    # ds_step: native fps -> target_hz downsampling ratio
    ds_step = max(1, int(round(fr / target_hz)))
    # window length at native fps
    T  = int(round(history_sec * target_hz))
    Tf = int(round(future_sec  * target_hz))
    win_native = (T + Tf - 1) * ds_step   # span in native frames (inclusive)

    lookup = build_lane_lookup(tracks_n)
    by_tid = {int(tid): g for tid, g in tracks_n.groupby("trackId", sort=False)}

    rows: List[Dict] = []
    rid  = int(xx)

    if keys is not None:
        # ── mmap mode: label exactly the requested (tid, t0) keys ────────────
        for tid, t0_frame in sorted(set(keys)):
            g = by_tid.get(int(tid))

            base = {
                "recordingId": rid,
                "trackId":     int(tid),
                "t0_frame":    int(t0_frame),
                "frameRate":   fr,
                "ds_step":     ds_step,
                "history_sec": history_sec,
                "future_sec":  future_sec,
                "target_hz":   target_hz,
            }

            if g is None:
                rows.append({**base, "event_label": "unknown",
                              "lc_frame": -1, "lc_count": 0,
                              "lc_direction": "unknown",
                              "has_adj_rear_or_alongside": False,
                              "state_label": "unknown",
                              "occupancy": float("nan")})
                continue

            hist_frames = [int(t0_frame) - (T - 1 - i) * ds_step for i in range(T)]
            hist = g[g["frame"].astype(int).isin(hist_frames)]
            state_label, occupancy = compute_raw_state_label(hist)

            t1_frame = int(t0_frame) + win_native
            w = g[(g["frame"] >= t0_frame) & (g["frame"] <= t1_frame)]

            if len(w) == 0:
                rows.append({**base, "event_label": "unknown",
                              "lc_frame": -1, "lc_count": 0,
                              "lc_direction": "unknown",
                              "has_adj_rear_or_alongside": False,
                              "state_label": state_label,
                              "occupancy": occupancy})
                continue

            result = label_window(w, lookup=lookup, W_adj=W_adj)
            rows.append({**base, **result, "state_label": state_label, "occupancy": occupancy})

    else:
        # ── standalone mode: enumerate windows via stride ─────────────────────
        stride_native = max(1, int(round(stride_sec * fr)))

        for tid, g in by_tid.items():
            frames = g["frame"].dropna().astype(int).to_numpy()
            if len(frames) == 0:
                continue
            f0, f1 = int(frames.min()), int(frames.max())

            t0 = f0
            while t0 + win_native <= f1:
                t1 = t0 + win_native
                w = g[(g["frame"] >= t0) & (g["frame"] <= t1)]
                if len(w) > 0:
                    hist_frames = [int(t0) + i * ds_step for i in range(T)]
                    hist = g[g["frame"].astype(int).isin(hist_frames)]
                    state_label, occupancy = compute_raw_state_label(hist)
                    result = label_window(w, lookup=lookup, W_adj=W_adj)
                    rows.append({
                        "recordingId": rid,
                        "trackId":     int(tid),
                        "t0_frame":    int(t0),
                        "t1_frame":    int(t1),
                        "frameRate":   fr,
                        "ds_step":     ds_step,
                        "history_sec": history_sec,
                        "future_sec":  future_sec,
                        "target_hz":   target_hz,
                        "state_label": state_label,
                        "occupancy":   occupancy,
                        **result,
                    })
                t0 += stride_native

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker  (module-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_recording(
    item: Tuple,
    raw_dir: Path,
    history_sec: float,
    future_sec: float,
    target_hz: float,
    stride_sec: float,
    W_adj: int,
) -> Tuple[List[Dict], Optional[List[Tuple[int, int]]], Optional[List[int]]]:
    xx, keys = item
    if keys is not None:
        keys_for_rec  = [(tid, t0) for tid, t0, _ in keys]
        mmap_idx_list = [idx       for _,   _,  idx in keys]
    else:
        keys_for_rec  = None
        mmap_idx_list = None

    rows = label_recording(
        xx          = xx,
        raw_dir     = raw_dir,
        keys        = keys_for_rec,
        history_sec = history_sec,
        future_sec  = future_sec,
        target_hz   = target_hz,
        stride_sec  = stride_sec,
        W_adj       = W_adj,
    )
    return rows, keys_for_rec, mmap_idx_list


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="highD window-level scenario labeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir",     default="data/highD",          help="Base data directory")
    ap.add_argument("--raw_dir",      default="raw",                 help="Raw CSV subdir under data_dir")
    ap.add_argument("--mmap_dir",     default="mmap",                help="Mmap subdir under data_dir (leave empty for standalone mode)")
    ap.add_argument("--out_csv",      default="scenario_labels.csv", help="Output CSV filename (saved inside mmap_dir)")

    ap.add_argument("--history_sec",  type=float, default=2.0)
    ap.add_argument("--future_sec",   type=float, default=5.0)
    ap.add_argument("--target_hz",    type=float, default=3.0)
    ap.add_argument("--stride_sec",   type=float, default=1.0,  help="Stride (standalone mode only)")
    ap.add_argument("--W_adj",        type=int,   default=25,   help="Pre-LC look-back window (native frames). 25 = 1sec at 25fps")
    ap.add_argument("--num_workers",  type=int,   default=0,    help="Worker processes (0 = os.cpu_count())")
    return ap.parse_args()


def main() -> None:
    args   = parse_args()
    data_dir = Path(args.data_dir)
    raw_dir  = data_dir / args.raw_dir
    mmap_dir = data_dir / args.mmap_dir
    out_csv  = mmap_dir / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Determine mode ────────────────────────────────────────────────────────
    keys_by_xx: Optional[Dict[str, List[Tuple[int, int, int]]]] = None  # (tid, t0, mmap_idx)

    if args.mmap_dir:
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

        # state label: nb_mask에서 직접 계산
        nb_mask_path = mmap_dir / "nb_mask.npy"
        if nb_mask_path.exists():
            nb_mask_arr              = np.load(nb_mask_path, mmap_mode="r")
            state_labels, state_occ  = compute_state_labels(nb_mask_arr)
            print(f"  state dense={(state_labels=='dense').mean()*100:.1f}%  "
                  f"free_flow={(state_labels=='free_flow').mean()*100:.1f}%")
            if float(state_occ.max()) <= STATE_THRESHOLD:
                print("  [WARN] nb_mask occupancy never exceeds the dense threshold. "
                      "If preprocessing used --gate_topn, raw neighbor IDs will be used "
                      "for state labels instead of gated nb_mask.")
        else:
            state_labels = None
            state_occ    = None
            print("  [WARN] nb_mask.npy not found — state_label will be 'unknown'")

        keys_by_xx = {}
        for i, (rid, tid, t0) in enumerate(zip(rids, tids, t0s)):
            xx = f"{int(rid):02d}"
            keys_by_xx.setdefault(xx, []).append((int(tid), int(t0), i))

        n_total = len(rids)
        print(f"[Mode] mmap  |  {n_total:,} samples  |  {len(keys_by_xx)} recordings")
    else:
        xxs = find_recording_ids(raw_dir)
        keys_by_xx   = {xx: None for xx in xxs}   # type: ignore[assignment]
        n_total      = None
        state_labels = None
        state_occ    = None
        print(f"[Mode] standalone  |  {len(keys_by_xx)} recordings  |  stride={args.stride_sec}s")

    # ── Process recordings (parallel) ────────────────────────────────────────
    n_workers = args.num_workers if args.num_workers > 0 else os.cpu_count()
    print(f"[Process] {len(keys_by_xx)} recordings  |  workers={n_workers}")

    all_rows: List[Dict] = []
    items = sorted(keys_by_xx.items())

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(_process_one_recording, item,
                           raw_dir, args.history_sec, args.future_sec,
                           args.target_hz, args.stride_sec, args.W_adj): item[0]
                for item in items}
        for fut in tqdm(concurrent.futures.as_completed(futs),
                        total=len(futs), desc="Labeling recordings"):
            rows, keys_for_rec, mmap_idx_list = fut.result()

            # state label 조인
            if state_labels is not None and mmap_idx_list is not None:
                idx_map = {(tid, t0): mmap_idx_list[i]
                           for i, (tid, t0) in enumerate(keys_for_rec)}
                for row in rows:
                    if row.get("state_label") != "unknown" and np.isfinite(row.get("occupancy", np.nan)):
                        continue
                    mi = idx_map.get((row["trackId"], row["t0_frame"]))
                    if mi is not None:
                        row["state_label"] = state_labels[mi]
                        row["occupancy"]   = float(state_occ[mi])
                    else:
                        row["state_label"] = "unknown"
                        row["occupancy"]   = float("nan")
            else:
                for row in rows:
                    row["state_label"] = "unknown"
                    row["occupancy"]   = float("nan")

            all_rows.extend(rows)

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["recordingId", "trackId", "t0_frame"])
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] {len(df):,} labels -> {out_csv}")

    if n_total is not None and len(df) != n_total:
        print(f"[WARN] mmap samples={n_total}, labeled={len(df)} (delta={n_total - len(df)})")

    if "event_label" in df.columns:
        print("\nEvent label counts:")
        print(df["event_label"].value_counts().to_string())

    if "state_label" in df.columns:
        print("\nState label counts:")
        print(df["state_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
