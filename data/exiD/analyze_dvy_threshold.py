#!/usr/bin/env python3
"""
analyze_dvy_threshold_exid.py — closing-in / moving-away threshold 분석 (exiD용)

highD 버전과 동일한 분석 로직. exiD 포맷 차이만 반영:
  - trackId (not id), xCenter/yCenter (not x/y)
  - lonVelocity/latVelocity (not xVelocity/yVelocity)
  - laneletId (not laneId)  → LC 감지는 laneletId 변화로 판단
  - neighbor cols: leadId/rearId/leftLeadId/... (not precedingId/followingId/...)
  - neighbor ID missing value: -1 (not 0)
  - 차량 크기: tracks 행의 width/length (not tracksMeta)
  - drivingDirection flip 없음

두 그룹 분리:
  Cross-lane (|dy| >= dy_same):
    - direction filter: dy * dvy < 0  (closing-in)
    - signal: instant |dvy|  AND  1-sec avg |dvy|
    - label : LC  OR  |dy| < dy_crit  in future

  Same-lane  (|dy| <  dy_same):
    - direction filter: 없음
    - signal: instant |dvy|  AND  1-sec avg |dvy|
    - label : LC  OR  |dy| > dy_same  in future  (separation event)

Usage:
    python analyze_dvy_threshold_exid.py --raw_dir data/exiD/raw
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# exiD neighbor columns (순서는 highD와 동일하게 유지: lead/rear → left → right)
NEIGHBOR_COLS = [
    "leadId",
    "rearId",
    "leftLeadId",
    "leftAlongsideId",
    "leftRearId",
    "rightLeadId",
    "rightAlongsideId",
    "rightRearId",
]
TARGET_HZ = 3.0
EXID_DEFAULT_HZ = 25.0


def find_recording_ids(raw_dir: Path) -> List[str]:
    return sorted(
        re.match(r"(\d+)_tracks\.csv$", p.name).group(1)
        for p in raw_dir.glob("*_tracks.csv")
        if re.match(r"(\d+)_tracks\.csv$", p.name)
    )


def get_frame_rate(rec_meta: pd.DataFrame) -> float:
    if "frameRate" in rec_meta.columns:
        return float(rec_meta.loc[0, "frameRate"])
    return EXID_DEFAULT_HZ


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_recording(
    raw_dir: Path, rec_id: str,
    dy_crit: float, dy_same: float, future_sec: float, win_sec: float,
) -> Dict[str, np.ndarray]:

    rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv", low_memory=False)

    frame_rate = get_frame_rate(rec_meta)
    step  = max(1, int(round(frame_rate / TARGET_HZ)))
    win_n = max(1, int(round(win_sec * TARGET_HZ)))
    fut_n = max(1, int(round(future_sec * TARGET_HZ)))

    # neighbor 컬럼 누락 시 -1로 채움 (exiD missing = -1)
    for c in NEIGHBOR_COLS:
        if c not in tracks.columns:
            tracks[c] = -1

    tracks = tracks.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)

    frame_arr = tracks["frame"].to_numpy(np.int32)
    # exiD: xCenter/yCenter (이미 center 좌표)
    y_arr     = tracks["yCenter"].astype(np.float32).to_numpy()
    vy_arr    = tracks["latVelocity"].astype(np.float32).to_numpy()
    # exiD: laneletId → LC 감지에 사용
    lane_arr  = tracks["laneletId"].fillna(-1).astype(np.int32).to_numpy() \
                if "laneletId" in tracks.columns \
                else np.full(len(tracks), -1, dtype=np.int32)

    # neighbor ID 파싱 (세미콜론 다중값 대응, missing=-1)
    nb_id_cols = []
    for c in NEIGHBOR_COLS:
        s = tracks[c].astype(str).str.strip().str.split(";").str[0]
        nb_id_cols.append(pd.to_numeric(s, errors="coerce").fillna(-1).astype(np.int32).to_numpy())
    nb_arr = np.stack(nb_id_cols, axis=1)   # (N, 8)

    # per-vehicle 인덱스 구축
    per_vid_rows: Dict[int, np.ndarray]     = {}
    per_vid_f2r:  Dict[int, Dict[int, int]] = {}
    for v, idxs in tracks.groupby("trackId").indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame_arr[idxs])]
        per_vid_rows[int(v)] = idxs
        per_vid_f2r[int(v)]  = {int(frame_arr[r]): int(r) for r in idxs}

    # LC 이벤트: laneletId 변화 프레임 (laneletId=-1이면 무시)
    lc_event: Dict[int, set] = {}
    for v, idxs in per_vid_rows.items():
        lids = lane_arr[idxs]
        # -1인 구간은 변화로 취급하지 않음
        chg = np.where(
            (lids[1:] != lids[:-1]) & (lids[1:] >= 0) & (lids[:-1] >= 0)
        )[0] + 1
        lc_event[v] = {int(frame_arr[idxs[i]]) for i in chg}

    # output lists
    cx_inst, cx_avg, cx_dy = [], [], []
    cx_lc,   cx_dyc        = [], []
    sa_inst, sa_avg, sa_dy = [], [], []
    sa_lc,   sa_dys        = [], []
    dy_all                 = []

    for v, idxs in per_vid_rows.items():
        frs     = frame_arr[idxs]
        sub_frs = frs[::step]
        f2r_v   = per_vid_f2r[v]

        for sf in sub_frs:
            if sf not in f2r_v:
                continue

            fut_frames = [sf + (i + 1) * step for i in range(fut_n)]
            if not all(f in f2r_v for f in fut_frames):
                continue

            win_frames = [sf - w * step for w in range(win_n)]

            ego_r  = f2r_v[sf]
            ego_y  = float(y_arr[ego_r])
            ego_vy = float(vy_arr[ego_r])

            ego_vy_win = []
            for wf in win_frames:
                wr = f2r_v.get(wf)
                if wr is not None:
                    ego_vy_win.append(float(vy_arr[wr]))
            if not ego_vy_win:
                continue

            ego_lc_fut = any(f in lc_event.get(v, set()) for f in fut_frames)
            ids8 = nb_arr[ego_r]

            for nid in ids8.tolist():
                nid = int(nid)
                if nid <= 0:   # exiD missing = -1, 0도 유효하지 않은 값으로 처리
                    continue
                f2r_n = per_vid_f2r.get(nid)
                if f2r_n is None or sf not in f2r_n:
                    continue

                nb_r  = f2r_n[sf]
                nb_y  = float(y_arr[nb_r])
                nb_vy = float(vy_arr[nb_r])
                dy    = nb_y - ego_y
                dvy   = nb_vy - ego_vy

                nb_vy_win = []
                for wf in win_frames:
                    wr = f2r_n.get(wf)
                    er = f2r_v.get(wf)
                    if wr is not None and er is not None:
                        nb_vy_win.append(float(vy_arr[wr]) - float(vy_arr[er]))
                avg_dvy = float(np.mean(nb_vy_win)) if nb_vy_win else dvy

                dy_all.append(abs(dy))

                nb_lc_fut  = any(f in lc_event.get(nid, set()) for f in fut_frames)
                lbl_lc_val = int(ego_lc_fut or nb_lc_fut)

                if abs(dy) >= dy_same:
                    # Cross-lane: closing-in direction only
                    if dy * dvy >= 0:
                        continue
                    lbl_dyc = 0
                    for ff in fut_frames:
                        nr = f2r_n.get(ff)
                        er = f2r_v.get(ff)
                        if nr is not None and er is not None:
                            if abs(float(y_arr[nr]) - float(y_arr[er])) < dy_crit:
                                lbl_dyc = 1
                                break
                    cx_inst.append(abs(dvy))
                    cx_avg.append(abs(avg_dvy))
                    cx_dy.append(abs(dy))
                    cx_lc.append(lbl_lc_val)
                    cx_dyc.append(lbl_dyc)

                else:
                    # Same-lane: all directions
                    lbl_dys = 0
                    for ff in fut_frames:
                        nr = f2r_n.get(ff)
                        er = f2r_v.get(ff)
                        if nr is not None and er is not None:
                            if abs(float(y_arr[nr]) - float(y_arr[er])) > dy_same:
                                lbl_dys = 1
                                break
                    sa_inst.append(abs(dvy))
                    sa_avg.append(abs(avg_dvy))
                    sa_dy.append(abs(dy))
                    sa_lc.append(lbl_lc_val)
                    sa_dys.append(lbl_dys)

    def arr(lst, dtype=np.float32): return np.array(lst, dtype)

    return {
        "cross_inst":    arr(cx_inst),
        "cross_avg":     arr(cx_avg),
        "cross_dy":      arr(cx_dy),
        "cross_lbl_lc":  arr(cx_lc,  np.int8),
        "cross_lbl_dyc": arr(cx_dyc, np.int8),
        "same_inst":     arr(sa_inst),
        "same_avg":      arr(sa_avg),
        "same_dy":       arr(sa_dy),
        "same_lbl_lc":   arr(sa_lc,  np.int8),
        "same_lbl_dys":  arr(sa_dys, np.int8),
        "dy_all":        arr(dy_all),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Precision curve helpers  (highD 버전과 동일)
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall(
    signal: np.ndarray, labels: np.ndarray, min_support: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order      = np.argsort(signal)
    sig_s      = signal[order]
    lb_s       = labels[order].astype(np.int64)
    rev_lc     = np.concatenate([np.cumsum(lb_s[::-1])[::-1], [0]])
    rev_all    = np.arange(len(sig_s), -1, -1, dtype=np.int64)
    n_lc_total = int(labels.sum())
    x_max      = float(sig_s[-min_support]) if len(sig_s) >= min_support else float(sig_s[-1])
    ths        = np.linspace(0.0, x_max * 1.05, 600)
    idx        = np.searchsorted(sig_s, ths, side="right")
    n_ab       = rev_all[idx]
    lc_ab      = rev_lc[idx]
    prec       = np.where(n_ab >= min_support, lc_ab / np.maximum(n_ab, 1), np.nan)
    rec        = lc_ab / max(n_lc_total, 1)
    return ths, prec, rec


def smooth(arr: np.ndarray, w: int = 20) -> np.ndarray:
    valid = ~np.isnan(arr)
    k = np.ones(w) / w
    s = np.convolve(np.where(valid, arr, 0.0), k, mode="same")
    c = np.convolve(valid.astype(float),       k, mode="same")
    return np.where(c > 0.3, s / np.maximum(c, 1e-9), np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers  (highD 버전과 동일)
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_panel(
    ax, signal_inst: np.ndarray, signal_avg: np.ndarray,
    labels: np.ndarray, title: str,
    confidences: Tuple[float, ...], min_support: int,
) -> None:
    ax2 = ax.twinx()

    for sig, style, sig_label in [
        (signal_inst, "-",  "instant dvy"),
        (signal_avg,  "--", "1s avg dvy"),
    ]:
        if len(sig) < min_support * 2:
            continue
        ths, prec, rec = precision_recall(sig, labels, min_support)
        prec_s = smooth(prec)

        ax.plot(ths, prec_s, color="green", lw=2.0, linestyle=style,
                label=f"Prec ({sig_label})")
        ax.plot(ths, rec,    color="gray",  lw=1.0, linestyle=style, alpha=0.7,
                label=f"Recall ({sig_label})")

        if style == "-":
            for conf, mc in zip(confidences, ["red", "blue", "purple"]):
                valid = np.where(prec_s >= conf)[0]
                if len(valid):
                    ti = valid[0]
                    ax.axvline(ths[ti], color=mc, lw=1.2, linestyle=":")
                    ax.annotate(f"{conf*100:.0f}%\n{ths[ti]:.2f}",
                                xy=(ths[ti], conf),
                                xytext=(ths[ti] + ths[-1]*0.04, conf - 0.12),
                                fontsize=7, color=mc)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("|dvy|  (m/s)", fontsize=8)
    ax.set_ylabel("Prec / Recall", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax2.set_visible(False)


def print_summary(
    tag: str, signal_inst: np.ndarray, signal_avg: np.ndarray,
    labels: np.ndarray, label_name: str,
    confidences: Tuple[float, ...], min_support: int,
) -> None:
    if labels.sum() == 0:
        return
    print(f"  [{tag}] label={label_name}  N={len(labels):,}  pos={labels.sum():,}")
    for sig, sname in [(signal_inst, "instant"), (signal_avg, "1s-avg")]:
        ths, prec, rec = precision_recall(sig, labels, min_support)
        prec_s = smooth(prec)
        for th_ref in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
            ti = np.searchsorted(ths, th_ref)
            if ti < len(prec) and not np.isnan(prec[ti]):
                print(f"    {sname:8s} |dvy|>{th_ref:.1f}: "
                      f"prec={prec[ti]:.3f}  rec={rec[ti]:.3f}")
        for conf in confidences:
            valid = np.where(prec_s >= conf)[0]
            if len(valid):
                ti = valid[0]
                print(f"    {sname:8s} P>={conf*100:.0f}% → "
                      f"th={ths[ti]:.3f}m/s  prec={prec_s[ti]:.3f}  rec={rec[ti]:.3f}")
            else:
                print(f"    {sname:8s} P>={conf*100:.0f}% → not reached "
                      f"(max={np.nanmax(prec_s):.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis  (highD 버전과 동일)
# ─────────────────────────────────────────────────────────────────────────────

def analyze(data: Dict[str, np.ndarray], out_prefix: str,
            dy_crit: float, dy_same: float, future_sec: float,
            confidences: Tuple[float, ...], min_support: int) -> None:

    cx_inst  = data["cross_inst"];  cx_avg = data["cross_avg"]
    cx_dy    = data["cross_dy"]
    cx_lc    = data["cross_lbl_lc"]; cx_dyc = data["cross_lbl_dyc"]
    cx_either = np.clip(cx_lc + cx_dyc, 0, 1).astype(np.int8)

    sa_inst  = data["same_inst"];   sa_avg = data["same_avg"]
    sa_dy    = data["same_dy"]
    sa_lc    = data["same_lbl_lc"]; sa_dys = data["same_lbl_dys"]
    sa_either = np.clip(sa_lc + sa_dys, 0, 1).astype(np.int8)

    dy_all = data["dy_all"]

    print(f"\n=== Cross-lane closing-in (|dy|>={dy_same}) ===")
    print(f"  N={len(cx_inst):,}  |dvy| mean={cx_inst.mean():.3f}  "
          f"lc_pos={cx_lc.sum():,}  dyc_pos={cx_dyc.sum():,}  either={cx_either.sum():,}")
    for lbl, name in [(cx_lc, "lc"), (cx_dyc, f"|dy|<{dy_crit}"), (cx_either, "either")]:
        print_summary("cross", cx_inst, cx_avg, lbl, name, confidences, min_support)

    print(f"\n=== Same-lane (|dy|<{dy_same}) ===")
    print(f"  N={len(sa_inst):,}  |dvy| mean={sa_inst.mean():.3f}  "
          f"lc_pos={sa_lc.sum():,}  dys_pos={sa_dys.sum():,}  either={sa_either.sum():,}")
    for lbl, name in [(sa_lc, "lc"), (sa_dys, f"|dy|>{dy_same}"), (sa_either, "either")]:
        print_summary("same ", sa_inst, sa_avg, lbl, name, confidences, min_support)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.35)

    cross_labels = [
        (cx_lc,     f"LC within {future_sec:.0f}s"),
        (cx_dyc,    f"|dy|<{dy_crit}m within {future_sec:.0f}s"),
        (cx_either, "LC OR |dy|<crit"),
    ]
    for col, (lbl, title) in enumerate(cross_labels):
        plot_precision_panel(
            axes[0, col], cx_inst, cx_avg, lbl,
            f"[Cross-lane] {title}", confidences, min_support,
        )

    same_labels = [
        (sa_lc,     f"LC within {future_sec:.0f}s"),
        (sa_dys,    f"|dy|>{dy_same}m within {future_sec:.0f}s"),
        (sa_either, "LC OR |dy|>same"),
    ]
    for col, (lbl, title) in enumerate(same_labels):
        plot_precision_panel(
            axes[1, col], sa_inst, sa_avg, lbl,
            f"[Same-lane] {title}", confidences, min_support,
        )

    ax_dy = axes[2, 0]
    x_dy  = np.linspace(0, min(float(np.percentile(dy_all, 99)), 12), 300)
    for vy, label_, color, alpha in [
        (dy_all,  "all |dy|",           "gray",     0.30),
        (cx_dy,   "cross-lane closing", "steelblue", 0.45),
        (sa_dy,   "same-lane",          "orange",    0.45),
    ]:
        if len(vy) > 10:
            kde = gaussian_kde(vy, bw_method=0.15)
            ax_dy.fill_between(x_dy, kde(x_dy), alpha=alpha, color=color, label=label_)
    ax_dy.axvline(dy_same, color="orange", linestyle="--", lw=1.5,
                  label=f"dy_same={dy_same}m")
    ax_dy.axvline(dy_crit, color="red",    linestyle="--", lw=1.5,
                  label=f"dy_crit={dy_crit}m")
    ax_dy.set_xlabel("|dy|  (m)"); ax_dy.set_ylabel("Density")
    ax_dy.set_title("|dy| Distribution", fontsize=9)
    ax_dy.legend(fontsize=7)

    ax_cx = axes[2, 1]
    x_cx  = np.linspace(0, float(np.percentile(cx_inst, 99.5)), 300)
    for vy, label_, color, alpha in [
        (cx_inst[cx_either==0], "no event (inst)", "steelblue", 0.35),
        (cx_inst[cx_either==1], "event (inst)",    "red",       0.45),
        (cx_avg[cx_either==0],  "no event (avg)",  "lightblue", 0.35),
        (cx_avg[cx_either==1],  "event (avg)",     "salmon",    0.45),
    ]:
        if len(vy) > 10:
            kde = gaussian_kde(vy, bw_method=0.2)
            ax_cx.fill_between(x_cx, kde(x_cx), alpha=alpha, color=color, label=label_)
    ax_cx.set_xlabel("|dvy|  (m/s)"); ax_cx.set_ylabel("Density")
    ax_cx.set_title("[Cross-lane] |dvy| Distribution", fontsize=9)
    ax_cx.legend(fontsize=7)

    ax_sa = axes[2, 2]
    x_sa  = np.linspace(0, float(np.percentile(sa_inst, 99.5)), 300)
    for vy, label_, color, alpha in [
        (sa_inst[sa_either==0], "no event (inst)", "steelblue", 0.35),
        (sa_inst[sa_either==1], "event (inst)",    "red",       0.45),
        (sa_avg[sa_either==0],  "no event (avg)",  "lightblue", 0.35),
        (sa_avg[sa_either==1],  "event (avg)",     "salmon",    0.45),
    ]:
        if len(vy) > 10:
            kde = gaussian_kde(vy, bw_method=0.2)
            ax_sa.fill_between(x_sa, kde(x_sa), alpha=alpha, color=color, label=label_)
    ax_sa.set_xlabel("|dvy|  (m/s)"); ax_sa.set_ylabel("Density")
    ax_sa.set_title("[Same-lane] |dvy| Distribution", fontsize=9)
    ax_sa.legend(fontsize=7)

    row_labels = [
        f"Row 0: Cross-lane (|dy|≥{dy_same}, closing-in direction)",
        f"Row 1: Same-lane  (|dy|<{dy_same}, all directions)",
        "Row 2: Distributions",
    ]
    fig.text(0.01, 0.97, " │ ".join(row_labels), fontsize=7, va="top", color="gray")
    fig.suptitle(
        f"dvy Threshold Analysis [exiD]  "
        f"(future={future_sec:.0f}s, dy_same={dy_same}m, dy_crit={dy_crit}m, "
        f"solid=instant / dashed=1s-avg)",
        fontsize=10, y=0.995,
    )

    out = f"{out_prefix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw_dir",     default="data/exiD/raw")
    ap.add_argument("--out",         default="dvy_threshold_exid")
    ap.add_argument("--dy_crit",     type=float, default=3.4)
    ap.add_argument("--dy_same",     type=float, default=1.5)
    ap.add_argument("--future_sec",  type=float, default=5.0)
    ap.add_argument("--win_sec",     type=float, default=1.0)
    ap.add_argument("--confidences", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--min_support", type=int,   default=100)
    return ap.parse_args()


def main() -> None:
    args    = parse_args()
    raw_dir = Path(args.raw_dir)
    rec_ids = find_recording_ids(raw_dir)
    if not rec_ids:
        raise FileNotFoundError(f"No recordings in {raw_dir}")

    print(f"[analyze_dvy_threshold_exid]  {len(rec_ids)} recordings  "
          f"dy_same={args.dy_same}m  dy_crit={args.dy_crit}m  "
          f"future={args.future_sec}s  win={args.win_sec}s")

    combined: Dict[str, List[np.ndarray]] = {}
    for rec_id in rec_ids:
        print(f"  {rec_id} ...", end="", flush=True)
        res = collect_recording(
            raw_dir, rec_id,
            args.dy_crit, args.dy_same, args.future_sec, args.win_sec,
        )
        cx = len(res["cross_inst"]); sa = len(res["same_inst"])
        print(f" cross={cx:,}  same={sa:,}")
        for k, v in res.items():
            combined.setdefault(k, []).append(v)

    data = {k: np.concatenate(v) for k, v in combined.items()}
    analyze(data, args.out, args.dy_crit, args.dy_same,
            args.future_sec, tuple(args.confidences), args.min_support)


if __name__ == "__main__":
    main()