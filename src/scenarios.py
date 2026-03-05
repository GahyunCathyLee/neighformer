#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenarios.py — Scenario label utilities and weighted sampling helpers.

labels_lut 구조
────────────────
  {(recordingId, trackId, t0_frame): {"event_label": str, "state_label": str}}

WeightedRandomSampler 사용 흐름
────────────────────────────────
  labels_lut = load_scenario_labels(path)
  weights    = build_sample_weights(dataset, labels_df, mode="event")
  sampler    = WeightedRandomSampler(weights[train_idx], num_samples=len(train_ds))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Label constants  (scenario_label.py 기준)
# ──────────────────────────────────────────────────────────────────────────────
EVENT_LABELS = ["cut_in", "lane_change", "lane_following"]
STATE_LABELS = ["dense", "free_flow"]

LabelMode = Literal["event", "state"]


# ──────────────────────────────────────────────────────────────────────────────
# labels_lut loader
# ──────────────────────────────────────────────────────────────────────────────

def load_scenario_labels(
    path: Path,
) -> Optional[Dict[Tuple[int, int, int], Dict[str, Any]]]:
    """
    scenario_labels.csv → labels_lut dict.

    Returns None (with warning) if the file is missing or lacks required columns.

    CSV 필수 컬럼: recordingId, trackId, t0_frame
    선택 컬럼   : event_label, state_label
    """
    path = Path(path)
    if not path.exists():
        print(f"[WARN] scenario_labels not found: {path} → stratified eval disabled")
        return None

    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[WARN] scenario_labels missing columns {missing} → stratified eval disabled")
        return None

    has_event = "event_label" in df.columns
    has_state = "state_label" in df.columns
    if not has_event and not has_state:
        print("[WARN] scenario_labels has no event_label/state_label → stratified eval disabled")
        return None

    lut: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        key = (int(row.recordingId), int(row.trackId), int(row.t0_frame))
        lut[key] = {
            "event_label": getattr(row, "event_label", None) if has_event else None,
            "state_label": getattr(row, "state_label", None) if has_state else None,
        }

    print(f"\n[INFO] Loaded scenario labels: {len(lut):,} entries from {path}")
    return lut


# ──────────────────────────────────────────────────────────────────────────────
# Weighted sampling
# ──────────────────────────────────────────────────────────────────────────────

def build_sample_weights(
    dataset,
    labels_csv: Path | str,
    mode: LabelMode = "event",
    alpha: float = 0.5,
    unknown_weight: float = 0.0,
    clip_max: Optional[float] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    WeightedRandomSampler 용 per-sample 가중치를 계산합니다.

    pandas merge 기반 벡터화 구현으로 대규모 데이터셋에서도 빠릅니다.

    Parameters
    ----------
    dataset       : HighDDataset  (meta_rec / meta_track / meta_frame 필드 필요)
    labels_csv    : scenario_labels.csv 경로
    mode          : "event" | "state"
    alpha         : 0 = uniform, 0.5 = sqrt inverse freq (권장), 1.0 = full inverse freq
    unknown_weight: 라벨 미매칭 샘플 가중치 (0 = 샘플링 제외)
    clip_max      : 가중치 상한 클리핑 (None = 클리핑 없음)
    verbose       : 분포 요약 출력 여부

    Returns
    -------
    weights : (N,) torch.DoubleTensor  — 전체 데이터셋 기준 인덱스
              train split 에만 쓰려면 weights[train_idx] 로 슬라이싱
    """
    # ── 메타 배열 확인 ────────────────────────────────────────────────────────
    for attr in ("meta_rec", "meta_track", "meta_frame"):
        if not hasattr(dataset, attr) or getattr(dataset, attr) is None:
            raise AttributeError(
                f"dataset.{attr} is None. "
                "HighDDataset을 return_meta=True 없이 생성했거나 meta 파일이 없습니다."
            )

    # ── labels CSV 로드 ───────────────────────────────────────────────────────
    col = "event_label" if mode == "event" else "state_label"
    labels_df = pd.read_csv(labels_csv)[["recordingId", "trackId", "t0_frame", col]]

    # ── 데이터셋 메타 → DataFrame ─────────────────────────────────────────────
    meta_df = pd.DataFrame({
        "recordingId": dataset.meta_rec,
        "trackId":     dataset.meta_track,
        "t0_frame":    dataset.meta_frame,
    })

    # ── left join으로 라벨 매칭 ───────────────────────────────────────────────
    merged = meta_df.merge(labels_df, on=["recordingId", "trackId", "t0_frame"], how="left")
    merged[col] = merged[col].fillna("unknown")

    # ── 가중치 계산 ───────────────────────────────────────────────────────────
    known_mask = merged[col] != "unknown"
    counts     = merged.loc[known_mask, col].value_counts().to_dict()

    def _weight(label: str) -> float:
        if label == "unknown":
            return float(unknown_weight)
        c = counts.get(label, 1)
        return 1.0 / (c ** float(alpha))

    weights = merged[col].map(_weight).to_numpy(dtype=np.float64)

    if clip_max is not None:
        weights = np.minimum(weights, float(clip_max))

    if weights.sum() <= 0:
        weights[:] = 1.0

    # ── 분포 요약 출력 ────────────────────────────────────────────────────────
    if verbose:
        total_known = sum(counts.values())
        n_unknown   = int((~known_mask).sum())
        print(
            f"[SCENARIO-SAMPLING] mode={mode}  alpha={alpha}  "
            f"known={total_known:,}  unknown={n_unknown:,}"
        )
        print("  raw label counts:")
        for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {lbl:<20s} {cnt:>8,}  ({cnt / total_known * 100:.1f}%)")

        # 기대 샘플링 비율: mass ∝ count^(1-alpha)
        mass     = {lbl: cnt ** (1.0 - float(alpha)) for lbl, cnt in counts.items()}
        mass_sum = sum(mass.values()) or 1.0
        print("  expected sampling share (approx):")
        for lbl, m in sorted(mass.items(), key=lambda x: -x[1]):
            print(f"    {lbl:<20s} {m / mass_sum * 100:.1f}%")

    return torch.from_numpy(weights).double()