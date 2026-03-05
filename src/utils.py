#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — General-purpose utilities and stats helpers.

Stats 파일명 규칙  (make_stats_filename)
────────────────────────────────────────
  TB{t_back}_TF{t_front}_vy{vy_eps*100:02d}.npz
  예) TB5_TF3_vy27.npz
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_path(p: str | Path) -> Path:
    """상대 경로는 프로젝트 루트 기준으로 절대 경로로 변환합니다."""
    p = Path(p)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / p).resolve()


# ──────────────────────────────────────────────────────────────────────────────
# Type conversion
# ──────────────────────────────────────────────────────────────────────────────

def _to_int(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.item())
    if isinstance(x, np.generic):
        return int(x)
    return int(x)


# ──────────────────────────────────────────────────────────────────────────────
# Latency measurement
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_latency_ms(
    fn: Callable[[], Any],
    device: torch.device,
    iters: int = 200,
    warmup: int = 30,
) -> Dict[str, float]:
    """
    fn() 의 호출 지연시간을 측정합니다.
    CUDA 환경에서는 torch.cuda.Event 기반 정밀 측정을 사용합니다.

    Returns
    -------
    {"avg_ms", "p50_ms", "p90_ms", "p99_ms", "iters"}
    """
    for _ in range(max(0, warmup)):
        fn()

    times_ms = []

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for _ in range(max(1, iters)):
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "iters":  float(len(arr)),
    }