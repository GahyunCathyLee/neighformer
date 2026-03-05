#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log.py — Evaluation result logging to CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def log_eval_to_csv(
    *,
    csv_out: Path,
    exp_tag: str,
    cfg_path: Path,
    ckpt_path: Path,
    split: str,
    ego_features: List[str],
    nb_features: List[str],
    metrics: Dict[str, Any],
) -> None:
    """
    평가 결과를 CSV 에 한 행씩 append 합니다.

    Parameters
    ----------
    exp_tag      : config의 exp_tag
    ego_features : dataset.feature_names["ego"]
    nb_features  : dataset.feature_names["nb"]
    metrics      : evaluate() 반환 dict
    """
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, Any] = {
        "exp_tag":     str(exp_tag),
        "split":       str(split),
        "config":      str(cfg_path),
        "ckpt":        str(ckpt_path),
        # feature summary
        "ego":         ", ".join(ego_features),
        "nb":          ", ".join(nb_features),
        # core metrics
        "loss":        _f(metrics.get("loss")),
        "ade":         _f(metrics.get("ade")),
        "fde":         _f(metrics.get("fde")),
        "rmse":        _f(metrics.get("rmse")),
        "rmse_1s":     _f(metrics.get("rmse_1s")),
        "rmse_2s":     _f(metrics.get("rmse_2s")),
        "rmse_3s":     _f(metrics.get("rmse_3s")),
        "rmse_4s":     _f(metrics.get("rmse_4s")),
        "rmse_5s":     _f(metrics.get("rmse_5s")),
    }

    df = pd.DataFrame([row])
    df.to_csv(csv_out, mode="a", header=not csv_out.exists(), index=False)


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x) if x is not None else default
    except Exception:
        return default