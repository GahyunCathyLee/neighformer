#!/usr/bin/env python3
"""
Benchmark highD feature-input generation latency for one recording.

This script reuses data/highD/preprocess.py directly and measures the latency of
_recording_to_buf(), i.e. raw CSV -> in-memory feature arrays for a single
recording. It does not write mmap files.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_preprocess():
    from preprocess import Config, _recording_to_buf, find_recording_ids

    return Config, _recording_to_buf, find_recording_ids


def _build_config(args: argparse.Namespace) -> Any:
    Config, _, _ = _load_preprocess()
    return Config(
        data_dir=Path(args.data_dir),
        raw_dir=Path(args.raw_dir),
        mmap_dir=Path("__latency_benchmark_unused__"),
        target_hz=args.target_hz,
        history_sec=args.history_sec,
        future_sec=args.future_sec,
        stride_sec=args.stride_sec,
        normalize_upper_xy=args.normalize_upper_xy,
        lis_mode=args.lis_mode,
        lambda_x=args.lambda_x,
        lambda_y=args.lambda_y,
        alpha=args.alpha,
        beta=args.beta,
        gate_topn=args.gate_topn,
        slot_importance_alpha=args.slot_importance_alpha,
        slot_importance_conditional=args.slot_importance_conditional,
        non_relative=args.non_relative,
        dry_run=True,
        num_workers=1,
    )


def _resolve_recording_id(cfg: Any, requested: str | None) -> str:
    _, _, find_recording_ids = _load_preprocess()
    rec_ids = find_recording_ids(cfg.raw_path)
    if not rec_ids:
        raise FileNotFoundError(f"No recordings found in {cfg.raw_path}")
    if requested is None:
        return rec_ids[0]

    requested = str(requested).zfill(2)
    if requested not in rec_ids:
        raise FileNotFoundError(
            f"Recording {requested!r} not found in {cfg.raw_path}. "
            f"Available example: {rec_ids[0]}"
        )
    return requested


def _summarize_array_shapes(buf: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
    keys = ["x_ego", "x_nb", "nb_mask", "y", "y_vel", "y_acc", "x_last_abs"]
    return {key: list(buf[key].shape) for key in keys if key in buf}


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    _, _recording_to_buf, _ = _load_preprocess()
    cfg = _build_config(args)
    rec_id = _resolve_recording_id(cfg, args.recording_id)

    last_buf = None
    for i in range(args.warmup):
        if not args.quiet:
            print(f"[warmup {i + 1}/{args.warmup}] recording={rec_id}")
        last_buf = _recording_to_buf(cfg, rec_id)
        if last_buf is None:
            raise RuntimeError(f"Recording {rec_id} produced no samples.")
        del last_buf
        gc.collect()

    latencies_ms: List[float] = []
    sample_count = 0
    shapes: Dict[str, List[int]] = {}

    for i in range(args.repeat):
        if not args.quiet:
            print(f"[run {i + 1}/{args.repeat}] recording={rec_id}")
        gc.collect()
        t0 = time.perf_counter_ns()
        last_buf = _recording_to_buf(cfg, rec_id)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        if last_buf is None:
            raise RuntimeError(f"Recording {rec_id} produced no samples.")

        sample_count = int(last_buf["x_ego"].shape[0])
        shapes = _summarize_array_shapes(last_buf)
        latencies_ms.append(elapsed_ms)
        del last_buf

    mean_ms = statistics.fmean(latencies_ms)
    per_sample_ms = [v / sample_count for v in latencies_ms] if sample_count else []
    result: Dict[str, Any] = {
        "recording_id": rec_id,
        "data_dir": str(cfg.data_dir),
        "raw_dir": str(cfg.raw_path),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "samples": sample_count,
        "latency_ms": {
            "runs": latencies_ms,
            "mean": mean_ms,
            "median": statistics.median(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "stdev": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        },
        "latency_per_sample_ms": {
            "runs": per_sample_ms,
            "mean": statistics.fmean(per_sample_ms) if per_sample_ms else None,
            "median": statistics.median(per_sample_ms) if per_sample_ms else None,
            "min": min(per_sample_ms) if per_sample_ms else None,
            "max": max(per_sample_ms) if per_sample_ms else None,
            "stdev": statistics.stdev(per_sample_ms) if len(per_sample_ms) > 1 else 0.0,
        },
        "shapes": shapes,
        "config": {
            "target_hz": cfg.target_hz,
            "history_sec": cfg.history_sec,
            "future_sec": cfg.future_sec,
            "stride_sec": cfg.stride_sec,
            "normalize_upper_xy": cfg.normalize_upper_xy,
            "lis_mode": cfg.lis_mode,
            "lambda_x": cfg.lambda_x,
            "lambda_y": cfg.lambda_y,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "gate_topn": cfg.gate_topn,
            "slot_importance_alpha": cfg.slot_importance_alpha,
            "slot_importance_conditional": cfg.slot_importance_conditional,
            "non_relative": cfg.non_relative,
        },
        "device": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
    return result


def print_report(result: Dict[str, Any]) -> None:
    latency = result["latency_ms"]
    per_sample = result["latency_per_sample_ms"]
    print("\n[highD preprocess latency]")
    print(f"recording          : {result['recording_id']}")
    print(f"raw_dir            : {result['raw_dir']}")
    print(f"samples            : {result['samples']:,}")
    print(f"repeat / warmup    : {result['repeat']} / {result['warmup']}")
    print(f"mean recording ms  : {latency['mean']:.3f}")
    print(f"median recording ms: {latency['median']:.3f}")
    print(f"min / max ms       : {latency['min']:.3f} / {latency['max']:.3f}")
    print(f"stdev ms           : {latency['stdev']:.3f}")
    if per_sample["mean"] is not None:
        print(f"mean per sample ms : {per_sample['mean']:.6f}")
        print(f"median/sample ms   : {per_sample['median']:.6f}")
        print(f"min / max sample ms: {per_sample['min']:.6f} / {per_sample['max']:.6f}")
        print(f"stdev/sample ms    : {per_sample['stdev']:.6f}")
    print(f"shapes             : {result['shapes']}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Measure highD preprocess feature generation latency for one recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir", default="data/highD", help="Base data directory")
    ap.add_argument("--raw_dir", default="raw", help="Raw CSV subdir under data_dir")
    ap.add_argument("--recording_id", default=None, help="Recording id, e.g. 01. Default: first available")
    ap.add_argument("--repeat", type=int, default=5, help="Measured runs")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs excluded from stats")
    ap.add_argument("--json_out", default=None, help="Optional path to save the full JSON result")
    ap.add_argument("--quiet", action="store_true", help="Only print final report")

    ap.add_argument("--target_hz", type=float, default=3.0)
    ap.add_argument("--history_sec", type=float, default=2.0)
    ap.add_argument("--future_sec", type=float, default=5.0)
    ap.add_argument("--stride_sec", type=float, default=1.0)
    ap.add_argument("--normalize_upper_xy", action="store_true", default=True)
    ap.add_argument("--lis_mode", default="7", choices=["3", "5", "7", "9"])
    ap.add_argument("--lambda_x", type=float, default=0.1)
    ap.add_argument("--lambda_y", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gate_topn", type=int, default=0)
    ap.add_argument("--slotImportance", type=float, default=0.0, dest="slot_importance_alpha")
    ap.add_argument(
        "--slotImportanceConditional",
        action="store_true",
        default=False,
        dest="slot_importance_conditional",
    )
    ap.add_argument("--non_relative", action="store_true", default=False)

    args = ap.parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print_report(result)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json saved          : {out}")


if __name__ == "__main__":
    main()
