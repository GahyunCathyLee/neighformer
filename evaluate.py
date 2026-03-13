#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — Evaluation entry point for EncDecFormer.

Usage
─────
  python evaluate.py --ckpt ckpts/exp0/best.pt
  python evaluate.py --ckpt ckpts/exp0/best.pt --split test
  python evaluate.py --ckpt ckpts/exp0/best.pt --scenario_labels data/scenario_labels.csv
  python evaluate.py --ckpt ckpts/exp0/best.pt --measure_time
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HighDDataset, collate_fn, load_stats
from src.metrics import ade, fde, rmse
from src.model import EncDecFormer, build_model
from src.scenarios import load_scenario_labels
from src.stats import make_stats_filename
from src.trainer import _forward, _resolve_pred_abs
from src.utils import _to_int, resolve_path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get(d: Dict, *keys, default=None):
    for k in keys[:-1]:
        d = d.get(k, {})
    return d.get(keys[-1], default)


def _build_dataset(
    data_dir: Path,
    split_indices: np.ndarray,
    stats,
    feat: Dict[str, Any],
    return_meta: bool = False,
) -> HighDDataset:
    return HighDDataset(
        data_dir      = data_dir,
        split_indices = split_indices,
        stats         = stats,
        ego_mode      = str(feat.get("ego_mode",     "pva")),
        nb_kin_mode   = str(feat.get("nb_kin_mode",  "pva")),
        use_lc_state  = bool(feat.get("use_lc_state", False)),
        use_lit       = bool(feat.get("use_lit",      False)),
        use_lis       = bool(feat.get("use_lis",      False)),
        use_gate      = bool(feat.get("use_gate",     False)),
        use_I_x       = bool(feat.get("use_I_x",      False)),
        use_I_y       = bool(feat.get("use_I_y",      False)),
        use_I         = bool(feat.get("use_I",         False)),
        return_meta   = return_meta,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Device info
# ──────────────────────────────────────────────────────────────────────────────

def print_device_info(device: torch.device) -> None:
    """Print device info that affects inference latency."""
    print("\n====== Device Info ======")
    print(f"  PyTorch version : {torch.__version__}")
    if device.type == "cuda":
        idx   = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        alloc_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        free_gb  = total_gb - alloc_gb
        print(f"  Device          : {props.name}  (index={idx})")
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  cuDNN version   : {torch.backends.cudnn.version()}")
        print(f"  SM count        : {props.multi_processor_count}")
        print(f"  VRAM            : {total_gb:.1f} GB total  /  {free_gb:.1f} GB free")
        print(f"  cuDNN benchmark : {torch.backends.cudnn.benchmark}")
        print(f"  AMP (bfloat16)  : enabled")
    else:
        import platform
        cpu_name = platform.processor() or platform.machine() or "unknown"
        print(f"  Device          : CPU  ({cpu_name})")


# ──────────────────────────────────────────────────────────────────────────────
# Latency measurement
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_latency(
    fn,
    device: torch.device,
    warmup: int = 1000,
    iters:  int = 10000,
) -> Dict[str, float]:
    """
    Measure single-inference latency (avg / min / max) in milliseconds.

    CUDA: torch.cuda.Event 기반 정밀 측정
    CPU : time.perf_counter 기반 측정
    """
    print(f"  GPU warm-up  : {warmup:,} iters ...", end=" ", flush=True)
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("done")

    times_ms = []
    print(f"  Measurement  : {iters:,} iters ...", end=" ", flush=True)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for _ in range(iters):
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    print("done")

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

# [sum_ade, sum_fde, sum_rmse, count]
_ScenStats = Dict[str, list]


@torch.no_grad()
def run_evaluate(
    model:      EncDecFormer,
    loader:     DataLoader,
    device:     torch.device,
    data_hz:    float,
    use_amp:    bool,
    labels_lut: Optional[Dict[Tuple[int, int, int], Dict[str, Any]]] = None,
) -> Tuple[Dict[str, float], Optional[_ScenStats], Optional[_ScenStats]]:
    """
    Full evaluation pass.

    Returns
    -------
    results    : overall metrics dict  (ade, fde, rmse, rmse_Xs, n_samples)
    ev_stats   : per event_label accumulators  (or None)
    st_stats   : per state_label accumulators  (or None)
    """
    model.eval()
    hz = float(data_hz)

    sum_ade = sum_fde = sum_rmse = 0.0
    n_samples = 0

    eval_secs = [1, 2, 3, 4, 5]
    rmse_se  = {s: 0.0 for s in eval_secs}
    rmse_cnt = {s: 0   for s in eval_secs}

    do_strat = labels_lut is not None
    # [sum_ade, sum_fde, sum_rmse, count]
    ev_stats: _ScenStats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    st_stats: _ScenStats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])

    pbar = tqdm(loader, desc="Evaluating", dynamic_ncols=True, leave=True)

    for batch in pbar:
        x_ego   = batch["x_ego"].to(device,  non_blocking=True)
        x_nb    = batch["x_nb"].to(device,   non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)
        y_abs   = batch["y"].to(device,       non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            pred, scores = _forward(model, x_ego, x_nb, nb_mask)
            pred_abs, _  = _resolve_pred_abs(pred, scores)

        B      = pred_abs.shape[0]
        Tf_cur = pred_abs.shape[1]

        ade_s  = ade(pred_abs,  y_abs)
        fde_s  = fde(pred_abs,  y_abs)
        rmse_s = rmse(pred_abs, y_abs)

        dist = torch.norm(pred_abs - y_abs, dim=-1)  # (B, Tf)
        for sec in eval_secs:
            idx = int(sec * hz) - 1
            if 0 <= idx < Tf_cur:
                rmse_se[sec]  += (dist[:, idx] ** 2).sum().item()
                rmse_cnt[sec] += B

        sum_ade  += ade_s.sum().item()
        sum_fde  += fde_s.sum().item()
        sum_rmse += rmse_s.sum().item()
        n_samples += B

        # ── stratified accumulation ───────────────────────────────────────────
        if do_strat and "meta" in batch:
            ade_np  = ade_s.cpu().numpy()
            fde_np  = fde_s.cpu().numpy()
            rmse_np = rmse_s.cpu().numpy()

            for i, meta in enumerate(batch["meta"]):
                if i >= B:
                    break
                key = (
                    _to_int(meta.get("recordingId")),
                    _to_int(meta.get("trackId")),
                    _to_int(meta.get("t0_frame")),
                )
                lab = labels_lut.get(key)
                if lab is None:
                    continue
                ev = lab.get("event_label") or "unknown"
                st = lab.get("state_label") or "unknown"
                for acc, lbl in ((ev_stats, ev), (st_stats, st)):
                    acc[lbl][0] += float(ade_np[i])
                    acc[lbl][1] += float(fde_np[i])
                    acc[lbl][2] += float(rmse_np[i])
                    acc[lbl][3] += 1

        pbar.set_postfix(
            ADE=f"{ade_s.mean().item():.4f}",
            FDE=f"{fde_s.mean().item():.4f}",
        )

    N = max(1, n_samples)
    results: Dict[str, float] = {
        "ade":       sum_ade  / N,
        "fde":       sum_fde  / N,
        "rmse":      sum_rmse / N,
        "n_samples": float(n_samples),
    }
    for sec in eval_secs:
        results[f"rmse_{sec}s"] = (
            math.sqrt(rmse_se[sec] / rmse_cnt[sec]) if rmse_cnt[sec] > 0 else float("nan")
        )

    return results, (ev_stats if do_strat else None), (st_stats if do_strat else None)


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sep(widths, left="+", mid="+", right="+", fill="-") -> str:
    return left + mid.join(fill * w for w in widths) + right


def print_metrics(results: Dict[str, float]) -> None:
    """Print ADE / FDE / RMSE summary tables."""
    # ── Table 1 : ADE | FDE | RMSE ───────────────────────────────────────────
    c1 = 15
    ws = [c1, c1, c1]

    print()
    print(_sep(ws))
    print(f"|{'ADE':^{c1}}|{'FDE':^{c1}}|{'RMSE':^{c1}}|")
    print(_sep(ws))
    print(f"|{results['ade']:^{c1}.4f}|{results['fde']:^{c1}.4f}|{results['rmse']:^{c1}.4f}|")
    print(_sep(ws))

    # ── Table 2 : RMSE @1s … @5s ─────────────────────────────────────────────
    secs  = [1, 2, 3, 4, 5]
    c2    = 9
    inner = c2 * len(secs) + (len(secs) - 1)
    title = f"{'RMSE':^{inner}}"

    print()
    print(f"+{'-' * inner}+")
    print(f"|{title}|")
    print(_sep([c2] * len(secs)))
    print("|" + "|".join(f"{'@' + str(s) + 's':^{c2}}" for s in secs) + "|")
    print(_sep([c2] * len(secs)))
    vals = [results.get(f"rmse_{s}s", float("nan")) for s in secs]
    print("|" + "|".join(f"{v:^{c2}.4f}" for v in vals) + "|")
    print(_sep([c2] * len(secs)))


def print_scenario_results(
    stats: _ScenStats,
    label_type: str,
) -> None:
    """
    Print per-scenario ADE / FDE / RMSE table.

    Parameters
    ----------
    stats      : {label: [sum_ade, sum_fde, sum_rmse, count]}
    label_type : "Event" or "State"
    """
    if not stats:
        return

    # sort: known labels first (alphabetically), unknown last
    rows = sorted(stats.items(), key=lambda x: (x[0] == "unknown", x[0]))

    c_lbl = max(len(lbl) for lbl, _ in rows)
    c_lbl = max(c_lbl, len(label_type)) + 2   # padding
    c_n   = 9
    c_m   = 11   # ADE / FDE / RMSE columns

    ws = [c_lbl, c_n, c_m, c_m, c_m]

    print(f"\n====== Scenario Results [{label_type}] ======")
    print(_sep(ws))
    print(
        f"|{label_type:^{c_lbl}}|{'n':^{c_n}}"
        f"|{'ADE':^{c_m}}|{'FDE':^{c_m}}|{'RMSE':^{c_m}}|"
    )
    print(_sep(ws))

    total_n = sum(v[3] for v in stats.values())
    for lbl, (sa, sf, sr, n) in rows:
        if n == 0:
            continue
        print(
            f"|{lbl:^{c_lbl}}|{n:^{c_n},}"
            f"|{sa/n:^{c_m}.4f}|{sf/n:^{c_m}.4f}|{sr/n:^{c_m}.4f}|"
        )

    print(_sep(ws))

    # total row
    total_sa = sum(v[0] for v in stats.values())
    total_sf = sum(v[1] for v in stats.values())
    total_sr = sum(v[2] for v in stats.values())
    N = max(1, total_n)
    print(
        f"|{'Total':^{c_lbl}}|{total_n:^{c_n},}"
        f"|{total_sa/N:^{c_m}.4f}|{total_sf/N:^{c_m}.4f}|{total_sr/N:^{c_m}.4f}|"
    )
    print(_sep(ws))


def print_latency(lat: Dict[str, float], batch_size: int, warmup: int, iters: int) -> None:
    """Print latency avg / min / max table."""
    c = 15
    ws = [c, c, c]

    print()
    print(f"  Batch size : {batch_size}   Warmup : {warmup:,}   Measurement : {iters:,}")
    print()
    print(_sep(ws))
    print(f"|{'Avg (ms)':^{c}}|{'Min (ms)':^{c}}|{'Max (ms)':^{c}}|")
    print(_sep(ws))
    print(f"|{lat['avg_ms']:^{c}.2f}|{lat['min_ms']:^{c}.2f}|{lat['max_ms']:^{c}.2f}|")
    print(_sep(ws))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate EncDecFormer checkpoint.")
    ap.add_argument("--ckpt",             type=str, required=True,
                    help="Path to .pt checkpoint file")
    ap.add_argument("--split",            type=str, default="test",
                    choices=["train", "val", "test"],
                    help="Dataset split to evaluate on  (default: test)")
    ap.add_argument("--scenario",         action="store_true",
                    help="Enable per-scenario breakdown (reads path from checkpoint cfg)")
    ap.add_argument("--batch_size",       type=int, default=None,
                    help="Override batch size from saved config")
    ap.add_argument("--num_workers",      type=int, default=None,
                    help="Override num_workers from saved config")
    ap.add_argument("--device",           type=str, default=None,
                    help="Override device  (cuda / cpu)")
    ap.add_argument("--measure_time",     action="store_true",
                    help="Measure inference latency (1,000 warmup + 10,000 iters)")
    args = ap.parse_args()

    LATENCY_WARMUP = 1_000
    LATENCY_ITERS  = 10_000

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Dict[str, Any] = ckpt["cfg"]
    print(f"[INFO] Checkpoint : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # ── 2. Device ─────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        want_cuda = str(_get(cfg, "train", "device", default="cuda")).lower() == "cuda"
        device    = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    gpu_name = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"[INFO] Device     : {device}{gpu_name}")

    # ── 3. Config ─────────────────────────────────────────────────────────────
    data_cfg  = cfg.get("data",     {})
    feat_cfg  = cfg.get("features", {})
    train_cfg = cfg.get("train",    {})

    mmap_dir   = resolve_path(data_cfg["mmap_dir"])
    splits_dir = resolve_path(data_cfg["splits_dir"])
    stats_dir  = resolve_path(data_cfg["stats_dir"])
    data_hz    = float(data_cfg.get("hz", 3.0))
    use_amp    = bool(train_cfg.get("use_amp", True)) and (device.type == "cuda")

    # ── 4. Scenario labels (optional) ─────────────────────────────────────────
    labels_lut = None
    if args.scenario and not args.measure_time:
        labels_path = data_cfg.get("scenario_labels", None)
        if labels_path:
            labels_lut = load_scenario_labels(resolve_path(labels_path))
        else:
            print("[WARN] --scenario: cfg has no data.scenario_labels → skipping")

    # ── 5. Stats ──────────────────────────────────────────────────────────────
    stats_fname = make_stats_filename(
        ego_mode     = str(feat_cfg.get("ego_mode",    "pva")),
        nb_kin_mode  = str(feat_cfg.get("nb_kin_mode", "pva")),
        use_lc_state = bool(feat_cfg.get("use_lc_state", False)),
        use_lit      = bool(feat_cfg.get("use_lit",      False)),
        use_lis      = bool(feat_cfg.get("use_lis",      False)),
        use_gate     = bool(feat_cfg.get("use_gate",     False)),
        use_I_x      = bool(feat_cfg.get("use_I_x",      False)),
        use_I_y      = bool(feat_cfg.get("use_I_y",      False)),
        use_I        = bool(feat_cfg.get("use_I",         False)),
    )
    stats = load_stats(stats_dir / stats_fname)

    # ── 6. Dataset & DataLoader ───────────────────────────────────────────────
    split_file = splits_dir / f"{args.split}_indices.npy"
    if not split_file.exists():
        raise FileNotFoundError(f"Split index file not found: {split_file}")
    split_idx = np.load(split_file)

    need_meta = labels_lut is not None
    ds = _build_dataset(mmap_dir, split_idx, stats, feat_cfg, return_meta=need_meta)
    print(f"[INFO] Dataset    : {args.split} split  n={len(ds):,}  {ds}")

    batch_size  = args.batch_size if args.batch_size is not None \
                  else int(data_cfg.get("batch_size", 512))
    num_workers = args.num_workers if args.num_workers is not None \
                  else int(data_cfg.get("num_workers", 8))

    # Latency mode uses a single sample directly; loader is only needed for metrics
    loader = None
    if not args.measure_time:
        loader = DataLoader(
            ds,
            batch_size          = batch_size,
            shuffle             = False,
            drop_last           = False,
            num_workers         = num_workers,
            pin_memory          = True,
            collate_fn          = collate_fn,
            prefetch_factor     = 4 if num_workers > 0 else None,
            persistent_workers  = num_workers > 0,
        )

    # ── 7. Model ──────────────────────────────────────────────────────────────
    model: EncDecFormer = build_model(
        cfg,
        ego_dim = ds.ego_feature_dim,
        nb_dim  = ds.nb_feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] Model      : params={sum(p.numel() for p in model.parameters()):,}")

    # ── 8a. Latency mode ──────────────────────────────────────────────────────
    if args.measure_time:
        print_device_info(device)

        # Single sample (batch_size=1) for accurate per-inference latency
        sample = ds[0]
        x_ego_l   = sample["x_ego"].unsqueeze(0).to(device)    # (1, T, ego_dim)
        x_nb_l    = sample["x_nb"].unsqueeze(0).to(device)     # (1, T, K, nb_dim)
        nb_mask_l = sample["nb_mask"].unsqueeze(0).to(device)  # (1, T, K)

        @torch.no_grad()
        def _infer():
            with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                pred, scores = _forward(model, x_ego_l, x_nb_l, nb_mask_l)
                if pred.dim() == 4:
                    _resolve_pred_abs(pred, scores)

        print(f"\n====== Inference Latency ======")
        lat = measure_latency(_infer, device, warmup=LATENCY_WARMUP, iters=LATENCY_ITERS)
        print_latency(lat, batch_size=1, warmup=LATENCY_WARMUP, iters=LATENCY_ITERS)

    # ── 8b. Metric evaluation mode ────────────────────────────────────────────
    else:
        print(f"\n====== Evaluation  [{args.split}] ======")
        results, ev_stats, st_stats = run_evaluate(
            model, loader, device, data_hz, use_amp, labels_lut
        )

        print(f"\n  n_samples = {int(results['n_samples']):,}")
        print_metrics(results)

        if ev_stats:
            print_scenario_results(ev_stats, label_type="Event")
        if st_stats:
            print_scenario_results(st_stats, label_type="State")

    print()


if __name__ == "__main__":
    main()
