#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Training entry point for EncDecFormer.

Usage
─────
  python train.py --config configs/exp0.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.dataset import HighDDataset, collate_fn, load_stats
from src.model import EncDecFormer, build_model, build_scheduler
from src.trainer import EvalConfig, TrainConfig, evaluate, train_one_epoch
from src.scenarios import load_scenario_labels, build_sample_weights
from src.stats import assert_stats_dims, compute_stats_if_needed, make_stats_filename
from src.utils import resolve_path, set_seed


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get(d: Dict, *keys, default=None):
    """중첩 dict에서 안전하게 값을 꺼냅니다."""
    for k in keys[:-1]:
        d = d.get(k, {})
    return d.get(keys[-1], default)


def _build_dataset(
    data_dir: Path,
    split_indices: np.ndarray,
    stats: Optional[Dict],
    feat: Dict[str, Any],
    return_meta: bool,
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
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    # ── 1. Environment ────────────────────────────────────────────────────────
    seed = int(_get(cfg, "train", "seed", default=42))
    set_seed(seed)

    want_cuda = str(_get(cfg, "train", "device", default="cuda")).lower() == "cuda"
    device    = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

    print("====== Environment ======")
    print(f"  torch  : {torch.__version__}\ndevice : {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
        torch.backends.cudnn.benchmark = True
    else:
        print()

    # ── 2. Config ─────────────────────────────────────────────────────────────
    data_cfg  = cfg.get("data",     {})
    feat_cfg  = cfg.get("features", {})
    train_cfg = cfg.get("train",    {})
    sam_cfg   = cfg.get("scenario_sampling", None)
    exp_tag   = str(cfg.get("exp_tag", "default"))

    mmap_dir   = resolve_path(data_cfg["mmap_dir"])
    splits_dir = resolve_path(data_cfg["splits_dir"])
    stats_dir  = resolve_path(data_cfg["stats_dir"])
    data_hz    = float(data_cfg.get("hz", 3.0))

    print("\n====== Features ======")
    for k in ("ego_mode", "nb_kin_mode",
              "use_lc_state", "use_lit", "use_lis",
              "use_gate", "use_I_x", "use_I_y", "use_I"):
        print(f"  {k:<18s} = {feat_cfg.get(k)}")

    # ── 3. Scenario Labels (optional) ─────────────────────────────────────────
    labels_path = data_cfg.get("scenario_labels", None)
    labels_lut  = load_scenario_labels(Path(labels_path)) if labels_path else None
    do_strat    = bool(train_cfg.get("stratified_eval", False)) and labels_lut is not None
    use_sampling = sam_cfg is not None and labels_lut is not None

    # ── 4. Stats ──────────────────────────────────────────────────────────────
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
    stats_path  = stats_dir / stats_fname

    compute_stats_if_needed(
        stats_path   = stats_path,
        data_dir     = mmap_dir,
        splits_dir   = splits_dir,
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
    stats = load_stats(stats_path)
    print(f"[INFO] Stats : {stats_path}")

    # ── 5. Datasets ───────────────────────────────────────────────────────────
    train_idx = np.load(splits_dir / "train_indices.npy")
    val_idx   = np.load(splits_dir / "val_indices.npy")

    train_ds = _build_dataset(mmap_dir, train_idx, stats, feat_cfg, return_meta=use_sampling)
    val_ds   = _build_dataset(mmap_dir, val_idx,   stats, feat_cfg, return_meta=True)

    print(f"[INFO] Dataset  train={len(train_ds):,}  val={len(val_ds):,}  {train_ds}")

    # stats 차원 검증
    assert_stats_dims(
        stats      = stats,
        ego_dim    = train_ds.ego_feature_dim,
        nb_dim     = train_ds.nb_feature_dim,
        stats_path = stats_path,
    )

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    batch_size  = int(data_cfg.get("batch_size",  512))
    num_workers = int(data_cfg.get("num_workers",   8))

    train_sampler = None
    if use_sampling:
        sam_mode   = str(sam_cfg.get("mode",           "event"))
        alpha      = float(sam_cfg.get("alpha",          0.5))
        unknown_w  = float(sam_cfg.get("unknown_weight", 0.0))
        clip_max   = sam_cfg.get("clip_max", None)
        if clip_max is not None:
            clip_max = float(clip_max)

        print(f"\n[INFO] Scenario sampling ON  mode={sam_mode}  alpha={alpha}")
        full_weights  = build_sample_weights(
            dataset    = train_ds,
            labels_csv = labels_path,
            mode       = sam_mode,
            alpha      = alpha,
            unknown_weight = unknown_w,
            clip_max   = clip_max,
        )
        train_weights = full_weights[train_idx]
        print(f"[INFO] Valid weights: {(train_weights > 0).sum():,} / {len(train_weights):,}")
        train_sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_ds), replacement=True
        )

    _loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
        collate_fn  = collate_fn,
        prefetch_factor      = 4 if num_workers > 0 else None,
        persistent_workers   = num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        shuffle = train_sampler is None,
        sampler = train_sampler,
        drop_last = True,
        **_loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **_loader_kwargs)

    # ── 7. Model / Optimizer / Scheduler ─────────────────────────────────────
    model: EncDecFormer = build_model(
        cfg,
        ego_dim = train_ds.ego_feature_dim,
        nb_dim  = train_ds.nb_feature_dim,
    ).to(device)
    print(f"[INFO] Model  params={sum(p.numel() for p in model.parameters()):,}  "
          f"ego_dim={train_ds.ego_feature_dim}  nb_dim={train_ds.nb_feature_dim}")

    lr           = float(train_cfg.get("lr",           3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = bool(train_cfg.get("use_amp", True)) and (device.type == "cuda")
    scaler  = GradScaler("cuda", enabled=use_amp)

    epochs       = int(train_cfg.get("epochs",       50))
    total_steps  = epochs * max(1, len(train_loader))
    scheduler    = build_scheduler(
        optimizer,
        total_steps  = total_steps,
        warmup_steps = int(train_cfg.get("warmup_steps", 0)),
        sched_type   = str(train_cfg.get("lr_schedule", "none")),
    )

    # ── 8. Loop configs ───────────────────────────────────────────────────────
    ckpt_dir = resolve_path(train_cfg.get("ckpt_dir", "ckpts")) / exp_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_obj = TrainConfig(
        use_amp        = use_amp,
        grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0)),
        w_ade          = float(train_cfg.get("w_ade",  1.0)),
        w_fde          = float(train_cfg.get("w_fde",  0.0)),
        w_rmse         = float(train_cfg.get("w_rmse", 0.0)),
        w_cls          = float(train_cfg.get("w_cls",  1.0)),
    )
    eval_cfg_obj = EvalConfig(
        use_amp         = use_amp,
        w_ade           = train_cfg_obj.w_ade,
        w_fde           = train_cfg_obj.w_fde,
        w_rmse          = train_cfg_obj.w_rmse,
        w_cls           = train_cfg_obj.w_cls,
        data_hz         = data_hz,
        labels_lut      = labels_lut if do_strat else None,
        save_event_path = ckpt_dir / "val_event.csv" if do_strat else None,
        save_state_path = ckpt_dir / "val_state.csv" if do_strat else None,
    )

    # ── 9. TensorBoard ────────────────────────────────────────────────────────
    tb_dir = resolve_path("tensorboard") / exp_tag
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[INFO] TensorBoard : {tb_dir}")
    print(f"[INFO] Checkpoint  : {ckpt_dir}")

    # ── 10. Training loop ─────────────────────────────────────────────────────
    monitor     = str(train_cfg.get("monitor", "val_ade")).lower()
    best_score  = float("inf")
    best_path   = ckpt_dir / "best.pt"
    global_step = 0

    print(f"\n====== Train ======")
    print(f"  Epochs={epochs}  bs={batch_size}  monitor={monitor}  ckpt={ckpt_dir}")

    for ep in range(1, epochs + 1):
        print(f"\n====== Epoch {ep}/{epochs} ======")

        tr = train_one_epoch(
            model       = model,
            loader      = train_loader,
            device      = device,
            optimizer   = optimizer,
            scheduler   = scheduler,
            scaler      = scaler,
            cfg         = train_cfg_obj,
            global_step = global_step,
            epoch       = ep,
        )
        global_step = int(tr["global_step_end"])

        va = evaluate(
            model  = model,
            loader = val_loader,
            device = device,
            cfg    = eval_cfg_obj,
            epoch  = ep,
        )

        print(
            f"Train loss={tr['loss']:.4f}  ADE={tr['ade']:.3f} | "
            f"Val loss={va['loss']:.4f}  ADE={va['ade']:.3f}  "
            f"FDE={va['fde']:.3f}  RMSE={va['rmse']:.3f}"
        )

        # TensorBoard
        tb_writer.add_scalar("Loss/train",    tr["loss"],  ep)
        tb_writer.add_scalar("Loss/val",      va["loss"],  ep)
        tb_writer.add_scalar("Metrics/ADE_train", tr["ade"],  ep)
        tb_writer.add_scalar("Metrics/ADE_val",   va["ade"],  ep)
        tb_writer.add_scalar("Metrics/FDE_val",   va["fde"],  ep)
        tb_writer.add_scalar("Metrics/RMSE_val",  va["rmse"], ep)

        # monitor score
        score = va.get(monitor.removeprefix("val_"), va["loss"])

        # checkpoint
        save_dict = {
            "epoch":         ep,
            "global_step":   global_step,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "cfg":           cfg,
            "best_score":    best_score,
            "monitor":       monitor,
        }
        torch.save(save_dict, ckpt_dir / "last.pt")

        if score < best_score:
            best_score = score
            torch.save(save_dict, best_path)
            print(f" ✅ best ckpt saved → {best_path}  ({monitor}={best_score:.4f})")

        # 100 epoch 단위 스냅샷
        if ep % 100 == 0:
            snap = ckpt_dir / f"best_{ep}.pt"
            torch.save(torch.load(best_path, weights_only=True), snap)
            print(f" 💾 Snapshot → {snap}")

    print(f"\n[DONE] Training finished.  Best {monitor}: {best_score:.4f}")
    tb_writer.close()


if __name__ == "__main__":
    main()