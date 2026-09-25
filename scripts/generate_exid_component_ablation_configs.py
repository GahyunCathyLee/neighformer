#!/usr/bin/env python3
"""Generate exiD component removal ablation configs for NeighFormer.

This ablation keeps the full feature set fixed and compares preprocessing-time
components that change the mmap tensors. The main comparisons are removals
from the full preprocessing:

1. full: neighbor filtering + slot weighting
2. no_nf: slot weighting only, i.e. neighbor filtering removed
3. no_slot: neighbor filtering only, i.e. slot weighting removed
4. no_both: both removed
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_SEEDS = [42, 1234, 3407]
DEFAULT_CONDITION_SLUGS = ["no_nf", "no_slot", "no_both"]


def _as_bool_text(value: bool) -> str:
    return "true" if value else "false"


def _slug_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_conditions(
    *,
    base_mmap_dir: str,
    variant_root: str,
    topn: int,
    slot_alpha: float,
    slot_conditional: bool,
) -> List[Dict[str, Any]]:
    slot_suffix = f"slot_a{_slug_float(slot_alpha)}"
    if slot_conditional:
        slot_suffix += "_cond"

    return [
        {
            "order": 1,
            "slug": "full",
            "label": f"full (neighbor filtering top-{topn} + slot weighting"
            + (" conditional" if slot_conditional else " global")
            + ")",
            "mmap_dir": f"{variant_root.rstrip('/')}/nf_top{topn}_{slot_suffix}",
            "gate_topn": int(topn),
            "slot_importance_alpha": float(slot_alpha),
            "slot_importance_conditional": bool(slot_conditional),
        },
        {
            "order": 2,
            "slug": "no_nf",
            "label": "without neighbor filtering (slot weighting only)",
            "mmap_dir": f"{variant_root.rstrip('/')}/{slot_suffix}",
            "gate_topn": 0,
            "slot_importance_alpha": float(slot_alpha),
            "slot_importance_conditional": bool(slot_conditional),
        },
        {
            "order": 3,
            "slug": "no_slot",
            "label": f"without slot weighting (neighbor filtering top-{topn} only)",
            "mmap_dir": f"{variant_root.rstrip('/')}/nf_top{topn}",
            "gate_topn": int(topn),
            "slot_importance_alpha": 0.0,
            "slot_importance_conditional": False,
        },
        {
            "order": 4,
            "slug": "no_both",
            "label": "without neighbor filtering and without slot weighting",
            "mmap_dir": f"{variant_root.rstrip('/')}/base_I",
            "gate_topn": 0,
            "slot_importance_alpha": 0.0,
            "slot_importance_conditional": False,
        },
    ]


def _render_config(
    *,
    exp_tag: str,
    seed: int,
    condition: Dict[str, Any],
    ckpt_dir: str,
    stats_root: str,
    batch_size: int | None,
    num_workers: int | None,
) -> str:
    bs = 512 if batch_size is None else int(batch_size)
    workers = 128 if num_workers is None else int(num_workers)
    mmap_dir = str(condition["mmap_dir"])
    stats_dir = f"{stats_root.rstrip('/')}/{condition['slug']}"
    return f"""exp_tag: {exp_tag}

data:
  mmap_dir: {mmap_dir}
  splits_dir: data/exiD/splits
  stats_dir: {stats_dir}
  hz: 3.0
  batch_size: {bs}
  num_workers: {workers}
  scenario_labels: {mmap_dir}/scenario_labels.csv

features:
  ego_mode: pva
  nb_kin_mode: pva
  use_s_x: true
  use_s_y: true
  use_dim: true
  use_I: true

model:
  name: encdecformer
  T: 6
  Tf: 15
  K: 8
  use_neighbors: true
  d_model: 128
  nhead: 4
  enc_layers: 3
  dec_layers: 3
  dropout: 0.1
  M: 6
  return_scores: true

train:
  device: cuda
  seed: {int(seed)}
  epochs: 100
  lr: 3e-4
  weight_decay: 0.0
  use_amp: false
  grad_clip_norm: 1.0
  w_ade: 1.0
  w_fde: 0.0
  w_rmse: 1.0
  w_cls: 0.1
  lr_schedule: cosine
  warmup_steps: 500
  monitor: val_ade
  ckpt_dir: {ckpt_dir}
  stratified_eval: false
"""


def _write_readme(
    out_dir: Path,
    manifest_name: str,
    conditions: List[Dict[str, Any]],
    topn: int,
    slot_alpha: float,
    slot_conditional: bool,
) -> None:
    lines = [
        "# exiD Component Ablation",
        "",
        "Full feature config is fixed (`pva + sx + sy + dim + I`).",
        "The ablation removes preprocessing-time components from the full setup.",
        "",
        f"- full neighbor filtering: `--gate_topn {topn}`",
        f"- full slot weighting: `--slotImportance {slot_alpha:g}`"
        + (" with `--slotImportanceConditional`" if slot_conditional else ""),
        "- `no_nf` is the comparison for not applying neighbor filtering.",
        "- `no_slot` is the comparison for not applying slot weighting.",
        "",
        "Conditions:",
    ]
    for cond in conditions:
        lines.append(
            f"- {cond['order']:02d}_{cond['slug']}: {cond['label']} "
            f"(mmap={cond['mmap_dir']})"
        )
    lines.extend(
        [
            "",
            "Prepare the mmap variants first:",
            "",
            "```bash",
            "TOPN=1 SLOT_ALPHA=1.0 ./scripts/preprocess_exid_component_ablation_data.sh",
            "```",
            "",
            f"See `{manifest_name}` for every generated config, seed, and checkpoint tag.",
            "",
            "Run all experiments with:",
            "",
            "```bash",
            "./scripts/run_exid_component_ablation.sh",
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def generate_configs(
    *,
    out_dir: Path,
    seeds: Iterable[int],
    ckpt_dir: str,
    stats_root: str,
    batch_size: int | None,
    num_workers: int | None,
    base_mmap_dir: str,
    variant_root: str,
    topn: int,
    slot_alpha: float,
    slot_conditional: bool,
    include_conditions: Iterable[str],
) -> List[Dict[str, str | int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_conditions = build_conditions(
        base_mmap_dir=base_mmap_dir,
        variant_root=variant_root,
        topn=topn,
        slot_alpha=slot_alpha,
        slot_conditional=slot_conditional,
    )
    include_set = set(include_conditions)
    known = {cond["slug"] for cond in all_conditions}
    unknown = sorted(include_set - known)
    if unknown:
        raise ValueError(
            f"Unknown condition(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(cond['slug'] for cond in all_conditions)}"
        )
    conditions = [cond for cond in all_conditions if cond["slug"] in include_set]
    if not conditions:
        raise ValueError("No conditions selected.")

    rows: List[Dict[str, str | int]] = []
    for cond in conditions:
        for seed in seeds:
            exp_tag = f"exiD_component_ablation_{cond['order']:02d}_{cond['slug']}_s{seed}"
            cfg_text = _render_config(
                exp_tag=exp_tag,
                seed=int(seed),
                condition=cond,
                ckpt_dir=ckpt_dir,
                stats_root=stats_root,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            cfg_path = out_dir / f"{exp_tag}.yaml"
            cfg_path.write_text(cfg_text, encoding="utf-8")

            rows.append(
                {
                    "condition_order": cond["order"],
                    "condition": cond["slug"],
                    "label": cond["label"],
                    "seed": int(seed),
                    "exp_tag": exp_tag,
                    "config": str(cfg_path),
                    "ckpt": f"{ckpt_dir.rstrip('/')}/{exp_tag}/best.pt",
                    "mmap_dir": cond["mmap_dir"],
                    "stats_dir": f"{stats_root.rstrip('/')}/{cond['slug']}",
                    "gate_topn": cond["gate_topn"],
                    "slot_importance_alpha": f"{float(cond['slot_importance_alpha']):g}",
                    "slot_importance_conditional": _as_bool_text(
                        bool(cond["slot_importance_conditional"])
                    ),
                }
            )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_readme(
        out_dir,
        manifest_path.name,
        conditions,
        topn,
        slot_alpha,
        slot_conditional,
    )
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate configs for the exiD NeighFormer component ablation."
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("configs/ablations/exiD_components"),
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds to generate configs for.",
    )
    ap.add_argument("--ckpt-dir", type=str, default="ckpts")
    ap.add_argument("--stats-root", type=str, default="data/exiD/stats_components")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--base-mmap-dir", type=str, default="data/exiD/dimI")
    ap.add_argument("--variant-root", type=str, default="data/exiD/ablations")
    ap.add_argument(
        "--topn",
        type=int,
        default=1,
        help="Neighbor filtering top-N. Current exiD/highD mmap has at most two active slots per frame, so 1 is the default effective ablation.",
    )
    ap.add_argument("--slot-alpha", type=float, default=1.0)
    ap.add_argument("--slot-conditional", action="store_true", default=False)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITION_SLUGS,
        choices=["full", "no_nf", "no_slot", "no_both"],
        help=(
            "Conditions to generate. Default assumes the full run already exists "
            "and only generates removal runs: no_nf no_slot no_both."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = generate_configs(
        out_dir=args.out_dir,
        seeds=args.seeds,
        ckpt_dir=args.ckpt_dir,
        stats_root=args.stats_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        base_mmap_dir=args.base_mmap_dir,
        variant_root=args.variant_root,
        topn=args.topn,
        slot_alpha=args.slot_alpha,
        slot_conditional=args.slot_conditional,
        include_conditions=args.conditions,
    )
    print(f"[OK] Generated {len(rows)} configs -> {args.out_dir}")
    print(f"[OK] Manifest -> {args.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
