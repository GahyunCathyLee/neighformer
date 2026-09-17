#!/usr/bin/env python3
"""Generate exiD minimal feature ablation configs for NeighFormer.

This creates the 8-condition minimal ablation set with 3 random seeds:

1. kinematics only
2. +sx
3. +sy
4. +sx + sy
5. +I
6. +dim
7. +I + dim
8. +sx + sy + I + dim
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CONDITIONS = [
    {
        "order": 1,
        "slug": "kin",
        "label": "kinematics only",
        "use_s_x": False,
        "use_s_y": False,
        "use_I": False,
        "use_dim": False,
    },
    {
        "order": 2,
        "slug": "sx",
        "label": "+sx",
        "use_s_x": True,
        "use_s_y": False,
        "use_I": False,
        "use_dim": False,
    },
    {
        "order": 3,
        "slug": "sy",
        "label": "+sy",
        "use_s_x": False,
        "use_s_y": True,
        "use_I": False,
        "use_dim": False,
    },
    {
        "order": 4,
        "slug": "sx_sy",
        "label": "+sx + sy",
        "use_s_x": True,
        "use_s_y": True,
        "use_I": False,
        "use_dim": False,
    },
    {
        "order": 5,
        "slug": "I",
        "label": "+I",
        "use_s_x": False,
        "use_s_y": False,
        "use_I": True,
        "use_dim": False,
    },
    {
        "order": 6,
        "slug": "dim",
        "label": "+dim",
        "use_s_x": False,
        "use_s_y": False,
        "use_I": False,
        "use_dim": True,
    },
    {
        "order": 7,
        "slug": "I_dim",
        "label": "+I + dim",
        "use_s_x": False,
        "use_s_y": False,
        "use_I": True,
        "use_dim": True,
    },
    {
        "order": 8,
        "slug": "full",
        "label": "+sx + sy + I + dim",
        "use_s_x": True,
        "use_s_y": True,
        "use_I": True,
        "use_dim": True,
    },
]

DEFAULT_SEEDS = [42, 1234, 3407]


def _as_bool_text(value: bool) -> str:
    return "true" if value else "false"


def _render_config(
    *,
    exp_tag: str,
    seed: int,
    condition: Dict[str, Any],
    ckpt_dir: str,
    stats_dir: str,
    batch_size: int | None,
    num_workers: int | None,
) -> str:
    bs = 512 if batch_size is None else int(batch_size)
    workers = 128 if num_workers is None else int(num_workers)
    sx = _as_bool_text(bool(condition["use_s_x"]))
    sy = _as_bool_text(bool(condition["use_s_y"]))
    use_i = _as_bool_text(bool(condition["use_I"]))
    dim = _as_bool_text(bool(condition["use_dim"]))
    return f"""exp_tag: {exp_tag}

data:
  mmap_dir: data/exiD/dimI
  splits_dir: data/exiD/splits
  stats_dir: {stats_dir}
  hz: 3.0
  batch_size: {bs}
  num_workers: {workers}
  scenario_labels: data/exiD/dimI/scenario_labels.csv

features:
  ego_mode: pva
  nb_kin_mode: pva
  use_s_x: {sx}
  use_s_y: {sy}
  use_dim: {dim}
  use_I: {use_i}

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


def _write_readme(out_dir: Path, manifest_name: str) -> None:
    lines = [
        "# exiD Minimal Feature Ablation",
        "",
        "Generated configs for NeighFormer minimal feature ablation.",
        "",
        "Conditions:",
    ]
    for cond in DEFAULT_CONDITIONS:
        flags = ", ".join(
            f"{key}={_as_bool_text(bool(cond[key]))}"
            for key in ("use_s_x", "use_s_y", "use_I", "use_dim")
        )
        lines.append(f"- {cond['order']:02d}_{cond['slug']}: {cond['label']} ({flags})")
    lines.extend(
        [
            "",
            f"See `{manifest_name}` for every generated config, seed, and checkpoint tag.",
            "",
            "Run all experiments with:",
            "",
            "```bash",
            "./scripts/run_exid_minimal_ablation.sh",
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
    stats_dir: str,
    batch_size: int | None,
    num_workers: int | None,
) -> List[Dict[str, str | int]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str | int]] = []
    for cond in DEFAULT_CONDITIONS:
        for seed in seeds:
            exp_tag = f"exiD_ablation_{cond['order']:02d}_{cond['slug']}_s{seed}"
            cfg_text = _render_config(
                exp_tag=exp_tag,
                seed=int(seed),
                condition=cond,
                ckpt_dir=ckpt_dir,
                stats_dir=stats_dir,
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
                    "use_s_x": _as_bool_text(bool(cond["use_s_x"])),
                    "use_s_y": _as_bool_text(bool(cond["use_s_y"])),
                    "use_I": _as_bool_text(bool(cond["use_I"])),
                    "use_dim": _as_bool_text(bool(cond["use_dim"])),
                }
            )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_readme(out_dir, manifest_path.name)
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate configs for the exiD minimal NeighFormer ablation."
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("configs/ablations/exiD_minimal"),
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--ckpt-dir", type=str, default="ckpts")
    ap.add_argument("--stats-dir", type=str, default="data/exiD/stats_ablation")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = generate_configs(
        out_dir=args.out_dir,
        seeds=args.seeds,
        ckpt_dir=args.ckpt_dir,
        stats_dir=args.stats_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[OK] Generated {len(rows)} configs in {args.out_dir}")
    print(f"[OK] Manifest: {args.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
