#!/usr/bin/env python3
"""Summarize exiD minimal ablation evaluation logs into CSV tables."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_RE = re.compile(
    r"\|\s*ADE\s*\|\s*FDE\s*\|\s*RMSE\s*\|.*?"
    r"\|\s*([0-9]+(?:\.[0-9]+)?)\s*"
    r"\|\s*([0-9]+(?:\.[0-9]+)?)\s*"
    r"\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|",
    re.DOTALL,
)
N_RE = re.compile(r"n_samples\s*=\s*([0-9,]+)")


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_log(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    metric = METRIC_RE.search(text)
    if metric is None:
        return {"status": "missing_metrics", "ade": "", "fde": "", "rmse": "", "n_samples": ""}

    n_match = N_RE.search(text)
    return {
        "status": "ok",
        "ade": metric.group(1),
        "fde": metric.group(2),
        "rmse": metric.group(3),
        "n_samples": n_match.group(1).replace(",", "") if n_match else "",
    }


def _mean(values: Iterable[float]) -> float:
    return statistics.mean(values)


def _std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(*, manifest: Path, log_dir: Path, split: str, out_dir: Path) -> None:
    rows = _read_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        tag = row["exp_tag"]
        log_path = log_dir / f"{tag}_{split}_eval.log"
        parsed = _parse_log(log_path) if log_path.exists() else {
            "status": "missing_log",
            "ade": "",
            "fde": "",
            "rmse": "",
            "n_samples": "",
        }
        out = {
            "condition_order": row["condition_order"],
            "condition": row["condition"],
            "label": row["label"],
            "seed": row["seed"],
            "exp_tag": tag,
            **parsed,
        }
        per_run.append(out)
        if parsed["status"] == "ok":
            grouped[row["condition"]].append(out)

    per_run_path = out_dir / f"summary_{split}_per_run.csv"
    with per_run_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)

    by_condition = []
    for condition, items in grouped.items():
        first = items[0]
        ade = [float(item["ade"]) for item in items]
        fde = [float(item["fde"]) for item in items]
        rmse = [float(item["rmse"]) for item in items]
        by_condition.append(
            {
                "condition_order": first["condition_order"],
                "condition": condition,
                "label": first["label"],
                "n_runs": len(items),
                "ade_mean": f"{_mean(ade):.6f}",
                "ade_std": f"{_std(ade):.6f}",
                "fde_mean": f"{_mean(fde):.6f}",
                "fde_std": f"{_std(fde):.6f}",
                "rmse_mean": f"{_mean(rmse):.6f}",
                "rmse_std": f"{_std(rmse):.6f}",
            }
        )
    by_condition.sort(key=lambda r: int(r["condition_order"]))

    by_condition_path = out_dir / f"summary_{split}_by_condition.csv"
    fieldnames = [
        "condition_order",
        "condition",
        "label",
        "n_runs",
        "ade_mean",
        "ade_std",
        "fde_mean",
        "fde_std",
        "rmse_mean",
        "rmse_std",
    ]
    with by_condition_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_condition)

    print(f"[OK] Per-run summary: {per_run_path}")
    print(f"[OK] By-condition summary: {by_condition_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize exiD minimal ablation logs.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/ablations/exiD_minimal/manifest.csv"),
    )
    ap.add_argument("--log-dir", type=Path, default=Path("logs/ablations/exiD_minimal"))
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out-dir", type=Path, default=Path("logs/ablations/exiD_minimal"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summarize(
        manifest=args.manifest,
        log_dir=args.log_dir,
        split=args.split,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
