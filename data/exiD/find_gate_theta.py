#!/usr/bin/env python3
"""
find_gate_threshold.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gate_mode='single' 조건에서,
  "sample 당 gate=1이 되는 neighbor 수의 최댓값이 4 이하"가 되도록 하는
  최소 I threshold를 탐색합니다.

분석 대상:
  x_nb  : (N, T, K, 12)  — idx 11 = I (composite importance)
  nb_mask: (N, T, K)     — True if neighbor exists

탐색 방법:
  1. 각 sample(N)에 대해, 전체 히스토리(T)에 걸쳐
     "이웃 슬롯 k에서 단 한 번이라도 I >= theta이면 해당 k를 active로 봄"
     → sample 당 active neighbor 수 = sum_k[max_t(I[n,t,k]) >= theta]
  2. 다양한 theta 후보에서 위 통계를 계산
  3. "P95 active count <= 4" 를 만족하는 최소 theta를 권장값으로 제시

사용법:
  python find_gate_threshold.py --mmap_dir data/highD/mmap

출력:
  - theta vs. active-neighbor 분포 요약 테이블 (콘솔)
  - threshold_analysis.png (시각화)
  - threshold_result.txt   (최종 권장 threshold)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# x_nb 인덱스
IDX_I   = 12   # composite importance I = sqrt((Ix^2 + Iy^2) / 2)

K       = 8    # neighbor slots


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mmap(mmap_dir: Path):
    """mmap 파일 로드. x_nb와 nb_mask만 필요."""
    x_nb_path   = mmap_dir / "x_nb.npy"
    mask_path   = mmap_dir / "nb_mask.npy"

    if not x_nb_path.exists():
        raise FileNotFoundError(f"x_nb.npy not found in {mmap_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(f"nb_mask.npy not found in {mmap_dir}")

    print(f"[Load] x_nb   : {x_nb_path}")
    x_nb    = np.load(str(x_nb_path),  mmap_mode="r")   # (N, T, K, 12)
    print(f"[Load] nb_mask: {mask_path}")
    nb_mask = np.load(str(mask_path),  mmap_mode="r")   # (N, T, K)

    print(f"  x_nb   shape: {x_nb.shape}")
    print(f"  nb_mask shape: {nb_mask.shape}")
    return x_nb, nb_mask


def compute_max_I_per_slot(x_nb: np.ndarray, nb_mask: np.ndarray) -> np.ndarray:
    """
    각 sample × slot에 대해, 전체 히스토리 T에 걸친 최대 I 값을 계산.

    Returns
    -------
    max_I : np.ndarray, shape (N, K)
        존재하지 않는 슬롯(nb_mask 전부 False)은 0.0
    """
    N, T, K_dim = nb_mask.shape
    print(f"[Compute] max I per slot  (N={N}, T={T}, K={K_dim}) ...")

    # I 값 추출: (N, T, K)
    I_vals = x_nb[:, :, :, IDX_I].astype(np.float32)   # mmap → float32

    # 존재하지 않는 timestep은 0으로 마스킹
    I_vals = I_vals * nb_mask.astype(np.float32)        # broadcast (N,T,K)

    # 각 (N, K) 슬롯에서 T 방향 최댓값
    max_I = I_vals.max(axis=1)   # (N, K)
    return max_I


def active_neighbor_count(max_I: np.ndarray, theta: float) -> np.ndarray:
    """
    주어진 theta에서 sample별 active neighbor 수를 반환.
    active = max_I[n, k] >= theta

    Returns
    -------
    counts : np.ndarray, shape (N,)
    """
    return (max_I >= theta).sum(axis=1)   # (N,)


def analyze(max_I: np.ndarray, thetas: np.ndarray, target_max: int = 4):
    """
    각 theta 후보에 대해 active-neighbor 통계를 계산하고,
    "P95 count <= target_max"를 만족하는 최소 theta를 반환.
    """
    results = []
    for theta in thetas:
        counts = active_neighbor_count(max_I, theta)
        results.append({
            "theta":  float(theta),
            "mean":   float(counts.mean()),
            "median": float(np.median(counts)),
            "p75":    float(np.percentile(counts, 75)),
            "p90":    float(np.percentile(counts, 90)),
            "p95":    float(np.percentile(counts, 95)),
            "p99":    float(np.percentile(counts, 99)),
            "max":    int(counts.max()),
            "frac_over_target": float((counts > target_max).mean()),
        })
    return results


def find_recommended_theta(results: list, target_max: int = 4,
                           percentile_key: str = "p95") -> float | None:
    """
    percentile_key (기본 p95) <= target_max 를 만족하는 최소 theta 반환.
    """
    candidates = [r for r in results if r[percentile_key] <= target_max]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["theta"])["theta"]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: list, recommended: float | None,
                 target_max: int, out_path: Path):
    thetas = [r["theta"]  for r in results]
    means  = [r["mean"]   for r in results]
    p75    = [r["p75"]    for r in results]
    p90    = [r["p90"]    for r in results]
    p95    = [r["p95"]    for r in results]
    p99    = [r["p99"]    for r in results]
    maxs   = [r["max"]    for r in results]
    frac   = [r["frac_over_target"] * 100 for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Gate Threshold Analysis  (gate_mode='single')", fontsize=13)

    # ── 상단: active neighbor count 분포 ─────────────────────────────────────
    ax = axes[0]
    ax.plot(thetas, means, label="Mean",   color="steelblue", linewidth=2)
    ax.plot(thetas, p75,   label="P75",    color="orange",    linewidth=1.5, linestyle="--")
    ax.plot(thetas, p90,   label="P90",    color="darkorange",linewidth=1.5, linestyle="-.")
    ax.plot(thetas, p95,   label="P95",    color="red",       linewidth=2,   linestyle="-")
    ax.plot(thetas, p99,   label="P99",    color="darkred",   linewidth=1.5, linestyle=":")
    ax.plot(thetas, maxs,  label="Max",    color="black",     linewidth=1,   linestyle=":", alpha=0.5)
    ax.axhline(target_max, color="green", linewidth=1.5, linestyle="--",
               label=f"Target = {target_max}")
    if recommended is not None:
        ax.axvline(recommended, color="purple", linewidth=2, linestyle="-",
                   label=f"Recommended θ = {recommended:.4f}")
    ax.set_ylabel("# Active Neighbors per Sample")
    ax.set_title("Active Neighbor Count vs. Threshold")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── 하단: threshold 초과 비율 ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(thetas, frac, color="crimson", linewidth=2)
    ax2.axhline(5.0, color="gray", linewidth=1, linestyle="--", label="5% line")
    if recommended is not None:
        ax2.axvline(recommended, color="purple", linewidth=2, linestyle="-",
                    label=f"Recommended θ = {recommended:.4f}")
    ax2.set_ylabel(f"% Samples with count > {target_max}")
    ax2.set_xlabel("Threshold θ  (I value)")
    ax2.set_title(f"Fraction of Samples Exceeding {target_max} Active Neighbors")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"[Plot] saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Find gate threshold s.t. max active neighbors per sample <= N",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mmap_dir",    default="data/exiD/mmap",
                    help="Path to mmap directory (contains x_nb.npy, nb_mask.npy)")
    ap.add_argument("--out_dir",     default=".",
                    help="Output directory for plots and result text")
    ap.add_argument("--target_max",  type=int,   default=4,
                    help="Target max active neighbors per sample")
    ap.add_argument("--theta_min",   type=float, default=0.0,
                    help="Minimum threshold to search")
    ap.add_argument("--theta_max",   type=float, default=1.0,
                    help="Maximum threshold to search")
    ap.add_argument("--theta_steps", type=int,   default=50,
                    help="Number of threshold candidates to evaluate")
    ap.add_argument("--percentile",  default="p95",
                    choices=["p75", "p90", "p95", "p99", "max"],
                    help="Percentile criterion for recommendation")
    args = ap.parse_args()

    mmap_dir = Path(args.mmap_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────
    x_nb, nb_mask = load_mmap(mmap_dir)

    # ── 2. 슬롯별 최대 I 계산 ─────────────────────────────────────────────────
    max_I = compute_max_I_per_slot(x_nb, nb_mask)   # (N, K)

    # 전체 I 분포 요약
    all_vals = max_I[max_I > 0]
    print(f"\n[I distribution (non-zero max-per-slot)]")
    for pct in [0.0, 0.1, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 90, 95, 99]:
        print(f"  P{pct:>2.1f}: {np.percentile(all_vals, pct):.4f}")
    print(f"  Max: {all_vals.max():.4f}")

    # ── 3. threshold 후보 생성 ────────────────────────────────────────────────
    thetas = np.linspace(args.theta_min, args.theta_max, args.theta_steps)

    # ── 4. 분석 ───────────────────────────────────────────────────────────────
    print(f"\n[Analyze] {len(thetas)} theta candidates ...")
    results = analyze(max_I, thetas, target_max=args.target_max)

    # ── 5. 결과 테이블 출력 ───────────────────────────────────────────────────
    header = f"{'theta':>8}  {'mean':>6}  {'P75':>5}  {'P90':>5}  {'P95':>5}  {'P99':>5}  {'max':>4}  {'%>target':>8}"
    print(f"\n{header}")
    print("─" * len(header))
    # 50개 간격으로 출력
    step_print = max(1, len(results) // 50)
    for r in results[::step_print]:
        print(f"  {r['theta']:6.4f}  {r['mean']:6.2f}  {r['p75']:5.2f}  "
              f"{r['p90']:5.2f}  {r['p95']:5.2f}  {r['p99']:5.2f}  "
              f"{r['max']:4d}  {r['frac_over_target']*100:7.1f}%")

    # ── 6. 권장 threshold 탐색 ────────────────────────────────────────────────
    recommended = find_recommended_theta(results, target_max=args.target_max,
                                         percentile_key=args.percentile)

    print(f"\n{'━'*60}")
    print(f"  Criterion : {args.percentile} of active-neighbor count <= {args.target_max}")
    if recommended is not None:
        rec_r = next(r for r in results if r["theta"] == recommended)
        print(f"  Recommended gate_theta = {recommended:.4f}")
        print(f"    mean   = {rec_r['mean']:.3f}")
        print(f"    P75    = {rec_r['p75']:.3f}")
        print(f"    P90    = {rec_r['p90']:.3f}")
        print(f"    P95    = {rec_r['p95']:.3f}")
        print(f"    P99    = {rec_r['p99']:.3f}")
        print(f"    max    = {rec_r['max']}")
        print(f"    % > {args.target_max} = {rec_r['frac_over_target']*100:.2f}%")
    else:
        print(f"  No theta in [{args.theta_min:.4f}, {args.theta_max:.4f}] satisfies the criterion.")
        print(f"  Consider increasing --theta_max.")
    print(f"{'━'*60}")

    # ── 7. 시각화 ─────────────────────────────────────────────────────────────
    plot_path = out_dir / "threshold_analysis.png"
    plot_results(results, recommended, args.target_max, plot_path)

    # ── 8. 결과 저장 ──────────────────────────────────────────────────────────
    result_path = out_dir / "threshold_result.txt"
    with open(result_path, "w") as f:
        f.write(f"gate_mode        : single\n")
        f.write(f"criterion        : {args.percentile} of active-neighbor count <= {args.target_max}\n")
        f.write(f"theta_search     : [{args.theta_min:.4f}, {args.theta_max:.4f}] ({args.theta_steps} steps)\n")
        f.write(f"mmap_dir         : {mmap_dir}\n\n")
        if recommended is not None:
            rec_r = next(r for r in results if r["theta"] == recommended)
            f.write(f"recommended gate_theta = {recommended:.6f}\n\n")
            f.write(f"  mean              = {rec_r['mean']:.4f}\n")
            f.write(f"  P75               = {rec_r['p75']:.4f}\n")
            f.write(f"  P90               = {rec_r['p90']:.4f}\n")
            f.write(f"  P95               = {rec_r['p95']:.4f}\n")
            f.write(f"  P99               = {rec_r['p99']:.4f}\n")
            f.write(f"  max               = {rec_r['max']}\n")
            f.write(f"  frac > {args.target_max}          = {rec_r['frac_over_target']*100:.2f}%\n")
        else:
            f.write("No threshold found. Increase --theta_max.\n")
    print(f"[Result] saved -> {result_path}")


if __name__ == "__main__":
    main()