#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gahyun/miniconda3/envs/tf/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

WARMUP="${WARMUP:-1000}"
ITERS="${ITERS:-10000}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs/latency}"
mkdir -p "$LOG_DIR"

# Optional data overrides:
#   DATA_ROOT=/path/holding/data_dirs ./scripts/run_latency.sh
#   EXID_MMAP_DIR=/path/to/exiD/dimI EXID_SPLITS_DIR=/path/to/exiD/splits ./scripts/run_latency.sh
# Optional checkpoint overrides:
#   CKPT_ROOT=/path/to/neighformer_ckpts ./scripts/run_latency.sh
#   EXID_BASE_CKPT=/path/to/exiD0-5.pt EXID_I_CKPT=/path/to/exiD2-5.pt ./scripts/run_latency.sh

cases=(
  "exiD-baseline|ckpts/exiD0-5/best.pt"
  "exiD-+I|ckpts/exiD2-5/best.pt"
)

for row in "${cases[@]}"; do
  IFS='|' read -r name ckpt <<< "$row"
  dataset="${name%%-*}"
  condition="${name#*-}"
  log_path="${LOG_DIR}/${name}.log"

  ckpt_key=""
  if [[ "$condition" == "baseline" ]]; then
    ckpt_key="${EXID_BASE_CKPT:-}"
  else
    ckpt_key="${EXID_I_CKPT:-}"
  fi
  if [[ -n "$ckpt_key" ]]; then
    ckpt="$ckpt_key"
  elif [[ -n "${CKPT_ROOT:-}" ]]; then
    ckpt="${CKPT_ROOT}/${ckpt#ckpts/}"
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] ${name}: missing ${ckpt}"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" evaluate.py
    --ckpt "$ckpt"
    --split test
    --measure_time
    --latency_warmup "$WARMUP"
    --latency_iters "$ITERS"
  )

  mmap_dir=""
  splits_dir=""
  mmap_dir="${EXID_MMAP_DIR:-}"
  splits_dir="${EXID_SPLITS_DIR:-}"
  if [[ -z "$mmap_dir" && -n "${DATA_ROOT:-}" ]]; then
    mmap_dir="${DATA_ROOT}/${dataset}/dimI"
  fi
  if [[ -z "$splits_dir" && -n "${DATA_ROOT:-}" ]]; then
    splits_dir="${DATA_ROOT}/${dataset}/splits"
  fi
  if [[ -n "$mmap_dir" ]]; then
    cmd+=(--mmap_dir "$mmap_dir")
  fi
  if [[ -n "$splits_dir" ]]; then
    cmd+=(--splits_dir "$splits_dir")
  fi

  echo "[RUN] ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
