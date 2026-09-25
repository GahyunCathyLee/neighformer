#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gahyun/miniconda3/envs/tf/bin/python}"
LOG_DIR="${LOG_DIR:-logs/highd_exid_0_2}"
mkdir -p "$LOG_DIR"

configs=(
  configs/highD0-1.yaml
  configs/highD0-2.yaml
  configs/highD0-3.yaml
  configs/highD0-4.yaml
  configs/highD0-5.yaml
  configs/exiD0-1.yaml
  configs/exiD0-2.yaml
  configs/exiD0-3.yaml
  configs/exiD0-4.yaml
  configs/exiD0-5.yaml
  configs/highD2-1.yaml
  configs/highD2-2.yaml
  configs/highD2-3.yaml
  configs/highD2-4.yaml
  configs/highD2-5.yaml
  configs/exiD2-1.yaml
  configs/exiD2-2.yaml
  configs/exiD2-3.yaml
  configs/exiD2-4.yaml
  configs/exiD2-5.yaml
)

for cfg in "${configs[@]}"; do
  exp_tag="$("$PYTHON_BIN" - "$cfg" <<'PY'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(cfg["exp_tag"])
PY
)"

  ckpt_path="ckpts/${exp_tag}/best.pt"
  log_path="${LOG_DIR}/${exp_tag}.log"

  if [[ -f "$ckpt_path" ]]; then
    echo "[SKIP] ${exp_tag}: ${ckpt_path} exists" | tee -a "${LOG_DIR}/run.log"
    continue
  fi

  echo "[RUN] ${exp_tag}: ${cfg}" | tee -a "${LOG_DIR}/run.log"
  "$PYTHON_BIN" train.py --config "$cfg" 2>&1 | tee "$log_path"
done
