#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_DIR="${CONFIG_DIR:-configs/ablations/exiD_minimal}"
LOG_DIR="${LOG_DIR:-logs/ablations/exiD_minimal}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SCENARIO="${EVAL_SCENARIO:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

mkdir -p "${LOG_DIR}"

"${PYTHON_BIN}" scripts/generate_exid_minimal_ablation_configs.py --out-dir "${CONFIG_DIR}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

missing = [
    mod for mod in ("numpy", "torch", "yaml", "tqdm")
    if importlib.util.find_spec(mod) is None
]
if missing:
    print(
        "[ERROR] Missing Python modules: "
        + ", ".join(missing)
        + "\nInstall the training dependencies first, e.g.:\n"
        + "  python3 -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "exiD_ablation_*.yaml" | sort)

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[ERROR] No generated configs found in ${CONFIG_DIR}" >&2
  exit 1
fi

if (( SHARD_COUNT < 1 )); then
  echo "[ERROR] SHARD_COUNT must be >= 1, got ${SHARD_COUNT}" >&2
  exit 1
fi
if (( SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "[ERROR] SHARD_INDEX must be in [0, SHARD_COUNT), got ${SHARD_INDEX}/${SHARD_COUNT}" >&2
  exit 1
fi

SELECTED_CONFIGS=()
total="${#CONFIGS[@]}"
start=$(( total * SHARD_INDEX / SHARD_COUNT ))
end=$(( total * (SHARD_INDEX + 1) / SHARD_COUNT ))
for i in "${!CONFIGS[@]}"; do
  if (( i >= start && i < end )); then
    SELECTED_CONFIGS+=("${CONFIGS[$i]}")
  fi
done

echo "[INFO] Total configs: ${#CONFIGS[@]}"
echo "[INFO] Running shard: ${SHARD_INDEX}/${SHARD_COUNT} (${#SELECTED_CONFIGS[@]} configs, indices ${start}..$((end - 1)))"

if [[ "${#SELECTED_CONFIGS[@]}" -eq 0 ]]; then
  echo "[ERROR] Selected shard is empty" >&2
  exit 1
fi

for cfg in "${SELECTED_CONFIGS[@]}"; do
  tag="$(basename "${cfg%.yaml}")"
  ckpt="ckpts/${tag}/best.pt"

  echo
  echo "====== ${tag} ======"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${ckpt}" ]]; then
    echo "[SKIP] Existing checkpoint: ${ckpt}"
  else
    "${PYTHON_BIN}" train.py --config "${cfg}" 2>&1 | tee "${LOG_DIR}/${tag}_train.log"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ ! -f "${ckpt}" ]]; then
      echo "[WARN] Missing checkpoint, skipping eval: ${ckpt}" >&2
      continue
    fi

    eval_args=(evaluate.py --ckpt "${ckpt}" --split "${EVAL_SPLIT}")
    if [[ "${EVAL_SCENARIO}" == "1" ]]; then
      eval_args+=(--scenario)
    fi
    "${PYTHON_BIN}" "${eval_args[@]}" 2>&1 | tee "${LOG_DIR}/${tag}_${EVAL_SPLIT}_eval.log"
  fi
done

if [[ "${RUN_EVAL}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/summarize_exid_minimal_ablation.py \
    --manifest "${CONFIG_DIR}/manifest.csv" \
    --log-dir "${LOG_DIR}" \
    --split "${EVAL_SPLIT}" \
    --out-dir "${LOG_DIR}"
fi

echo
echo "[DONE] exiD minimal ablation finished."
