#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_DIR="${CONFIG_DIR:-configs/ablations/exiD_components}"
LOG_DIR="${LOG_DIR:-logs/ablations/exiD_components}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SCENARIO="${EVAL_SCENARIO:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
RUN_PREPROCESS="${RUN_PREPROCESS:-0}"
TOPN="${TOPN:-1}"
SLOT_ALPHA="${SLOT_ALPHA:-1.0}"
SLOT_CONDITIONAL="${SLOT_CONDITIONAL:-0}"
CONDITIONS="${CONDITIONS:-no_nf no_slot no_both}"

mkdir -p "${LOG_DIR}"

generator_args=(
  scripts/generate_exid_component_ablation_configs.py
  --out-dir "${CONFIG_DIR}"
  --topn "${TOPN}"
  --slot-alpha "${SLOT_ALPHA}"
)
read -r -a condition_args <<< "${CONDITIONS}"
generator_args+=(--conditions "${condition_args[@]}")
if [[ "${SLOT_CONDITIONAL}" == "1" ]]; then
  generator_args+=(--slot-conditional)
fi

"${PYTHON_BIN}" "${generator_args[@]}"

if [[ "${RUN_PREPROCESS}" == "1" ]]; then
  TOPN="${TOPN}" SLOT_ALPHA="${SLOT_ALPHA}" SLOT_CONDITIONAL="${SLOT_CONDITIONAL}" \
    "${BASH}" scripts/preprocess_exid_component_ablation_data.sh
fi

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

"${PYTHON_BIN}" - "${CONFIG_DIR}/manifest.csv" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
missing = []
with manifest.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        mmap_dir = Path(row["mmap_dir"])
        for name in ("x_ego.npy", "x_nb.npy", "nb_mask.npy", "y.npy"):
            path = mmap_dir / name
            if not path.exists():
                missing.append(str(path))

if missing:
    print("[ERROR] Missing preprocessed mmap files:", file=sys.stderr)
    for path in sorted(set(missing)):
        print(f"  - {path}", file=sys.stderr)
    print(
        "\nPrepare them first with:\n"
        "  TOPN=1 SLOT_ALPHA=1.0 ./scripts/preprocess_exid_component_ablation_data.sh\n"
        "or run this script with RUN_PREPROCESS=1.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "exiD_component_ablation_*.yaml" | sort)

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
echo "[DONE] exiD component ablation finished."
