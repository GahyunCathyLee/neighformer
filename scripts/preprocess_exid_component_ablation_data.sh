#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-data/exiD}"
RAW_DIR="${RAW_DIR:-raw}"
BASE_MMAP_DIR="${BASE_MMAP_DIR:-dimI}"
VARIANT_ROOT="${VARIANT_ROOT:-ablations}"
TOPN="${TOPN:-1}"
SLOT_ALPHA="${SLOT_ALPHA:-1.0}"
SLOT_CONDITIONAL="${SLOT_CONDITIONAL:-0}"
LIS_MODE="${LIS_MODE:-7}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FORCE="${FORCE:-0}"
CONDITIONS="${CONDITIONS:-base_I neighbor_filter slot_weighting neighbor_filter_slot_weighting}"

slot_suffix() {
  local alpha_slug
  alpha_slug="$("${PYTHON_BIN}" - "${SLOT_ALPHA}" <<'PY'
import sys

text = f"{float(sys.argv[1]):g}"
print(text.replace("-", "m").replace(".", "p"))
PY
)"
  if [[ "${SLOT_CONDITIONAL}" == "1" ]]; then
    printf "slot_a%s_cond" "${alpha_slug}"
  else
    printf "slot_a%s" "${alpha_slug}"
  fi
}

scenario_labels_for() {
  local mmap_dir="$1"
  local out_dir="${DATA_DIR}/${mmap_dir}"
  local base_dir="${DATA_DIR}/${BASE_MMAP_DIR}"

  if [[ -f "${base_dir}/scenario_labels.csv" ]]; then
    if "${PYTHON_BIN}" - "${base_dir}" "${out_dir}" <<'PY'
import sys
from pathlib import Path
import numpy as np

base = Path(sys.argv[1])
out = Path(sys.argv[2])
names = ["meta_recordingId.npy", "meta_trackId.npy", "meta_frame.npy"]
ok = True
for name in names:
    a = np.load(base / name, mmap_mode="r")
    b = np.load(out / name, mmap_mode="r")
    if a.shape != b.shape or not np.array_equal(a, b):
        ok = False
        break
raise SystemExit(0 if ok else 1)
PY
    then
      cp "${base_dir}/scenario_labels.csv" "${out_dir}/scenario_labels.csv"
      echo "  [OK] scenario_labels.csv copied from ${base_dir}"
      return
    fi
  fi

  "${PYTHON_BIN}" data/exiD/scenario_label.py \
    --data_dir "${DATA_DIR}" \
    --raw_dir "${RAW_DIR}" \
    --mmap_dir "${mmap_dir}" \
    --out_csv scenario_labels.csv \
    --target_hz 3.0 \
    --history_sec 2.0 \
    --future_sec 5.0 \
    --stride_sec 1.0 \
    --num_workers "${NUM_WORKERS}"
}

write_metadata() {
  local mmap_dir="$1"
  local gate_topn="$2"
  local slot_alpha="$3"
  local slot_conditional="$4"
  local out_dir="${DATA_DIR}/${mmap_dir}"

  "${PYTHON_BIN}" - \
    "${out_dir}/preprocess_args.json" \
    "${DATA_DIR}" \
    "${RAW_DIR}" \
    "${mmap_dir}" \
    "${gate_topn}" \
    "${slot_alpha}" \
    "${slot_conditional}" \
    "${LIS_MODE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "data_dir": sys.argv[2],
    "raw_dir": sys.argv[3],
    "mmap_dir": sys.argv[4],
    "target_hz": 3.0,
    "history_sec": 2.0,
    "future_sec": 5.0,
    "stride_sec": 1.0,
    "lis_mode": sys.argv[8],
    "gate_topn": int(sys.argv[5]),
    "slot_importance_alpha": float(sys.argv[6]),
    "slot_importance_conditional": sys.argv[7] == "1",
    "drop_vru": True,
    "non_relative": False,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_preprocess() {
  local condition="$1"
  local mmap_dir="$2"
  local gate_topn="$3"
  local slot_alpha="$4"
  local slot_conditional="$5"
  local out_dir="${DATA_DIR}/${mmap_dir}"

  echo
  echo "====== ${condition} -> ${out_dir} ======"

  if [[ "${FORCE}" != "1" && -f "${out_dir}/x_nb.npy" && -f "${out_dir}/nb_mask.npy" ]]; then
    echo "[SKIP] Existing mmap found. Set FORCE=1 to rebuild: ${out_dir}"
    "${PYTHON_BIN}" scripts/align_mmap_to_baseline.py \
      --base-dir "${DATA_DIR}/${BASE_MMAP_DIR}" \
      --target-dir "${out_dir}"
    scenario_labels_for "${mmap_dir}"
    write_metadata "${mmap_dir}" "${gate_topn}" "${slot_alpha}" "${slot_conditional}"
    return
  fi

  args=(
    data/exiD/preprocess.py
    --data_dir "${DATA_DIR}"
    --raw_dir "${RAW_DIR}"
    --mmap_dir "${mmap_dir}"
    --target_hz 3.0
    --history_sec 2.0
    --future_sec 5.0
    --stride_sec 1.0
    --lis_mode "${LIS_MODE}"
    --gate_topn "${gate_topn}"
    --slotImportance "${slot_alpha}"
    --num_workers "${NUM_WORKERS}"
  )

  if [[ "${slot_conditional}" == "1" ]]; then
    args+=(--slotImportanceConditional)
  fi

  "${PYTHON_BIN}" "${args[@]}"
  "${PYTHON_BIN}" scripts/align_mmap_to_baseline.py \
    --base-dir "${DATA_DIR}/${BASE_MMAP_DIR}" \
    --target-dir "${out_dir}"
  scenario_labels_for "${mmap_dir}"
  write_metadata "${mmap_dir}" "${gate_topn}" "${slot_alpha}" "${slot_conditional}"
}

slot_dir="$(slot_suffix)"

for condition in ${CONDITIONS}; do
  case "${condition}" in
    base_I|no_both)
      run_preprocess "${condition}" "${VARIANT_ROOT}/base_I" "0" "0.0" "0"
      ;;
    neighbor_filter)
      run_preprocess "${condition}" "${VARIANT_ROOT}/nf_top${TOPN}" "${TOPN}" "0.0" "0"
      ;;
    slot_weighting)
      run_preprocess "${condition}" "${VARIANT_ROOT}/${slot_dir}" "0" "${SLOT_ALPHA}" "${SLOT_CONDITIONAL}"
      ;;
    neighbor_filter_slot_weighting)
      run_preprocess "${condition}" "${VARIANT_ROOT}/nf_top${TOPN}_${slot_dir}" "${TOPN}" "${SLOT_ALPHA}" "${SLOT_CONDITIONAL}"
      ;;
    *)
      echo "[ERROR] Unknown condition: ${condition}" >&2
      exit 1
      ;;
  esac
done

echo
echo "[DONE] exiD component ablation mmap variants are ready."
