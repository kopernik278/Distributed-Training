#!/usr/bin/env bash
set -euo pipefail

# Engineering suite: larger model + WikiText-2 + multi-config GPU benchmarks.
# Records real metrics via --metrics-path; never invents numbers.
#
# Usage (on a multi-GPU RunPod node after bootstrap):
#   ./scripts/benchmark_engineering_suite.sh
#   GPU_COUNT=4 OUT_DIR=results/engineering_suite ./scripts/benchmark_engineering_suite.sh
#
# Env knobs:
#   GPU_COUNT   visible GPUs to use (default: torch.cuda.device_count or 2)
#   STEPS       throughput steps (default 40)
#   PROFILE     if 1, also emit short profiled runs for one TP config

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

OUT_DIR="${OUT_DIR:-results/engineering_suite}"
STEPS="${STEPS:-40}"
PROFILE="${PROFILE:-0}"
DATA_DIR="${DATA_DIR:-data}"
mkdir -p "${OUT_DIR}"

GPU_COUNT="${GPU_COUNT:-}"
if [[ -z "${GPU_COUNT}" ]]; then
  GPU_COUNT="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
fi

# Large-ish model that fits 2×24GB with TP=2 / batch=2.
HIDDEN="${HIDDEN:-1024}"
LAYERS="${LAYERS:-12}"
HEADS="${HEADS:-16}"
SEQ="${SEQ:-512}"
BATCH="${BATCH:-2}"
VOCAB="${VOCAB:-8192}"
MLP_RATIO="${MLP_RATIO:-4}"

COMMON=(
  --dataset wikitext2
  --data-dir "${DATA_DIR}"
  --steps "${STEPS}"
  --warmup-discard 3
  --batch-size "${BATCH}"
  --seq-len "${SEQ}"
  --hidden-size "${HIDDEN}"
  --num-layers "${LAYERS}"
  --num-heads "${HEADS}"
  --mlp-ratio "${MLP_RATIO}"
  --vocab-size "${VOCAB}"
  --dropout 0.0
  --log-interval 5
)

echo "[eng-suite] commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[eng-suite] gpus=${GPU_COUNT} out=${OUT_DIR}"
echo "[eng-suite] model hidden=${HIDDEN} layers=${LAYERS} heads=${HEADS} seq=${SEQ} batch=${BATCH} vocab=${VOCAB}"

run_case() {
  local name="$1"
  shift
  local metrics="${OUT_DIR}/${name}.json"
  local log="${OUT_DIR}/${name}.log"
  echo "[eng-suite] === ${name} ==="
  set +e
  "$@" --metrics-path "${metrics}" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "[eng-suite] FAILED ${name} rc=${rc}" | tee -a "${OUT_DIR}/failures.txt"
    return 0
  fi
  echo "[eng-suite] wrote ${metrics}"
}

# --- Always: single-GPU baseline (dp=1) ---
run_case "gpu1_ddp" \
  ./scripts/run_dp_tp.sh 1 1 "${COMMON[@]}"

if [[ "${GPU_COUNT}" -ge 2 ]]; then
  run_case "gpu2_ddp" \
    ./scripts/run_dp_tp.sh 2 1 "${COMMON[@]}"

  run_case "gpu2_tp" \
    ./scripts/run_tp.sh 2 "${COMMON[@]}"

  run_case "gpu2_tp_overlap" \
    ./scripts/run_tp.sh 2 "${COMMON[@]}" --overlap

  # seq_len % tp == 0 already; enable SP
  run_case "gpu2_tp_sp" \
    ./scripts/run_tp.sh 2 "${COMMON[@]}" --sequence-parallel

  run_case "gpu2_tp_sp_vp" \
    ./scripts/run_tp.sh 2 "${COMMON[@]}" --sequence-parallel --vocab-parallel

  # PP needs layers % pp == 0 (12 % 2 == 0)
  run_case "gpu2_pp" \
    ./scripts/run_pp.sh 2 "${COMMON[@]}" --num-microbatches 4
fi

if [[ "${GPU_COUNT}" -ge 4 ]]; then
  run_case "gpu4_ddp" \
    ./scripts/run_dp_tp.sh 4 1 "${COMMON[@]}"

  run_case "gpu4_dp2_tp2" \
    ./scripts/run_dp_tp.sh 2 2 "${COMMON[@]}"

  run_case "gpu4_dp2_tp2_overlap_sp" \
    ./scripts/run_dp_tp.sh 2 2 "${COMMON[@]}" --overlap --sequence-parallel
fi

if [[ "${PROFILE}" == "1" && "${GPU_COUNT}" -ge 2 ]]; then
  mkdir -p "${OUT_DIR}/profile_tp" "${OUT_DIR}/profile_tp_overlap"
  run_case "gpu2_tp_profiled" \
    ./scripts/run_tp.sh 2 "${COMMON[@]}" --steps 8 \
    --profile-dir "${OUT_DIR}/profile_tp" --profile-wait 1 --profile-warmup 1 --profile-active 3
  run_case "gpu2_tp_overlap_profiled" \
    ./scripts/run_tp.sh 2 "${COMMON[@]}" --steps 8 --overlap \
    --profile-dir "${OUT_DIR}/profile_tp_overlap" --profile-wait 1 --profile-warmup 1 --profile-active 3
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_engineering_suite.py" "${OUT_DIR}"

echo "[eng-suite] done → ${OUT_DIR}/suite_summary.json / suite_report.md"
