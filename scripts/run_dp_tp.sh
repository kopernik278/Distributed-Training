#!/usr/bin/env bash
set -euo pipefail

# Launch DP × TP (2D parallel) training.
# Usage:
#   ./scripts/run_dp_tp.sh 2 2 --steps 10 --batch-size 4
#   ./scripts/run_dp_tp.sh 1 2 --steps 5   # pure TP
#   ./scripts/run_dp_tp.sh 2 1 --steps 5   # pure DP / DDP

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dp_size> <tp_size> [train args...]"
  exit 1
fi

DP_SIZE="$1"
TP_SIZE="$2"
shift 2

if [[ "${DP_SIZE}" -lt 1 || "${TP_SIZE}" -lt 1 ]]; then
  echo "dp_size and tp_size must be >= 1"
  exit 1
fi

NPROC=$((DP_SIZE * TP_SIZE))

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]] && "${ROOT_DIR}/.venv/bin/python" -c "import torch, mini_training" >/dev/null 2>&1; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

BACKEND="${BACKEND:-nccl}"
if ! "${PYTHON_BIN}" - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  BACKEND="gloo"
fi

echo "[run_dp_tp] dp=${DP_SIZE} tp=${TP_SIZE} nproc=${NPROC} backend=${BACKEND} python=${PYTHON_BIN}"

if [[ "${NPROC}" -eq 1 ]]; then
  "${PYTHON_BIN}" -m mini_training.train \
    --data-parallel-size "${DP_SIZE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --backend "${BACKEND}" \
    "$@"
else
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "${NPROC}" \
    -m mini_training.train \
    --data-parallel-size "${DP_SIZE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --backend "${BACKEND}" \
    "$@"
fi
