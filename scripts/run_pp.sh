#!/usr/bin/env bash
set -euo pipefail

# Launch pipeline-parallel training (1F1B).
# Usage:
#   ./scripts/run_pp.sh 2 --steps 10 --num-layers 4 --num-microbatches 4
#   ./scripts/run_pp.sh 2 --tensor-parallel-size 1 ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pp_size> [train args...]"
  exit 1
fi

PP_SIZE="$1"
shift

if [[ "${PP_SIZE}" -lt 1 ]]; then
  echo "pp_size must be >= 1"
  exit 1
fi

# Optional TP is taken from --tensor-parallel-size in remaining args; default 1.
TP_SIZE=1
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tensor-parallel-size)
      TP_SIZE="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --tensor-parallel-size=*)
      TP_SIZE="${1#*=}"
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

NPROC=$((PP_SIZE * TP_SIZE))

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

echo "[run_pp] pp=${PP_SIZE} tp=${TP_SIZE} nproc=${NPROC} backend=${BACKEND} python=${PYTHON_BIN}"

if [[ "${NPROC}" -eq 1 ]]; then
  "${PYTHON_BIN}" -m mini_training.train \
    --pipeline-parallel-size "${PP_SIZE}" \
    --backend "${BACKEND}" \
    "${ARGS[@]}"
else
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "${NPROC}" \
    -m mini_training.train \
    --pipeline-parallel-size "${PP_SIZE}" \
    --backend "${BACKEND}" \
    "${ARGS[@]}"
fi
