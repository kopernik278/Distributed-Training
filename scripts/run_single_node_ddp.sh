#!/usr/bin/env bash
set -euo pipefail

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

python3 -m torch.distributed.run \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  -m mini_training.train "$@"
