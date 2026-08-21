#!/usr/bin/env bash
set -euo pipefail

# Offline helper: print checkpoint metadata.
# Usage: ./scripts/inspect_checkpoint.sh checkpoints/demo/step_000002

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_step_dir_or_root>"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import json
from mini_training.checkpoint import resolve_checkpoint_dir, load_metadata, shard_filename

step_dir = resolve_checkpoint_dir("$1")
meta = load_metadata(step_dir)
print(json.dumps({"step_dir": str(step_dir), **meta}, indent=2))
pp = meta["parallel"]["pipeline_parallel_size"]
tp = meta["parallel"]["tensor_parallel_size"]
for p in range(pp):
    for t in range(tp):
        path = step_dir / shard_filename(p, t)
        print(f"shard pp={p} tp={t}: exists={path.is_file()} path={path}")
PY
