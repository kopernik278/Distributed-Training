#!/usr/bin/env bash
set -euo pipefail

# Phase-1 DDP scaling benchmark.
# Records structured JSON results and never invents GPU/NCCL numbers.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

WORLD_SIZES="${WORLD_SIZES:-1,2}"
STEPS="${STEPS:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
BACKEND="${BACKEND:-gloo}"
OUT_DIR="${OUT_DIR:-results/phase1}"
WARMUP_DISCARD="${WARMUP_DISCARD:-1}"

export OUT_DIR
mkdir -p "${OUT_DIR}"

IFS=',' read -r -a WORLD_SIZE_ARRAY <<< "${WORLD_SIZES}"

for world_size in "${WORLD_SIZE_ARRAY[@]}"; do
  metrics_path="${OUT_DIR}/ws${world_size}.json"
  echo "[benchmark] world_size=${world_size} backend=${BACKEND} -> ${metrics_path}"

  if [[ "${world_size}" -eq 1 ]]; then
    python3 -m mini_training.train \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --seq-len "${SEQ_LEN}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --num-layers "${NUM_LAYERS}" \
      --num-heads "${NUM_HEADS}" \
      --backend "${BACKEND}" \
      --warmup-discard "${WARMUP_DISCARD}" \
      --metrics-path "${metrics_path}"
  else
    python3 -m torch.distributed.run \
      --nnodes 1 \
      --nproc_per_node "${world_size}" \
      -m mini_training.train \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --seq-len "${SEQ_LEN}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --num-layers "${NUM_LAYERS}" \
      --num-heads "${NUM_HEADS}" \
      --backend "${BACKEND}" \
      --warmup-discard "${WARMUP_DISCARD}" \
      --metrics-path "${metrics_path}"
  fi
done

python3 - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ.get("OUT_DIR", "results/phase1"))
rows = []
for path in sorted(out_dir.glob("ws*.json")):
    payload = json.loads(path.read_text())
    summary = payload.get("summary", {})
    env = payload.get("environment", {})
    rows.append(
        {
            "world_size": payload.get("world_size"),
            "backend": payload.get("backend"),
            "device": env.get("device"),
            "cuda_available": env.get("cuda_available"),
            "avg_global_tokens_per_second": summary.get("avg_global_tokens_per_second"),
            "avg_step_time_ms": summary.get("avg_step_time_ms"),
            "avg_forward_ms": summary.get("avg_forward_ms"),
            "avg_backward_ms": summary.get("avg_backward_ms"),
            "avg_optimizer_ms": summary.get("avg_optimizer_ms"),
            "metrics_file": str(path),
        }
    )

if not rows:
    raise SystemExit("No benchmark result files found")

baseline = next((row for row in rows if row["world_size"] == 1), None)
for row in rows:
    if baseline and baseline["avg_global_tokens_per_second"]:
        row["scaling_efficiency"] = (
            row["avg_global_tokens_per_second"]
            / (baseline["avg_global_tokens_per_second"] * row["world_size"])
        )
    else:
        row["scaling_efficiency"] = None

summary_path = out_dir / "scaling_summary.json"
summary_path.write_text(json.dumps({"runs": rows}, indent=2))
print(json.dumps({"event": "scaling_summary", "runs": rows}, indent=2))
print(f"[benchmark] wrote {summary_path}")
PY
