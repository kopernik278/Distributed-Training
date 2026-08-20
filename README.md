# Distributed-Training

Mini LLM training infrastructure project for AI Infra / Distributed Training
Engineer interviews.

Long-term context and learning background: [`AI_INFRA_CONTEXT.md`](./AI_INFRA_CONTEXT.md).

## Current milestone

**Project 1 / Phase 1 (Stage 1-2): DDP baseline**

- minimal Transformer language model
- synthetic next-token workload
- PyTorch DDP training loop
- forward / backward / optimizer timing breakdown
- correctness tests
- DDP scaling benchmark script
- RFC + experiment documentation

This intentionally comes before Tensor Parallelism / Pipeline Parallelism /
Checkpoint Resharding. If the baseline training loop and metrics are not
trustworthy, later optimization claims become weak.

## Repository layout

```text
AI_INFRA_CONTEXT.md
src/mini_training/
  config.py
  data.py
  distributed.py
  model.py
  train.py
scripts/
  run_single_node_ddp.sh
  benchmark_ddp_scaling.sh
tests/
  test_phase1_correctness.py
docs/
  design/RFC-001-phase1-ddp-baseline.md
  experiments/phase1-ddp-scaling.md
  phase1-roadmap.md
  phase1-implementation-notes.md
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If `venv` is unavailable:

```bash
python3 -m pip install --break-system-packages torch
python3 -m pip install --break-system-packages -e .
```

## Run

### Single process smoke test

```bash
python3 -m mini_training.train --steps 5 --batch-size 2 --seq-len 64
```

### Single-node DDP

```bash
chmod +x scripts/run_single_node_ddp.sh
NPROC_PER_NODE=2 ./scripts/run_single_node_ddp.sh --steps 10 --batch-size 4 --seq-len 128
```

When CUDA is available, the code uses NCCL. On CPU-only machines it falls back
to Gloo so the training loop can still be validated.

### Correctness tests

```bash
python3 -m unittest discover -s tests -v
```

### Scaling benchmark

```bash
chmod +x scripts/benchmark_ddp_scaling.sh
WORLD_SIZES=1,2 BACKEND=gloo STEPS=6 ./scripts/benchmark_ddp_scaling.sh
```

## Metrics

Each step logs:

- `loss`
- `step_time_ms`
- `forward_ms`
- `backward_ms`
- `optimizer_ms`
- `rank_tokens_per_second`
- `global_tokens_per_second`

Metrics files also include:

- environment metadata (device, backend, commit, PyTorch version)
- summary averages after warm-up discard

## Important rule

Never report fabricated performance numbers. CPU Gloo smoke results are useful
for correctness and instrumentation checks, but are **not** GPU / NCCL training
portfolio claims.

## Next stages (Project 1)

1. Tensor Parallel Linear (`ColumnParallel` / `RowParallel`)
2. Pipeline Parallel + 1F1B
3. Distributed Checkpoint save/resume/reshard
4. Profiling hooks and communication/computation overlap
