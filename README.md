# Distributed-Training

Mini LLM training infrastructure project built step by step for AI Infra /
Training Engineer interviews.

## Current milestone

Phase 1 implements a measurable distributed training baseline:

- minimal Transformer language model,
- synthetic next-token training workload,
- PyTorch DDP training loop,
- single-node multi-process launcher,
- structured metrics for throughput and step time.

This project intentionally starts with DDP before adding tensor parallelism,
pipeline parallelism, and distributed checkpointing. The reason is simple: if the
baseline training loop and metrics are not trustworthy, every later scaling or
optimization claim becomes weak.

## Repository layout

```text
src/mini_training/
  config.py         Training configuration
  data.py           Synthetic token workload
  distributed.py    Distributed utilities
  model.py          Minimal Transformer LM
  train.py          DDP training entrypoint
scripts/
  run_single_node_ddp.sh
docs/
  phase1-roadmap.md
  phase1-implementation-notes.md
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If `venv` is unavailable on the machine, use:

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

When CUDA is available, the code uses NCCL. On CPU-only machines, it falls back
to Gloo so the training loop can still be validated.

## Metrics

Each training step logs JSON records with:

- `loss`
- `step_time_ms`
- `rank_tokens_per_second`
- `global_tokens_per_second`

You can also write a full metrics file:

```bash
python3 -m mini_training.train --steps 5 --metrics-path metrics.json
```

## Why this version matters for interviews

This first phase already gives you material to explain:

- how DDP performs gradient synchronization,
- why distributed training needs process-level launch control,
- how to compare 1 GPU vs 2 GPU vs 4 GPU throughput,
- what data you would collect before attempting optimization.

## Planned next steps

1. Add tensor parallel linear layers and communication primitives.
2. Add pipeline stage partitioning and a simple schedule.
3. Add distributed checkpoint save/resume/reshard.
4. Add profiling workflow and scaling experiments.