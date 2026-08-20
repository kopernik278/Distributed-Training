# Phase 1 Experiment Log

## Purpose

Record reproducible DDP baseline experiments for Project 1 Stage 1/2.

## Hardware note

This environment currently has:

- CUDA available: **No**
- Backend used for smoke benchmarks: **gloo**
- Therefore these numbers are **correctness / instrumentation checks**, not GPU training portfolio claims.

Never present CPU Gloo throughput as NCCL multi-GPU results.

## How to run

```bash
chmod +x scripts/benchmark_ddp_scaling.sh
WORLD_SIZES=1,2 BACKEND=gloo STEPS=6 ./scripts/benchmark_ddp_scaling.sh
```

On a real multi-GPU machine:

```bash
WORLD_SIZES=1,2,4 BACKEND=nccl STEPS=20 \
BATCH_SIZE=8 SEQ_LEN=256 HIDDEN_SIZE=256 NUM_LAYERS=4 NUM_HEADS=8 \
OUT_DIR=results/phase1_gpu ./scripts/benchmark_ddp_scaling.sh
```

## Metrics definitions

- `global_tokens_per_second`: total tokens across ranks / average step time
- `forward_ms`: measured forward + loss construction time
- `backward_ms`: measured backward time (includes DDP gradient sync)
- `optimizer_ms`: measured optimizer step time
- `scaling_efficiency`:
  `tok/s(world_size=N) / (tok/s(world_size=1) * N)`

## Interview talking points

1. Why discard warm-up steps before averaging.
2. Why backward time includes communication in DDP.
3. Why CPU Gloo results can look “too good” or unstable and must be labeled.
4. What additional counters are needed next: MFU, memory, NCCL kernel time, idle gaps.
