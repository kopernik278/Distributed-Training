# Phase 4: Pipeline Parallel + 1F1B

## What landed

- RFC: `docs/design/RFC-004-pipeline-parallel-1f1b.md`
- Rank layout: `world = dp × pp × tp` (Megatron ordering)
- `PipelineStage`: embed on stage0, LM head on last stage, layers split evenly
- `PipelineEngine`: warmup → steady **1F1B** → cooldown with **async P2P**
- Launcher: `scripts/run_pp.sh <pp_size>`
- Tests: `tests/test_pipeline_parallel.py`

## Why async P2P

In steady 1F1B, stage0 wants to **send** the next activation while stage1 wants to
**send** the previous gradient. Blocking `dist.send`/`recv` deadlocks.
`isend`/`irecv` (with waits) is the educational fix.

## How to run

```bash
# pp=2, 4 microbatches, 4 layers (2 per stage)
./scripts/run_pp.sh 2 --steps 3 --batch-size 2 --seq-len 16 \
  --hidden-size 32 --num-layers 4 --num-heads 4 --mlp-ratio 2 \
  --vocab-size 128 --dropout 0.0 --num-microbatches 4
```

## Correctness focus

- Shared microbatch stream across PP ranks (same data seed within a DP replica)
- Loss computed only on last stage, then broadcast for metrics
- Local grads exist on every stage after one 1F1B step

## Next

- Distributed Checkpoint (save/resume/reshard)
- Profiling / communication-computation overlap
- Optional RunPod PP NCCL metrics when multi-GPU is available
