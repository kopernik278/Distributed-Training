# RFC-001: Phase 1 DDP Training Baseline

## Motivation

Project 1 of this repository is a mini distributed LLM training engine.
Before Tensor Parallelism, Pipeline Parallelism, or Checkpoint Resharding,
we need a trustworthy single-node DDP baseline that can:

- launch correctly on 1..N processes,
- synchronize gradients,
- report reproducible metrics,
- support correctness tests,
- produce scaling experiment artifacts.

Without this baseline, later optimization claims are weak.

## Goals

1. Keep a minimal Transformer LM training loop.
2. Support single-process and multi-process DDP.
3. Measure step time and tokens/sec with synchronized timers.
4. Break step time into forward / backward / optimizer components.
5. Provide unit tests for model shapes and DDP launch smoke tests.
6. Provide a scaling benchmark script that writes structured results.
7. Document limitations clearly when GPU / NCCL hardware is unavailable.

## Non-goals

- Tensor Parallelism
- Pipeline Parallelism
- Distributed Checkpoint
- Real dataset pipelines
- Fabricated GPU / RoCE / InfiniBand numbers

## Architecture

```text
torchrun / scripts/run_single_node_ddp.sh
                │
                ▼
        mini_training.train
                │
   ┌────────────┼────────────┐
   ▼            ▼            ▼
 config      model+DDP     metrics
   │            │            │
   ▼            ▼            ▼
 TrainingConfig  AdamW     JSON logs
 RandomToken     CE loss   experiment summary
```

## Data Flow

1. Each rank initializes process group and local device.
2. Each rank builds a full model replica.
3. Synthetic tokens are generated on-device.
4. Forward computes logits; loss is next-token cross entropy.
5. Backward triggers DDP gradient all-reduce.
6. Optimizer updates identical weights on all ranks.
7. Metrics are reduced and logged by rank 0.

## Communication

- DDP gradient synchronization: AllReduce under the hood.
- Metric aggregation: explicit `all_reduce` for mean/sum.
- Backend selection:
  - CUDA available → requested backend (default NCCL)
  - CPU-only → Gloo fallback

## Memory Behavior

- Full model replica per rank (standard DDP).
- No parameter sharding yet.
- Memory scaling with world size is approximately linear in activations
  (local batch), not in parameters.

## Correctness

- Model output shape: `[B, S, V]`
- Loss is finite and decreases over short warm-up runs under fixed seed.
- Multi-process launch completes without hang/deadlock for smoke configs.
- Metrics file contains required keys and world_size metadata.

## Performance Hypothesis

On real multi-GPU systems with NCCL:

- Global tokens/sec should increase with world size when local batch is fixed.
- Scaling efficiency usually drops as communication and synchronization grow.
- On CPU Gloo smoke tests, absolute throughput is not representative of GPU
  training and must not be used as portfolio performance claims.

## Benchmark Plan

1. Run single-process baseline.
2. Run 2-process DDP smoke benchmark.
3. Record config, backend, device, commit hash, and metrics.
4. Compute average tokens/sec after dropping warm-up steps.
5. Label hardware clearly in the experiment report.

## Risks

- CPU-only environments can produce misleading “speedups”.
- Tiny models can be launch-bound or CPU-bound, hiding communication costs.
- Without warm-up discard, first-step overhead contaminates averages.

## Future Work

- Stage 3: Tensor Parallel Linear layers
- Stage 4/5: Pipeline Parallel + 1F1B
- Stage 6: Distributed Checkpoint
- Stage 7/8: Profiling hooks and overlap experiments
