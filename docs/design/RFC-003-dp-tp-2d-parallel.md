# RFC-003: Data Parallel × Tensor Parallel (2D Parallel)

## Motivation

Phase 2 added Tensor Parallelism. Real training systems almost always combine:

```text
DP  (scale throughput by replicating sharded models on different data)
 ×
TP  (shard large layers inside each replica)
```

This milestone makes the engine support:

```text
world_size = data_parallel_size × tensor_parallel_size
```

## Goals

1. Keep Megatron-style rank layout already introduced in Phase 2.
2. Ensure ranks with the same `tp_rank` across DP replicas start from identical shards.
3. Ensure ranks inside one TP group consume the **same** microbatch.
4. Ensure different DP replicas consume **different** microbatches.
5. Wrap DDP only on the DP process group.
6. Provide launcher + smoke tests for `dp=2,tp=2`.

## Non-goals

- Pipeline Parallel
- ZeRO / FSDP parameter sharding
- Sequence Parallel
- Expert Parallel

## Rank Layout

For `world_size=4`, `tp_size=2` (`dp_size=2`):

```text
global rank:  0    1    2    3
tp_rank:      0    1    0    1
dp_rank:      0    0    1    1

TP groups:  {0,1} and {2,3}
DP groups:  {0,2} (tp0 shards) and {1,3} (tp1 shards)
```

## Data Flow

```text
DP replica 0: batch A
  rank0 (tp0) ─┐
  rank1 (tp1) ─┴─ TP collectives inside layers

DP replica 1: batch B
  rank2 (tp0) ─┐
  rank3 (tp1) ─┴─ TP collectives inside layers

Then DDP AllReduce gradients only within DP groups:
  grad(rank0 shard) ↔ grad(rank2 shard)
  grad(rank1 shard) ↔ grad(rank3 shard)
```

## Correctness Rules

1. Model init seed depends on `tp_rank` (not global rank).
2. After construction, broadcast parameters from `dp_rank=0` within each DP group.
3. Data RNG seed depends on `dp_rank` so TP peers share the same batch.
4. Token accounting:
   - local tokens = batch_size × seq_len
   - global tokens = local tokens × dp_size
   - do **not** multiply by tp_size

## Performance Hypothesis

On tiny models, 2D parallel may not beat pure DDP.
Value here is systems correctness and the communication pattern:

- TP collectives on activations
- DP AllReduce on gradients of matching shards

## Benchmark Plan

1. CPU/Gloo smoke: `dp=2,tp=2` finishes with finite loss.
2. When 4 GPUs are available on RunPod: NCCL `dp=2,tp=2` metrics with metadata.
3. Compare qualitatively against Phase 1 DDP and Phase 2 pure TP.

## Risks

- Seeding bugs that desynchronize TP peers (silent divergence).
- Wrapping DDP on the default world group instead of DP group.
- Inflating tokens/s by multiplying TP size.

## Future Work

- Pipeline Parallel on top of DP×TP
- Sequence Parallel
- Overlap DP/TP communication with compute
