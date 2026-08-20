# RFC-002: Tensor Parallel Linear Layers

## Motivation

Phase 1 established a trustworthy DDP baseline on real GPUs.
The next capability for a mini Megatron-style engine is **Tensor Parallelism (TP)**:

> shard matrix multiplies inside a layer across GPUs, and reconstruct
> correct activations with collective communication.

This is a core Training Infra skill: understanding when communication moves from
gradient AllReduce (DDP) into activation collectives (AllGather / ReduceScatter /
AllReduce) inside the forward/backward of a single layer.

## Goals

1. Implement `ColumnParallelLinear` and `RowParallelLinear`.
2. Implement TP process-group state (`tp_rank`, `tp_size`, TP group).
3. Apply TP to Transformer MLP and attention projections.
4. Support pure TP launch (`world_size == tensor_parallel_size`).
5. Prove numerical correctness against a single-process reference.
6. Keep Phase 1 DDP path intact when `tensor_parallel_size=1`.

## Non-goals

- Pipeline Parallel / 1F1B
- Sequence Parallel
- Vocab-parallel embedding / cross-entropy
- Full Megatron fused kernels
- Multi-node TP tuning

## Architecture

```text
               input x  [B, S, H]
                     │
         ColumnParallelLinear (split columns)
                     │
              local activation
                     │
                   GELU / Attn
                     │
          RowParallelLinear (split rows)
                     │
                 AllReduce
                     │
               output [B, S, H]
```

### MLP pattern (Megatron-style)

```text
x --ColumnParallel--> GELU --RowParallel(+AllReduce)--> y
```

### Attention pattern

```text
x --ColumnParallel(QKV)--> local heads attention
  --RowParallel(out)--> AllReduce --> y
```

## Communication

| Op | Where used | Forward | Backward |
|----|------------|---------|----------|
| copy-to-TP | ColumnParallel input | identity | AllReduce |
| gather-from-TP | optional Column output | AllGather | split |
| scatter-to-TP | RowParallel input (if needed) | split | AllGather |
| reduce-from-TP | RowParallel output | AllReduce | identity |

## Memory Behavior

- Each TP rank stores `1/tp_size` of sharded weights for parallel linears.
- Embeddings / final norm may remain replicated in this milestone.
- Activation traffic increases with TP collectives.

## Correctness

1. Unit tests for Column/Row linear against a full `nn.Linear`.
2. End-to-end TP model logits close to non-TP reference (same weights, sharded).
3. Training smoke: multi-process TP launch completes with finite loss.

Tolerance target on CPU/GPU fp32:

- max abs error on logits typically `< 1e-4` for tiny models (exact bound in tests).

## Performance Hypothesis

On tiny models, TP may not beat DDP because activation collectives dominate.
TP becomes valuable when layer GEMMs are large enough that sharding compute wins.

Benchmark plan:

1. Keep Phase 1 DDP baseline as reference.
2. Run `tp_size=2` smoke on RunPod when available.
3. Report tokens/s and forward/backward split; do not claim speedup without data.

## Risks

- Incorrect autograd collectives silently train wrong models.
- Head count / hidden size not divisible by `tp_size`.
- Mixing DDP and TP without separate process groups.

## Future Work

- DP × TP 2D parallel
- Sequence Parallel
- Vocab Parallel
- Overlap TP communication with compute
