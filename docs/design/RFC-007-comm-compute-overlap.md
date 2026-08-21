# RFC-007: Communication / Computation Overlap

## Motivation

Phase 6 made comm vs compute *visible*. This milestone hides some communication
behind independent compute — the core performance skill after “correct collectives”.

On a GPU timeline, **overlap** means a NCCL AllReduce bar sits under an `aten::mm`
bar instead of after a gap (idle GPU).

## What we overlap (this milestone)

### 1. Tensor Parallel — ColumnParallel backward

```text
dY  ─┬─►  dX_local = dY @ W     ─► async AllReduce(dX)  ─┐
     └─►  dW = X^T @ dY          (GEMM on default stream) ─┴► wait ► dX ready
```

`dW` does **not** need the reduced `dX`, so AllReduce and the weight GEMM are independent.

### 2. Data Parallel — DDP buckets

PyTorch DDP already overlaps gradient AllReduce of *ready* buckets with backward of
*earlier* layers. We expose `--ddp-bucket-cap-mb` (smaller cap ⇒ more buckets ⇒ more
chances to overlap; too small ⇒ launch overhead).

### 3. Pipeline — already async P2P

Phase 4 `isend`/`irecv` is overlap of P2P with the next microbatch. No change required.

## Non-goals

- Sequence-parallel reduce-scatter hiding
- Overlapping RowParallel *forward* AllReduce with the next layer (needs delayed wait)
- Claiming speedup on tiny CPU/Gloo models

## Correctness

Overlap must not change numerics vs the synchronous path (fp32 tests, tight tolerance).

## Flag

`--overlap` enables the fused ColumnParallel backward. Default **on** when `tp_size>1`
would surprise users comparing Phase 2 numbers; default **off**, opt-in.

## Future

- CUDA dedicated comm stream + `wait_stream` for RowParallel forward
- Overlap optimizer with last DDP bucket
