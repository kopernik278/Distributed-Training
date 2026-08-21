# RFC-008: Delayed RowParallel Forward AllReduce Wait

## Motivation

Phase 7 overlapped ColumnParallel **backward** `AllReduce(dX)` with `dW` GEMM.
RowParallel **forward** still waited on its output AllReduce immediately inside the
mapping helper — so NCCL could not run under the next consumer's setup work.

Megatron exposes the same idea with async AllReduce handles and
`skip_bias_add`: launch reduce, let the caller insert independent work, wait
only when the reduced activation is consumed (bias + residual).

## Design

### Delayed reduce

When `--overlap` is on and `tp_size > 1`:

1. `_ReduceFromModelParallelRegion.forward` clones, launches async TP AllReduce
   (CUDA: dedicated comm stream; CPU/Gloo: `async_op=True`), registers a pending
   `(work, tensor)`, and returns **without** waiting.
2. `flush_pending_tp_reduces()` waits every pending handle and syncs the comm
   stream into the current stream.
3. Backward of reduce remains identity (unchanged).

`copy_to_TP` backward still reduces **synchronously** — delaying input-grad
AllReduce without an immediate consumer flush would corrupt the previous layer.

### `skip_bias_add` on RowParallel

Bias must be added **after** the reduced sum (same on every TP rank). When the
caller passes `skip_bias_add=True`:

- RowParallel returns the (possibly still-reducing) activation and does not add bias.
- The Transformer block flushes, then applies `out + bias` and the residual add.

That is the flush point: the earliest correct consumer of the reduced tensor.

### Overlap window

```text
RowParallel local GEMM  →  async AllReduce (comm stream)
        ╰── caller may issue independent default-stream work ──╯
flush → (+ bias) → residual add → next LayerNorm / ColumnParallel
```

In a standard Pre-LN block there is little independent work between attention
AllReduce and residual add; the hooks still match Megatron and matter once the
caller (or a future sequence-parallel path) inserts real compute before flush.
CUDA comm-stream overlap remains visible in Phase 6 traces when GEMMs are large.

## Non-goals

- Changing Pre-LN residual math (no parallel attn‖mlp reordering)
- Sequence-parallel reduce-scatter / AllGather GEMM overlap
- Claiming CPU/Gloo tokens/s speedup

## Correctness

`--overlap` on vs off must match fp32 forward activations and gradients
(tight tolerance), including full Transformer blocks with `skip_bias_add`.

## Flag

Same `--overlap` as Phase 7 (opt-in, default off).
