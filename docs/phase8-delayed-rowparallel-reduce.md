# Phase 8: Delayed RowParallel Forward AllReduce Wait

## What landed

- RFC: `docs/design/RFC-008-delayed-rowparallel-reduce.md`
- Pending forward TP AllReduce queue in `overlap.py` (`flush_pending_tp_reduces`)
- `reduce_from_TP` under `--overlap` launches async and does **not** wait
- `RowParallelLinear(..., skip_bias_add=True)` + `finalize_tensor_parallel_output` in
  `TransformerBlock` (bias after flush, then residual)
- ColumnParallel entry flushes pending reduces as a safety net
- Same CLI flag: `--overlap`

## Mental model

```text
RowParallel local GEMM
    → async AllReduce on comm stream   ⎫  overlap window
    → (optional independent compute)   ⎭
flush → + bias → residual → next LN / ColumnParallel
```

## How to run

```bash
BACKEND=gloo ./scripts/run_tp.sh 2 --steps 3 --hidden-size 64 --num-heads 8 --overlap
python3 -m unittest tests.test_tensor_parallel.TensorParallelMultiProcessTests -v
```

## Next

- Sequence / Vocab Parallel: `docs/phase9-sequence-vocab-parallel.md`
- Optional Adam-state TP reshard / PP consolidate
