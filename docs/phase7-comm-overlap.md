# Phase 7: Communication / Computation Overlap

## What landed

- RFC: `docs/design/RFC-007-comm-compute-overlap.md`
- `src/mini_training/overlap.py`: async TP AllReduce (+ CUDA comm stream)
- ColumnParallel backward fuses `AllReduce(dX)` with `dW` GEMM
- DDP: `gradient_as_bucket_view=True`, configurable `--ddp-bucket-cap-mb`
- CLI: `--overlap`
- Test: overlap path matches sync TP grads (`test_overlap_matches_sync_tp2`)

## Mental model

```text
Without overlap:   [GEMM dX] [AllReduce dX] [GEMM dW]
With overlap:      [GEMM dX] [AllReduce dX ||||| GEMM dW]
```

`dW` does not need the reduced `dX`, so those two can run together.

## How to run

```bash
# correctness still holds with overlap
BACKEND=gloo ./scripts/run_tp.sh 2 --steps 3 --hidden-size 64 --num-heads 8 --overlap

# DDP bucket overlap (dp>1)
NPROC_PER_NODE=2 ./scripts/run_single_node_ddp.sh --steps 5 --overlap --ddp-bucket-cap-mb 10
```

On tiny CPU models you will **not** see a speedup. Overlap is for GPU + large GEMMs.
Use Phase 6 traces on RunPod to *look* for NCCL under `aten::mm`.

## Next

- Optional RunPod overlap vs no-overlap measurement on a larger hidden size
- See Phase 8 for delayed RowParallel forward wait: `docs/phase8-delayed-rowparallel-reduce.md`
