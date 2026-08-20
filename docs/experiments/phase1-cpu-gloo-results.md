# Phase 1 Measured Smoke Results (CPU / Gloo)

> Status: measured on this cloud agent environment.
> Hardware: CPU only (`cuda_available=false`)
> Backend: `gloo`
> These numbers validate the benchmark pipeline. They are **not** GPU/NCCL portfolio claims.

## Config

- steps: 6 (summary discards first 1)
- batch size / rank: 2
- seq len: 32
- hidden size: 64
- layers: 2
- heads: 4
- commit at run time: `fb38fb6` (pre-hardening commit recorded in metrics files)

## Results

| world_size | avg step time (ms) | avg forward (ms) | avg backward (ms) | avg optimizer (ms) | avg global tok/s | scaling efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.61 | 1.41 | 1.91 | 1.23 | 13957 | 1.00 |
| 2 | 7.52 | 2.33 | 3.50 | 1.63 | 17121 | 0.61 |

## Observations worth discussing in interviews

1. **Backward grows more than forward when world_size increases.**
   In DDP, gradient AllReduce is part of backward, so this is expected.
2. **Scaling efficiency < 1 does not automatically mean “bad code”.**
   Tiny models on CPU are often synchronization / launch / GIL sensitive.
3. **Absolute tok/s looks high because the model is tiny and sequence is short.**
   Do not compare these numbers with real 7B GPU training.
4. **Next meaningful data point requires multi-GPU + NCCL.**
   Then we can discuss MFU, NCCL time share, and overlap.

## How to regenerate

```bash
WORLD_SIZES=1,2 BACKEND=gloo STEPS=6 ./scripts/benchmark_ddp_scaling.sh
```
