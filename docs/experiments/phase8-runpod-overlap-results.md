# Phase 8 Measured Results (RunPod / TP overlap vs sync)

> Measured on real multi-GPU RunPod hardware. Numbers are not fabricated.

## Hardware / software

- GPU: 2 x NVIDIA GeForce RTX 4090 (24GB)
- Backend: `nccl`
- PyTorch: `2.8.0+cu128`
- CUDA: `12.8`
- NCCL: `2.27.3`
- Commit: `84cb87f`
- `tensor_parallel_size`: `2`
- `data_parallel_size`: `1`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

## How to reproduce

```bash
./scripts/runpod/bootstrap.sh
./scripts/runpod/check_gpu_env.sh
OUT_DIR=results/phase8_overlap TP_SIZE=2 HIDDEN=1024 LAYERS=8 HEADS=16 \
  BATCH=4 SEQ=512 STEPS=30 PROFILE_STEPS=6 \
  ./scripts/benchmark_tp_overlap.sh
```

Throughput runs do **not** enable `--profile-dir`. Profiled runs are qualitative only.

## Primary workload (hidden=1024)

| | sync | `--overlap` |
|--|--:|--:|
| batch / seq / layers / heads | 4 / 512 / 8 / 16 | same |
| steps (after warmup discard 2) | 28 | 28 |
| avg loss | 8.4388 | 8.4388 |
| avg step time (ms) | 62.41 | 59.66 |
| avg forward (ms) | 25.27 | 24.16 |
| avg backward (ms) | 30.89 | 29.12 |
| avg global tok/s | 33287 | 34688 |
| tok/s speedup (overlap / sync) | — | **1.042× (~+4.2%)** |

Raw JSON: `phase8_tp2_sync_summary.json`, `phase8_tp2_overlap_summary.json`,
`phase8_overlap_comparison.json`.

## Secondary workload (hidden=2048, smaller batch)

Same GPUs/commit; `batch=2`, `seq=512`, `layers=8`, `heads=16`, 30 steps.

| | sync | `--overlap` |
|--|--:|--:|
| avg global tok/s | 12658 | 12293 |
| tok/s speedup | — | **0.971× (~−2.9%)** |

Raw JSON: `phase8_h2048_tp2_sync_summary.json`, `phase8_h2048_tp2_overlap_summary.json`,
`phase8_overlap_h2048_comparison.json`.

This point is within run-to-run noise / stream-sync overhead territory: smaller
microbatch means less GEMM work to hide behind NCCL, so `--overlap` is not a
guaranteed win.

## Profiler (qualitative)

Short `--profile-dir` runs (6 steps) wrote Chrome traces on the pod. Rank-0
tables: `phase8_profile_sync_rank0_summary.txt`,
`phase8_profile_overlap_rank0_summary.txt`.

Both traces contain `tp_all_reduce` and `aten::mm`. **Do not** publish tokens/s
from profiled steps (profiler overhead).

## Notes

1. Identical avg loss on the primary pair is a useful sanity check that overlap
   did not change numerics on this GPU path.
2. Primary +4.2% is a real but modest gain on this model size; expect larger
   wins only when TP collectives are a bigger fraction of step time and there
   is enough independent GEMM to hide them.
3. Chrome traces were not committed (multi‑MB). Re-run with `PROFILE_STEPS` if
   you need to inspect NCCL vs `aten::mm` timelines locally.
