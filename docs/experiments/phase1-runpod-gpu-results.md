# Phase 1 Measured Results (RunPod / CUDA / NCCL)

> Status: measured on a real multi-GPU RunPod pod.
> These numbers are valid portfolio baseline measurements.

## Hardware / software

- Pod id: `m12t69icy6jeu8`
- GPU: 2 x NVIDIA GeForce RTX 4090 (24GB)
- Cloud: Community
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- PyTorch: `2.8.0+cu128`
- CUDA: `12.8`
- NCCL: `2.27.3`
- Commit: `ee18ae1`
- Backend: `nccl`

## Workload

- model: MiniTransformerLM (~4.27M params)
- batch size / rank: 8
- seq len: 256
- hidden: 256
- layers: 4
- heads: 8
- steps: 20 (summary discards first 2)

## Results

| world_size | avg step time (ms) | avg forward (ms) | avg backward (ms) | avg optimizer (ms) | avg global tok/s | scaling efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.78 | 3.62 | 7.64 | 1.32 | 166468 | 1.00 |
| 2 | 20.73 | 3.31 | 16.12 | 1.14 | 199364 | 0.60 |

## Observations

1. This is a **real CUDA/NCCL** baseline, unlike earlier CPU/Gloo smoke tests.
2. Moving from 1→2 GPUs increased global tokens/s, but scaling efficiency is ~0.60.
3. Backward time roughly doubles on 2 GPUs, consistent with DDP AllReduce cost living in backward.
4. The current model is tiny; communication overhead dominates more than it would for a large compute-bound LLM.
5. Next optimization/feature work should start from this measured baseline, not from CPU numbers.

## Reproduce

```bash
./scripts/runpod/bootstrap.sh
WORLD_SIZES=1,2 BACKEND=nccl STEPS=20 WARMUP_DISCARD=2 \
BATCH_SIZE=8 SEQ_LEN=256 HIDDEN_SIZE=256 NUM_LAYERS=4 NUM_HEADS=8 \
OUT_DIR=results/phase1_runpod_gpu \
./scripts/benchmark_ddp_scaling.sh
```
