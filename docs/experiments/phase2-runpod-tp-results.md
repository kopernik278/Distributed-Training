# Phase 2 Measured Results (RunPod / Tensor Parallel)

> Measured on real multi-GPU RunPod hardware.

## Hardware / software

- GPU: 2 x NVIDIA GeForce RTX 3090
- Backend: `nccl`
- PyTorch: `2.8.0+cu128`
- CUDA: `12.8`
- NCCL: `2.27.3`
- Commit: `66eb3f1`
- tensor_parallel_size: `2`
- data_parallel_size: `1`

## Workload

- batch/rank: 8
- seq_len: 256
- hidden: 256
- layers: 4
- heads: 8
- steps: 20 (discard first 2)

## Summary

| metric | value |
|---|---:|
| avg loss | 8.3646 |
| avg step time (ms) | 15.201 |
| avg forward (ms) | 6.540 |
| avg backward (ms) | 7.807 |
| avg optimizer (ms) | 0.720 |
| avg global tok/s | 134785 |

## Notes

- End-to-end CUDA/NCCL TP path validated.
- Absolute tokens/s on this tiny model should not be over-interpreted versus DDP.
- Numerical correctness was validated separately by multiprocess dense-vs-sharded MLP tests.
