# Phase 3: DP × TP (2D Parallel)

## What landed

- RFC: `docs/design/RFC-003-dp-tp-2d-parallel.md`
- Correctness rules for seeding + DP parameter broadcast
- `--data-parallel-size` validation (`world_size == dp × tp`)
- Launcher: `scripts/run_dp_tp.sh <dp> <tp>`
- Tests: `tests/test_dp_tp.py` (4-process Gloo smoke)

## Rank layout

```text
world_size = dp_size × tp_size

TP groups: contiguous ranks
DP groups: same tp_rank across replicas
DDP wraps only the DP process group
```

## Correctness rules

1. Model init seed = `seed + tp_rank`
2. Broadcast parameters from `dp_rank=0` within each DP group
3. Data RNG seed = `seed + offset + dp_rank` (TP peers share batches)
4. Global tokens = local × **dp_size** (never × tp_size)

## How to run

```bash
# dp=2, tp=2  (4 processes)
./scripts/run_dp_tp.sh 2 2 --steps 5 --batch-size 2 --seq-len 32 \
  --hidden-size 64 --num-layers 2 --num-heads 8 --dropout 0.0

# pure TP (dp=1)
./scripts/run_dp_tp.sh 1 2 --steps 5 --batch-size 2 --seq-len 32 --hidden-size 64 --num-heads 8

# pure DP (tp=1)
./scripts/run_dp_tp.sh 2 1 --steps 5 --batch-size 2 --seq-len 32
```

On RunPod with 4 GPUs:

```bash
BACKEND=nccl ./scripts/run_dp_tp.sh 2 2 --steps 20 --batch-size 8 --seq-len 256 \
  --hidden-size 256 --num-layers 4 --num-heads 8 --metrics-path results/dp2_tp2.json
```

## Next

- See Phase 4: `docs/phase4-pipeline-parallel.md` (Pipeline + 1F1B)
- Distributed Checkpoint
- Profiling / overlap
