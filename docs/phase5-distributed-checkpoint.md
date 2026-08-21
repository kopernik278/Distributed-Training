# Phase 5: Distributed Checkpoint

## What landed

- RFC: `docs/design/RFC-005-distributed-checkpoint.md`
- `src/mini_training/checkpoint.py`: sharded save / same-layout resume / TP reshard
- TP params tagged with `partition_dim` (Column=`0`, Row weight=`1`)
- Train CLI: `--checkpoint-dir`, `--save-interval`, `--resume`
- Tests: `tests/test_checkpoint.py`

## Layout

```text
checkpoints/step_000005/
  metadata.json
  mp_rank_pp000_tp000.pt
  mp_rank_pp000_tp001.pt   # when tp>1
checkpoints/latest         # points at newest step dir name
```

Only `dp_rank == 0` writes each unique `(pp, tp)` shard.

## How to run

```bash
# train + save
./scripts/run_tp.sh 1 --steps 5 --batch-size 2 --seq-len 32 \
  --hidden-size 64 --num-layers 2 --num-heads 8 --dropout 0.0 \
  --checkpoint-dir checkpoints/demo --save-interval 2

# resume same layout
./scripts/run_tp.sh 1 --steps 3 --batch-size 2 --seq-len 32 \
  --hidden-size 64 --num-layers 2 --num-heads 8 --dropout 0.0 \
  --resume checkpoints/demo

# save with tp=1 then resume/reshard into tp=2
./scripts/run_tp.sh 1 --steps 3 ... --checkpoint-dir checkpoints/tp1
./scripts/run_tp.sh 2 --steps 2 ... --resume checkpoints/tp1
```

On TP topology change, **model weights** are resharded; **optimizer state is re-initialized**.

## Correctness

- Same-layout round-trip: exact parameter equality + optimizer restore
- `tp=1 → tp=2` reshard: loaded shards match `consolidate → split`
- PP topology change is explicitly unsupported in this milestone

## Next

- Profiling / communication-computation overlap
- Optional Adam-state TP reshard + PP consolidate
