# Phase 2: Tensor Parallelism

## What landed

- RFC: `docs/design/RFC-002-tensor-parallel-linear.md`
- `parallel_state.py`: TP/DP process groups
- `mappings.py`: copy / reduce / scatter / gather autograd collectives
- `layers.py`: `ColumnParallelLinear`, `RowParallelLinear`
- model attention + MLP rewritten onto TP linears
- launch: `scripts/run_tp.sh`
- tests: `tests/test_tensor_parallel.py` (includes 2-process numerical check)

## Why this design

Megatron-style MLP:

```text
x -> ColumnParallel (no gather) -> GELU -> RowParallel (+ AllReduce) -> y
```

Attention:

```text
x -> ColumnParallel(QKV) -> local heads -> RowParallel(out) -> y
```

This moves communication from DDP gradient AllReduce into **intra-layer activation collectives**.

## How to run

```bash
# single process (tp=1)
./scripts/run_tp.sh 1 --steps 5 --batch-size 2 --seq-len 64

# 2-way tensor parallel
./scripts/run_tp.sh 2 --steps 5 --batch-size 2 --seq-len 64 --hidden-size 64 --num-heads 8
```

On RunPod GPU:

```bash
BACKEND=nccl ./scripts/run_tp.sh 2 --steps 20 --batch-size 8 --seq-len 256 \
  --hidden-size 256 --num-layers 4 --num-heads 8 --metrics-path results/tp2.json
```

## Correctness focus

The most important TP bug class is silent wrong gradients from incorrect collective autograd.
That is why we added a multiprocess test comparing sharded Column+Row MLP against a dense reference.

## Next

- See Phase 3: `docs/phase3-dp-tp.md` (DP×TP)
- Sequence Parallel
- Vocab Parallel LM head
