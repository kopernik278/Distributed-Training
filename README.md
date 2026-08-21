# Distributed-Training

Mini LLM training infrastructure project for AI Infra / Distributed Training
Engineer interviews.

Long-term learning context: [`AI_INFRA_CONTEXT.md`](./AI_INFRA_CONTEXT.md)

## Development mode (canonical)

```text
Cursor  +  GitHub  +  RunPod
```

- **Cursor**: design, implement, test scaffolding, docs/RFC
- **GitHub**: source of truth and reproducible commit history
- **RunPod**: real GPU / multi-GPU / NCCL execution and profiling

## Current milestone

**Project 1 / Phase 5: Distributed Checkpoint (save / resume / TP reshard)**

Completed:

- Phase 1 DDP baseline + RunPod NCCL measurements
- Phase 2 Tensor Parallel
- Phase 3 DP×TP
- Phase 4 Pipeline Parallel + 1F1B (+ numeric + RunPod)
- Phase 5 Sharded checkpoint, same-layout resume, TP reshard

## Repository layout

```text
src/mini_training/
  parallel_state.py / mappings.py / layers.py
  model.py / pipeline.py
  checkpoint.py
  train.py
scripts/
  run_tp.sh / run_dp_tp.sh / run_pp.sh
  inspect_checkpoint.sh
docs/
  design/RFC-001 … RFC-005
  phase2…phase5-*.md
  experiments/
```

## Install

```bash
python3 -m pip install -e .
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Run

### Phase 1–4 training

```bash
NPROC_PER_NODE=2 ./scripts/run_single_node_ddp.sh --steps 10
./scripts/run_tp.sh 2 --steps 5 --hidden-size 64 --num-heads 8
./scripts/run_dp_tp.sh 2 2 --steps 5 --hidden-size 64 --num-heads 8 --dropout 0.0
./scripts/run_pp.sh 2 --steps 3 --num-layers 4 --num-microbatches 4 --dropout 0.0
```

### Phase 5 Checkpoint

```bash
./scripts/run_tp.sh 1 --steps 5 --checkpoint-dir checkpoints/demo --save-interval 2 ...
./scripts/run_tp.sh 1 --steps 3 --resume checkpoints/demo ...
./scripts/run_tp.sh 2 --steps 2 --resume checkpoints/demo ...   # TP reshard
./scripts/inspect_checkpoint.sh checkpoints/demo
```

Measured PP results: [`docs/experiments/phase4-runpod-pp-results.md`](./docs/experiments/phase4-runpod-pp-results.md)

## Important rule

Never report fabricated performance numbers. CPU/Gloo is for correctness; RunPod CUDA/NCCL is for portfolio measurements.

## Next stages

1. Profiling hooks and communication/computation overlap
2. Optional Adam-state TP reshard / PP consolidate
3. Optional larger-model multi-GPU scaling studies
