# RunPod Setup Guide (Step by Step)

This guide is for you (human) to complete in the RunPod console.
Cursor cannot create your RunPod account or pay for GPUs.

Official docs worth bookmarking:

- Pods: https://docs.runpod.io/
- Instant Clusters (multi-node PyTorch): https://docs.runpod.io/instant-clusters/pytorch

---

## Phase A — Account and billing (one-time)

1. Open https://www.runpod.io/ and create an account.
2. Add credits / payment method.
3. Enable SSH key in account settings (recommended):
   - generate local key if needed: `ssh-keygen -t ed25519`
   - paste public key into RunPod SSH Keys.

---

## Phase B — First GPU Pod for this project (recommended start)

Start simple: **1 node, 2 GPUs** (or 1 GPU first if budget-constrained).

### B1. Create Pod

In RunPod Console → **Pods** → **Deploy**:

1. **GPU**: prefer `A100 80GB` or `H100` if available; for early bring-up, `A4000/A5000/L4/4090` also works.
2. **GPU count**: start with `1`, then move to `2`.
3. **Template**: choose **RunPod PyTorch** (CUDA 12 / PyTorch 2.x style template).
4. **Container disk**: >= 40GB.
5. **Volume** (recommended): attach a Network Volume for persistent `checkpoints/`, `results/`, datasets.
6. Deploy and wait until Running.

### B2. Connect

Use either:

- Web terminal in RunPod UI, or
- SSH from your laptop

Verify GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected:

- `torch.cuda.is_available() == True`
- `device_count` matches purchased GPUs

---

## Phase C — Bring this repository onto the Pod

### C1. Clone your GitHub branch

```bash
cd /workspace   # or your volume mount path
git clone https://github.com/kopernik278/Distributed-Training.git
cd Distributed-Training
git fetch origin
git checkout cursor/mini-training-engine-phase1-ed3e
git pull origin cursor/mini-training-engine-phase1-ed3e
```

If the repo is private, use a GitHub PAT or SSH deploy key.

### C2. Bootstrap project env

```bash
chmod +x scripts/runpod/*.sh
./scripts/runpod/bootstrap.sh
```

This installs the package editable and prints environment metadata.

### C3. Hardware / NCCL check

```bash
./scripts/runpod/check_gpu_env.sh
```

If this fails, fix CUDA/PyTorch/GPU visibility before any training claims.

---

## Phase D — First real GPU training runs

### D1. Single GPU smoke

```bash
./scripts/runpod/run_ddp_gpu.sh 1
```

### D2. Single-node 2 GPU DDP (NCCL)

```bash
./scripts/runpod/run_ddp_gpu.sh 2
```

### D3. Scaling benchmark (1 vs 2 GPUs)

```bash
WORLD_SIZES=1,2 BACKEND=nccl STEPS=20 \
OUT_DIR=results/phase1_runpod_gpu \
./scripts/benchmark_ddp_scaling.sh
```

Then inspect:

```bash
cat results/phase1_runpod_gpu/scaling_summary.json
```

Commit experiment notes back to GitHub from Cursor after you download/copy summary metrics.

---

## Phase E — Cost control (important)

1. Stop / terminate Pod when idle.
2. Prefer Network Volume for persistent artifacts so Pods can be ephemeral.
3. Do short smoke tests before long jobs.
4. Record auto-shutdown settings if offered.
5. Never leave multi-GPU pods running overnight without a job.

---

## Phase F — Later: multi-node Instant Clusters

Only after single-node NCCL DDP is stable.

1. Console → **Instant Clusters** → Create Cluster
2. Template: RunPod PyTorch
3. Start with 2 nodes x N GPUs
4. Use `scripts/runpod/run_ddp_multinode.sh`
5. Set `NCCL_SOCKET_IFNAME=ens1` for inter-node traffic (RunPod guidance)

Do **not** jump to multi-node before single-node metrics are trustworthy.

---

## Recommended first-week hardware path

| Day | Setup | Goal |
|-----|-------|------|
| 1 | 1x GPU Pod | CUDA import + single-process train |
| 2 | 2x GPU same node | NCCL DDP correctness + tokens/s |
| 3 | 2x/4x GPU | scaling efficiency table |
| 4+ | keep same node | start Tensor Parallel Stage 3 |

---

## What you do vs what Cursor does

### You do on RunPod

- create/stop pods
- pay/manage credits
- `git pull`
- run GPU scripts
- confirm `nvidia-smi` and NCCL health

### Cursor does

- implement training engine features
- keep docs/scripts updated
- review metrics you paste/commit
- prepare next RFC/code for TP/PP/Checkpoint

---

## Acceptance checklist before calling the platform “ready”

- [ ] Pod has visible GPUs in `nvidia-smi`
- [ ] `torch.cuda.is_available()` is true
- [ ] repo branch checked out and bootstrapped
- [ ] single GPU smoke passes
- [ ] 2 GPU NCCL DDP passes
- [ ] scaling summary JSON generated with commit hash and GPU name
- [ ] Pod stopped after experiment
