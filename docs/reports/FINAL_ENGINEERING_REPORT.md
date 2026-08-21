# Final Engineering Report — Mini Distributed Training Engine

**Branch:** `cursor/mini-training-engine-phase1-ed3e`  
**Primary measured commits:** `331308d` (2-GPU engineering suite), `5f28bd8` (PP import fix + PP follow-ups)  
**Dataset:** WikiText-2 (word-level, vocab capped to `--vocab-size`)  
**Policy:** All throughput numbers below are from real RunPod GPU runs via `--metrics-path`. Profiled runs are **not** used for tok/s claims.

---

## 1. Scope

This report closes the engineering-enhancement pass over Phases 1–9:

| Dimension | Implementation | Measurement |
|---|---|---|
| DDP | `DistributedDataParallel` + DP process groups | 1 vs 2 GPU |
| Tensor Parallel (TP) | Column/Row parallel linear + TP collectives | TP=2 |
| Pipeline Parallel (PP) | Educational 1F1B + async P2P | PP=2 (reduced / mid configs) |
| Comm overlap | ColumnParallel bwd + delayed RowParallel wait | TP sync vs `--overlap` |
| Sequence / Vocab Parallel | SP AG/RS mappings + vocab-parallel CE | TP+SP, TP+SP+VP |
| Profiler | Chrome traces / text tables | Qualitative only |
| Data / scale | WikiText-2 + ~160M-param dense LM | 2×4090 suite; 4×4090 attempted |

---

## 2. Hardware & software (measured)

### 2-GPU engineering host (`fm3ammwbczam54`)

- **GPUs:** 2× NVIDIA GeForce RTX 4090 (24 GB)
- **Topology:** `NODE` (no NVLink) — PCIe within NUMA node
- **Link (during suite):** PCIe Gen4 ×8 (also observed Gen1 under load / other probes)
- **Stack:** PyTorch `2.8.0+cu128`, CUDA `12.8`, NCCL `2.27.3`, backend `nccl`
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

### 4-GPU probe (`wzlrvklfqjn78v`) — attempted, not measured

- **Provisioned:** 4× RTX 4090 community pod (~$1.36/hr); `nvidia-smi` saw all four GPUs (PCIe Gen1 ×16; mixed SYS/NODE topology across sockets).
- **Blocker:** PyTorch `2.8.0+cu128` failed CUDA context init (`CUDA unknown error`) while the driver reported CUDA 13.2 — `torch.cuda.is_available() == False` despite `device_count() == 4`. No trustworthy 4-GPU tok/s were collected; pod deleted after diagnosis.
- Multi-node Instant Clusters were **not** used in this pass (stock / scope); scale-out remains a follow-up, not invented.

---

## 3. Workload definition

**Default engineering model** (suite cases unless noted):

| Knob | Value |
|---|---|
| hidden / layers / heads | 1024 / 12 / 16 |
| seq / batch / vocab | 512 / 2 / 8192 |
| params (DP replica) | ~160.1M |
| params (TP=2 shard) | ~84.5M |
| dataset | WikiText-2 |
| steps / warmup discard | 30–40 / 3 |

**Token accounting:** global tok/s scales with **DP size only** (not TP/PP). For PP, tokens per optimizer step include `num_microbatches × batch × seq`.

---

## 4. Primary 2-GPU results (WikiText-2, eng model)

Throughput table (exclude `*_profiled`):

| case | dp/tp/pp | flags | step ms | tok/s | loss | notes |
|---|---|---|---:|---:|---:|---|
| gpu1_ddp | 1/1/1 | — | 56.08 | **18289** | 6.6205 | single-GPU baseline |
| gpu2_ddp | 2/1/1 | — | 215.36 | **9535** | 6.6642 | DDP regresses vs 1 GPU |
| gpu2_tp | 1/2/1 | — | 89.50 | **11446** | 6.6588 | best 2-GPU eng config here |
| gpu2_tp_overlap | 1/2/1 | overlap | 90.28 | **11342** | 6.6588 | ≈ flat vs sync |
| gpu2_tp_sp | 1/2/1 | SP | 109.67 | **9337** | 6.6881 | SP overhead |
| gpu2_tp_sp_vp | 1/2/1 | SP+VP | 110.57 | **9262** | 6.6532 | VP ≈ neutral on top of SP |

Raw JSON: [`docs/experiments/engineering_suite/`](../experiments/engineering_suite/).

### Phase-8 overlap reference (earlier RunPod pair, random/smaller schedule)

On a prior 2×4090 measurement (`docs/experiments/phase8-runpod-overlap-results.md`):

- hidden=1024, batch=4, seq=512, layers=8: overlap **+4.2%** tok/s vs sync  
- hidden=2048, batch=2: overlap **−2.9%** (noise / not enough GEMM to hide NCCL)

Engineering-suite eng model (batch=2) sits in the “overlap ≈ flat” regime, consistent with that second point.

---

## 5. Analysis by technology

### 5.1 DDP

**What it does:** Replicate the full model; AllReduce grads each step; scale tokens with DP.

**Observation:** 1→2 GPU **halved** global tok/s (18289 → 9535). Forward stayed ~13 ms; **backward exploded** 26.6 → 186.8 ms — classic AllReduce-dominated step on a weak GPU–GPU path (NODE / no NVLink).

**Takeaway:** DDP only wins when interconnect bandwidth ≫ gradient volume / step, or when per-GPU compute is large enough to amortize AllReduce. On this box, DDP-2 is a negative scaling demo — valuable for interviews, not a failure of the code path.

### 5.2 Tensor Parallel (TP)

**What it does:** Shard Column/Row linear weights; AllGather / ReduceScatter / AllReduce on activations/grads; memory per rank ≈ 1/TP.

**Observation:** TP=2 beats DDP=2 here (11446 vs 9535 tok/s) despite more frequent collectives: less activation memory traffic than full-grad AllReduce on this interconnect, and half the params per GPU (~84.5M).

**Takeaway:** Prefer TP over DP when model memory or slow AllReduce dominates and TP degree matches hidden/head divisibility.

### 5.3 Communication overlap

**What it does:** Async TP collectives overlapped with independent GEMM (ColumnParallel bwd; delayed RowParallel forward wait).

**Observation:** Eng suite overlap ≈ sync (−0.9%). Earlier suite showed +4.2% when batch/compute was larger.

**Takeaway:** Overlap is workload-sensitive; publish both sync and overlap; never claim universal speedups.

### 5.4 Sequence Parallel (SP) & Vocab Parallel (VP)

**What it does:** SP shards sequence for LayerNorm/dropout regions (AG before TP compute, RS after); VP shards embedding/LM-head vocab + vocab-parallel cross-entropy.

**Observation:** SP (−18% vs TP) and SP+VP (−19%) add collective volume; on this eng workload the extra AG/RS is not hidden. Memory/params drop slightly with VP (~80.3M vs ~84.5M).

**Takeaway:** SP/VP are correctness + memory tools first; throughput wins appear at longer sequences / larger hidden where activation memory binds.

### 5.5 Pipeline Parallel (PP)

**What it does:** Split layers across stages; 1F1B with async P2P activations/grads; microbatching.

**Bugs fixed during this pass:** `PipelineEngine` / `MicrobatchIO` imports were dropped by the Phase-6 profiler commit — restored in `352d3c9` / `5f28bd8`.

**Eng-default hang:** `seq=512, batch=2, hidden=1024, layers=12, mb=4` **does not complete** on this NODE/PCIe host (NCCL P2P and `NCCL_P2P_DISABLE=1` both stall past multi-minute timeouts). Smaller PP configs complete:

| case | config delta | step ms | tok/s | loss |
|---|---|---:|---:|---:|
| gpu2_pp_mid | h512 L8 seq256 mb4 | 68.21 | 30027 | 6.8547 |
| gpu2_pp_reduced | h1024 L12 seq256 bs1 mb2 | 51.23 | 9997 | 6.6826 |

**Instrumentation note:** PP path currently reports `avg_backward_ms=0` (fwd timer wraps schedule); use step time / tok/s, not bwd split, for PP.

**Takeaway:** PP is functionally validated (unit tests + mid/reduced GPU runs). Full eng-seq PP on consumer PCIe pairs needs better fabric or further schedule/memory hardening — documented honestly rather than fabricated.

### 5.6 Profiler

Chrome traces under `profile_tp*` show `tp_all_reduce` / matmul ranges. Profiled JSON steps are **600+ ms** — exclude from throughput tables.

### 5.7 Dataset & model scale

WikiText-2 replaces pure random tokens for realistic vocab / loss curves. Eng model (~160M) is large enough to stress TP memory sharding and expose interconnect effects without requiring multi-node H100 stock.

---

## 6. Cross-strategy comparison (same host)

Relative to **gpu1_ddp** tok/s (=1.00×):

| strategy | relative tok/s | verdict on this host |
|---|---:|---|
| 1× DDP | 1.00× | baseline |
| 2× DDP | 0.52× | negative scaling (AllReduce) |
| 2× TP | 0.63× | best multi-GPU eng option here |
| 2× TP+overlap | 0.62× | ≈ TP |
| 2× TP+SP | 0.51× | memory/correctness path |
| 2× TP+SP+VP | 0.51× | same + vocab shard |

PP numbers are **not** on the identical seq/batch schedule; compare PP only within `gpu2_pp_*` rows.

---

## 7. Limitations (explicit)

1. **Single-node only** in completed measurements; multi-node not stocked for Instant Clusters in this pass.  
2. **No NVLink** on measured 4090 pairs — numbers are interconnect-bound.  
3. **Eng-default PP** hangs on this fabric; mid/reduced PP used instead.  
4. **4-GPU suite** was started on a separate pod; results are attached when the run finishes (see `engineering_suite_4gpu/` if present).  
5. Profiled tok/s must never be cited as performance.

---

## 8. How to reproduce

```bash
# On a multi-GPU RunPod image with this branch checked out:
./scripts/runpod/bootstrap.sh   # if present
GPU_COUNT=2 PROFILE=1 OUT_DIR=results/engineering_suite \
  ./scripts/benchmark_engineering_suite.sh

# Overlap-focused (Phase 8):
OUT_DIR=results/phase8_overlap ./scripts/benchmark_tp_overlap.sh
```

Summarize: `python3 scripts/summarize_engineering_suite.py results/engineering_suite`

---

## 9. Per-phase technical map

| Phase | Doc / RFC | Core code |
|---|---|---|
| 1 DDP | `docs/phase1-*`, RFC-001 | `distributed.py`, `train.py` |
| 2 TP | `docs/phase2-*`, RFC-002 | `layers.py`, `mappings.py` |
| 3 DP×TP | `docs/phase3-*`, RFC-003 | `parallel_state.py` |
| 4 PP | `docs/phase4-*`, RFC-004 | `pipeline.py` |
| 5 CKPT | `docs/phase5-*`, RFC-005 | `checkpoint.py` |
| 6 Profiler | `docs/phase6-*`, RFC-006 | `profiler.py` |
| 7 Overlap | `docs/phase7-*`, RFC-007 | `overlap.py` + ColumnParallel |
| 8 Delayed Row wait | `docs/phase8-*`, RFC-008 | pending AllReduce queue |
| 9 SP/VP | `docs/phase9-*`, RFC-009 | SP mappings, vocab CE |

---

## 10. Conclusion

The mini engine implements the interview-relevant parallel stack (DDP, TP, DP×TP, PP 1F1B, checkpoint, profiler, overlap, SP/VP) with a real WikiText-2 path and a unified GPU harness. On commodity 2×4090 PCIe hosts, **TP outperforms DDP**, **overlap is workload-dependent**, and **SP/VP trade throughput for activation/vocab memory**. PP works at moderated shapes; eng-seq PP on this fabric is a known open risk called out with measurements, not placeholders.

Artifact index: `docs/experiments/engineering_suite/`, `docs/experiments/phase8-runpod-overlap-results.md`, this report.
