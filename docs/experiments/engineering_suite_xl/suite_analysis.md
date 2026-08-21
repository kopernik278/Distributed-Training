# Engineering suite analysis

Baseline: `gpu1_ddp` @ 3837 tok/s

## Relative throughput

| case | dp/tp/pp | tok/s | vs baseline | fwd% | bwd% | opt% |
|---|---|---:|---:|---:|---:|---:|
| gpu1_ddp | 1/1/1 | 3837 | 1.00× | 27% | 53% | 19% |
| gpu2_ddp | 2/1/1 | 2029 | 0.53× | 7% | 88% | 5% |
| gpu2_pp | 1/1/2 | 5493 | 1.43× | 85% | 0% | 15% |
| gpu2_tp | 1/2/1 | 4051 | 1.06× | 36% | 52% | 11% |
| gpu2_tp_overlap | 1/2/1 | 4306 | 1.12× | 39% | 48% | 12% |
| gpu2_tp_sp | 1/2/1 | 3786 | 0.99× | 37% | 52% | 11% |
| gpu2_tp_sp_vp | 1/2/1 | 3859 | 1.01× | 37% | 52% | 10% |

## Pairwise deltas

| comparison | A tok/s | B tok/s | B/A | Δ% |
|---|---:|---:|---:|---:|
| DDP scale 1→2 | 3837 | 2029 | 0.529× | -47.1% |
| DDP-2 vs TP-2 | 2029 | 4051 | 1.997× | +99.7% |
| TP overlap vs sync | 4051 | 4306 | 1.063× | +6.3% |
| Sequence Parallel overhead | 4051 | 3786 | 0.935× | -6.5% |
| Vocab Parallel on top of SP | 3786 | 3859 | 1.019× | +1.9% |

Raw: `suite_analysis.json`
