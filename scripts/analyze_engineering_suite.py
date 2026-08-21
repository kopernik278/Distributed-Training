#!/usr/bin/env python3
"""Deeper analysis of engineering-suite JSON metrics.

Produces efficiency ratios, phase breakdowns, and a markdown analysis section.
Never invents numbers — only aggregates files present in ``out_dir``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_cases(out: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(out.glob("*.json")):
        if path.name.startswith("suite_") or path.name.endswith("_analysis.json"):
            continue
        if "profiled" in path.stem:
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        summary = payload.get("summary") or {}
        env = payload.get("environment") or {}
        cfg = payload.get("config") or {}
        if not summary:
            continue
        rows.append(
            {
                "name": path.stem,
                "tok_s": float(summary.get("avg_global_tokens_per_second") or 0.0),
                "step_ms": float(summary.get("avg_step_time_ms") or 0.0),
                "fwd_ms": float(summary.get("avg_forward_ms") or 0.0),
                "bwd_ms": float(summary.get("avg_backward_ms") or 0.0),
                "opt_ms": float(summary.get("avg_optimizer_ms") or 0.0),
                "loss": float(summary.get("avg_loss") or 0.0),
                "params": int(payload.get("parameter_count") or 0),
                "dp": cfg.get("data_parallel_size") or env.get("data_parallel_size") or 1,
                "tp": cfg.get("tensor_parallel_size") or 1,
                "pp": cfg.get("pipeline_parallel_size") or 1,
                "overlap": bool(cfg.get("overlap")),
                "sp": bool(cfg.get("sequence_parallel")),
                "vp": bool(cfg.get("vocab_parallel")),
                "hidden": cfg.get("hidden_size"),
                "layers": cfg.get("num_layers"),
                "seq": cfg.get("seq_len"),
                "batch": cfg.get("batch_size"),
                "dataset": cfg.get("dataset"),
                "gpu_name": env.get("gpu_name"),
                "commit": env.get("commit"),
            }
        )
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <out_dir>")
    out = Path(sys.argv[1])
    rows = _load_cases(out)
    if not rows:
        raise SystemExit(f"no throughput cases in {out}")

    by_name = {r["name"]: r for r in rows}
    baseline = by_name.get("gpu1_ddp") or rows[0]
    base_tok = baseline["tok_s"] or 1.0

    analysis_rows = []
    for r in rows:
        step = r["step_ms"] or 1.0
        analysis_rows.append(
            {
                **r,
                "rel_tok_s": r["tok_s"] / base_tok,
                "fwd_frac": r["fwd_ms"] / step,
                "bwd_frac": r["bwd_ms"] / step,
                "opt_frac": r["opt_ms"] / step,
                "tok_per_param_s": (r["tok_s"] / r["params"]) if r["params"] else None,
            }
        )

    # Pairwise comparisons of interest when both sides exist.
    pairs = []
    wanted = [
        ("gpu1_ddp", "gpu2_ddp", "DDP scale 1→2"),
        ("gpu1_ddp", "gpu4_ddp", "DDP scale 1→4"),
        ("gpu2_ddp", "gpu2_tp", "DDP-2 vs TP-2"),
        ("gpu2_tp", "gpu2_tp_overlap", "TP overlap vs sync"),
        ("gpu2_tp", "gpu2_tp_sp", "Sequence Parallel overhead"),
        ("gpu2_tp_sp", "gpu2_tp_sp_vp", "Vocab Parallel on top of SP"),
        ("gpu4_ddp", "gpu4_dp2_tp2", "4-GPU DDP vs DP2×TP2"),
        ("gpu4_dp2_tp2", "gpu4_dp2_tp2_overlap_sp", "4-GPU + overlap + SP"),
    ]
    for a, b, label in wanted:
        if a in by_name and b in by_name:
            ta, tb = by_name[a]["tok_s"], by_name[b]["tok_s"]
            pairs.append(
                {
                    "label": label,
                    "a": a,
                    "b": b,
                    "tok_a": ta,
                    "tok_b": tb,
                    "speedup_b_over_a": (tb / ta) if ta else None,
                    "delta_pct": ((tb - ta) / ta * 100.0) if ta else None,
                }
            )

    payload = {
        "baseline": baseline["name"],
        "num_cases": len(analysis_rows),
        "cases": analysis_rows,
        "pairwise": pairs,
        "notes": [
            "Profiled JSON files are excluded.",
            "rel_tok_s is relative to gpu1_ddp when present, else the first case.",
            "bwd_frac near 1.0 usually means collectives dominate the step.",
        ],
    }
    (out / "suite_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Engineering suite analysis",
        "",
        f"Baseline: `{baseline['name']}` @ {baseline['tok_s']:.0f} tok/s",
        "",
        "## Relative throughput",
        "",
        "| case | dp/tp/pp | tok/s | vs baseline | fwd% | bwd% | opt% |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in analysis_rows:
        lines.append(
            "| {name} | {dp}/{tp}/{pp} | {tok:.0f} | {rel:.2f}× | {ff:.0%} | {bf:.0%} | {of:.0%} |".format(
                name=r["name"],
                dp=r["dp"],
                tp=r["tp"],
                pp=r["pp"],
                tok=r["tok_s"],
                rel=r["rel_tok_s"],
                ff=r["fwd_frac"],
                bf=r["bwd_frac"],
                of=r["opt_frac"],
            )
        )

    if pairs:
        lines.extend(
            [
                "",
                "## Pairwise deltas",
                "",
                "| comparison | A tok/s | B tok/s | B/A | Δ% |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for p in pairs:
            lines.append(
                "| {label} | {a:.0f} | {b:.0f} | {s:.3f}× | {d:+.1f}% |".format(
                    label=p["label"],
                    a=p["tok_a"],
                    b=p["tok_b"],
                    s=p["speedup_b_over_a"] or 0.0,
                    d=p["delta_pct"] or 0.0,
                )
            )

    lines.extend(["", "Raw: `suite_analysis.json`", ""])
    (out / "suite_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
