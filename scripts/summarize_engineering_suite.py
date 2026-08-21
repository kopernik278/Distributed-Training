#!/usr/bin/env python3
"""Aggregate engineering suite JSON metrics into a comparison table + markdown."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <out_dir>")
    out = Path(sys.argv[1])
    rows = []
    for path in sorted(out.glob("*.json")):
        if path.name.startswith("suite_"):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        summary = payload.get("summary") or {}
        env = payload.get("environment") or {}
        cfg = payload.get("config") or {}
        rows.append(
            {
                "name": path.stem,
                "path": str(path),
                "gpu_name": env.get("gpu_name"),
                "gpu_count": env.get("gpu_count"),
                "world_size": env.get("world_size"),
                "backend": env.get("backend"),
                "pytorch": env.get("pytorch"),
                "nccl": env.get("nccl_version"),
                "commit": env.get("commit"),
                "dataset": cfg.get("dataset"),
                "hidden": cfg.get("hidden_size"),
                "layers": cfg.get("num_layers"),
                "seq": cfg.get("seq_len"),
                "batch": cfg.get("batch_size"),
                "vocab": cfg.get("vocab_size"),
                "dp": cfg.get("data_parallel_size") or env.get("data_parallel_size"),
                "tp": cfg.get("tensor_parallel_size"),
                "pp": cfg.get("pipeline_parallel_size"),
                "overlap": cfg.get("overlap"),
                "sequence_parallel": cfg.get("sequence_parallel"),
                "vocab_parallel": cfg.get("vocab_parallel"),
                "avg_loss": summary.get("avg_loss"),
                "step_ms": summary.get("avg_step_time_ms"),
                "fwd_ms": summary.get("avg_forward_ms"),
                "bwd_ms": summary.get("avg_backward_ms"),
                "opt_ms": summary.get("avg_optimizer_ms"),
                "tok_s": summary.get("avg_global_tokens_per_second"),
                "param_count": payload.get("parameter_count"),
            }
        )

    suite = {
        "num_cases": len(rows),
        "cases": rows,
        "notes": [
            "Throughput runs exclude profiler; profiled JSON filenames contain 'profiled'.",
            "Tokens/s is global (DP-scaled). Do not multiply by TP/PP.",
            "WikiText-2 word vocab is capped to --vocab-size; OOV → <unk>.",
        ],
    }
    (out / "suite_summary.json").write_text(json.dumps(suite, indent=2) + "\n")

    lines = [
        "# Engineering suite summary",
        "",
        f"Cases: **{len(rows)}**",
        "",
        "| case | dp/tp/pp | overlap | SP/VP | step ms | tok/s | loss |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        spvp = f"{int(bool(r.get('sequence_parallel')))}/{int(bool(r.get('vocab_parallel')))}"
        lines.append(
            "| {name} | {dp}/{tp}/{pp} | {ov} | {spvp} | {step:.2f} | {tok:.0f} | {loss:.4f} |".format(
                name=r["name"],
                dp=r.get("dp"),
                tp=r.get("tp"),
                pp=r.get("pp"),
                ov=r.get("overlap"),
                spvp=spvp,
                step=float(r.get("step_ms") or 0.0),
                tok=float(r.get("tok_s") or 0.0),
                loss=float(r.get("avg_loss") or 0.0),
            )
        )
    lines.extend(["", "Raw: `suite_summary.json`", ""])
    (out / "suite_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
