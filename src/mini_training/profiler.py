from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch.profiler import ProfilerActivity, record_function


def record_range(name: str):
    """Named range visible in Chrome traces. Cheap no-op when profiler is off."""
    return record_function(name)


def _activities(device: torch.device) -> list[ProfilerActivity]:
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts


@contextmanager
def maybe_profile(
    *,
    enabled: bool,
    profile_dir: str | Path,
    device: torch.device,
    wait: int,
    warmup: int,
    active: int,
    rank: int,
) -> Iterator[torch.profiler.profile | None]:
    """Yield a torch.profiler or None.

    Schedule is wait / warmup / active (then repeats until training ends).
    Only rank 0 is required to export, but every rank profiles so collectives match.
    """
    if not enabled:
        yield None
        return

    wait = max(0, int(wait))
    warmup = max(1, int(warmup))
    active = max(1, int(active))

    out_dir = Path(profile_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    def _trace_handler(prof: torch.profiler.profile) -> None:
        trace_path = out_dir / f"rank{rank}.json"
        prof.export_chrome_trace(str(trace_path))
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        summary_path = out_dir / f"rank{rank}_summary.txt"
        summary_path.write_text(table + "\n", encoding="utf-8")

    with torch.profiler.profile(
        activities=_activities(device),
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=_trace_handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        yield prof


def profiler_step(prof: torch.profiler.profile | None) -> None:
    if prof is not None:
        prof.step()
