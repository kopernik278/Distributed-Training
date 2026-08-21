from __future__ import annotations

import torch
import torch.distributed as dist

from .parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from .profiler import record_range

_OVERLAP_ENABLED = False
_COMM_STREAM: torch.cuda.Stream | None = None


def set_overlap_enabled(enabled: bool) -> None:
    global _OVERLAP_ENABLED
    _OVERLAP_ENABLED = bool(enabled)


def overlap_enabled() -> bool:
    return _OVERLAP_ENABLED


def _comm_stream_for(tensor: torch.Tensor) -> torch.cuda.Stream | None:
    if not (overlap_enabled() and tensor.is_cuda and torch.cuda.is_available()):
        return None
    global _COMM_STREAM
    if _COMM_STREAM is None:
        _COMM_STREAM = torch.cuda.Stream()
    return _COMM_STREAM


def async_tp_all_reduce(tensor: torch.Tensor) -> dist.Work | None:
    """SUM AllReduce on the TP group. Returns a handle if launched asynchronously.

    CUDA + overlap: NCCL runs on a dedicated stream so the default stream can GEMM.
    Caller must ``wait_tp_all_reduce`` before reading ``tensor``.
    """
    if get_tensor_model_parallel_world_size() <= 1:
        return None
    group = get_tensor_model_parallel_group()
    comm_stream = _comm_stream_for(tensor)
    with record_range("tp_all_reduce"):
        if overlap_enabled() and comm_stream is not None:
            default = torch.cuda.current_stream()
            comm_stream.wait_stream(default)
            with torch.cuda.stream(comm_stream):
                work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
            return work
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return None


def wait_tp_all_reduce(work: dist.Work | None, tensor: torch.Tensor) -> None:
    if work is not None:
        work.wait()
    comm_stream = _comm_stream_for(tensor) if tensor.is_cuda else None
    if comm_stream is not None:
        torch.cuda.current_stream().wait_stream(comm_stream)
