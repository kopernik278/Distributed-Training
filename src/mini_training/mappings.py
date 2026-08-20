from __future__ import annotations

import torch
import torch.distributed as dist

from .parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def _tp_group() -> dist.ProcessGroup | None:
    return get_tensor_model_parallel_group()


def _tp_size() -> int:
    return get_tensor_model_parallel_world_size()


def _tp_rank() -> int:
    return get_tensor_model_parallel_rank()


def _split_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    tp_size = _tp_size()
    if tp_size == 1:
        return input_
    last_dim = input_.size(-1)
    assert last_dim % tp_size == 0, "last dimension must be divisible by tp size"
    local = last_dim // tp_size
    start = _tp_rank() * local
    return input_[..., start : start + local].contiguous()


def _gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    tp_size = _tp_size()
    if tp_size == 1:
        return input_
    group = _tp_group()
    tensor_list = [torch.empty_like(input_) for _ in range(tp_size)]
    dist.all_gather(tensor_list, input_.contiguous(), group=group)
    return torch.cat(tensor_list, dim=-1).contiguous()


def _reduce(input_: torch.Tensor) -> torch.Tensor:
    if _tp_size() == 1:
        return input_
    dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=_tp_group())
    return input_


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Forward identity; backward AllReduce."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return _reduce(grad_output.clone())


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Forward AllReduce; backward identity."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _reduce(input_.clone())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Forward split; backward AllGather."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Forward AllGather; backward split."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return _split_along_last_dim(grad_output)


def copy_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)
