from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .parallel_state import (
    get_tensor_model_parallel_world_size,
)


def _set_tensor_parallel_attributes(tensor: torch.Tensor, partition_dim: int) -> None:
    setattr(tensor, "tensor_model_parallel", True)
    setattr(tensor, "partition_dim", int(partition_dim))


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    Weight is partitioned along the output dimension.
    Y = X A + b, where A is split by columns across TP ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        gather_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        tp_size = get_tensor_model_parallel_world_size()
        if out_features % tp_size != 0:
            raise ValueError(f"out_features={out_features} must be divisible by tp_size={tp_size}")
        self.output_size_per_partition = out_features // tp_size

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        _set_tensor_parallel_attributes(self.weight, partition_dim=0)
        if self.bias is not None:
            _set_tensor_parallel_attributes(self.bias, partition_dim=0)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            return gather_from_tensor_model_parallel_region(output_parallel)
        return output_parallel


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    Weight is partitioned along the input dimension.
    Y = X A + b, where A is split by rows across TP ranks, then outputs are AllReduced.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        input_is_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        tp_size = get_tensor_model_parallel_world_size()
        if in_features % tp_size != 0:
            raise ValueError(f"in_features={in_features} must be divisible by tp_size={tp_size}")
        self.input_size_per_partition = in_features // tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.input_size_per_partition))
        if bias:
            # Bias is replicated on every rank and added AFTER AllReduce so each
            # rank observes the same output.
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        _set_tensor_parallel_attributes(self.weight, partition_dim=1)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        output_parallel = F.linear(input_parallel, self.weight)
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output + self.bias
        return output


def shard_column_weight(full_weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Shard nn.Linear.weight [out, in] along output rows for ColumnParallel."""
    out_features = full_weight.size(0)
    assert out_features % tp_size == 0
    local = out_features // tp_size
    start = tp_rank * local
    return full_weight[start : start + local, :].contiguous()


def shard_row_weight(full_weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Shard nn.Linear.weight [out, in] along input columns for RowParallel."""
    in_features = full_weight.size(1)
    assert in_features % tp_size == 0
    local = in_features // tp_size
    start = tp_rank * local
    return full_weight[:, start : start + local].contiguous()


def parameter_count(module: nn.Module) -> int:
    return sum(math.prod(param.shape) for param in module.parameters())
