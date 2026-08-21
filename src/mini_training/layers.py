from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from .mappings import (
    finalize_tensor_parallel_output,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .overlap import async_tp_all_reduce, flush_pending_tp_reduces, wait_tp_all_reduce
from .parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from .profiler import record_range


def _set_tensor_parallel_attributes(tensor: torch.Tensor, partition_dim: int) -> None:
    setattr(tensor, "tensor_model_parallel", True)
    setattr(tensor, "partition_dim", int(partition_dim))


def _all_gather_seq(input_: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return input_
    group = get_tensor_model_parallel_group()
    pieces = [torch.empty_like(input_) for _ in range(tp_size)]
    with record_range("sp_all_gather"):
        dist.all_gather(pieces, input_.contiguous(), group=group)
    return torch.cat(pieces, dim=1).contiguous()


def _reduce_scatter_seq(input_: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return input_
    group = get_tensor_model_parallel_group()
    chunks = [c.contiguous() for c in input_.chunk(tp_size, dim=1)]
    output = torch.empty_like(chunks[0])
    with record_range("sp_reduce_scatter"):
        dist.reduce_scatter(output, chunks, group=group)
    return output


class _ColumnParallelLinearFn(torch.autograd.Function):
    """Y = X W^T + b with TP input-grad reduce.

    Without SP: AllReduce(dX). With SP: AllGather(X) in forward, ReduceScatter(dX)
    in backward (equivalent to AllReduce then slice).
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, sequence_parallel: bool):  # type: ignore[override]
        ctx.sequence_parallel = bool(sequence_parallel)
        if ctx.sequence_parallel:
            input_ = _all_gather_seq(input_)
        ctx.save_for_backward(input_, weight, bias)
        ctx.has_bias = bias is not None
        return F.linear(input_, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        input_, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        input_2d = input_.reshape(-1, input_.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        grad_input_2d = grad_output_2d.matmul(weight)
        grad_input = grad_input_2d.view_as(input_)

        if ctx.sequence_parallel:
            # Fuse AllReduce+slice into ReduceScatter along sequence.
            grad_weight = grad_output_2d.t().matmul(input_2d)
            grad_bias = grad_output_2d.sum(dim=0) if ctx.has_bias else None
            grad_input = _reduce_scatter_seq(grad_input)
            return grad_input, grad_weight, grad_bias, None

        work = async_tp_all_reduce(grad_input_2d)
        grad_weight = grad_output_2d.t().matmul(input_2d)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output_2d.sum(dim=0)
        wait_tp_all_reduce(work, grad_input_2d)
        grad_input = grad_input_2d.view_as(input_)
        return grad_input, grad_weight, grad_bias, None


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    Weight is partitioned along the output dimension.
    Y = X A + b, where A is split by columns across TP ranks.

    When ``sequence_parallel`` is True, input is sequence-sharded ``[B, S/tp, H]``;
    forward AllGathers sequence, backward ReduceScatters ``dX``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        gather_output: bool = True,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel

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
        flush_pending_tp_reduces()
        output_parallel = _ColumnParallelLinearFn.apply(
            input_,
            self.weight,
            self.bias if self.bias is not None else None,
            self.sequence_parallel,
        )
        if self.gather_output:
            return gather_from_tensor_model_parallel_region(output_parallel)
        return output_parallel


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    When ``sequence_parallel`` is True, output uses ReduceScatter on the sequence
    dim instead of AllReduce (activations stay ``[B, S/tp, H]``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        input_is_parallel: bool = False,
        skip_bias_add: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.sequence_parallel = sequence_parallel

        tp_size = get_tensor_model_parallel_world_size()
        if in_features % tp_size != 0:
            raise ValueError(f"in_features={in_features} must be divisible by tp_size={tp_size}")
        self.input_size_per_partition = in_features // tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.input_size_per_partition))
        if bias:
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
        if self.sequence_parallel:
            output = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output = reduce_from_tensor_model_parallel_region(output_parallel)
        if self.skip_bias_add:
            return output
        return finalize_tensor_parallel_output(output, self.bias)


class _SyncAllReduce(torch.autograd.Function):
    """Forward SUM AllReduce; backward identity (Megatron reduce-from)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output = input_.clone()
        if get_tensor_model_parallel_world_size() > 1:
            with record_range("tp_all_reduce"):
                dist.all_reduce(
                    output,
                    op=dist.ReduceOp.SUM,
                    group=get_tensor_model_parallel_group(),
                )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output


class VocabParallelEmbedding(nn.Module):
    """Embedding with vocabulary partitioned across the TP group."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if num_embeddings % tp_size != 0:
            raise ValueError(f"vocab_size={num_embeddings} must be divisible by tp_size={tp_size}")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vocab_size_per_partition = num_embeddings // tp_size
        self.vocab_start = get_tensor_model_parallel_rank() * self.vocab_size_per_partition
        self.vocab_end = self.vocab_start + self.vocab_size_per_partition

        self.weight = nn.Parameter(torch.empty(self.vocab_size_per_partition, embedding_dim))
        self.reset_parameters()
        _set_tensor_parallel_attributes(self.weight, partition_dim=0)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if get_tensor_model_parallel_world_size() == 1:
            return F.embedding(input_ids, self.weight)

        # Mask tokens outside this rank's vocab slice; shift ids into local range.
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = input_ids - self.vocab_start
        local_ids = local_ids.masked_fill(~mask, 0)
        output_parallel = F.embedding(local_ids, self.weight)
        output_parallel = output_parallel * mask.unsqueeze(-1).to(output_parallel.dtype)
        return _SyncAllReduce.apply(output_parallel)

class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Parallel softmax CE over sharded logits ``[..., V/tp]`` (Megatron-style)."""

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, target: torch.Tensor):  # type: ignore[override]
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        group = get_tensor_model_parallel_group()

        logits = vocab_parallel_logits.float()
        # Global max for numerical stability.
        logits_max = logits.max(dim=-1).values
        if tp_size > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        logits = logits - logits_max.unsqueeze(-1)

        exp_logits = logits.exp()
        sum_exp = exp_logits.sum(dim=-1)
        if tp_size > 1:
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        vocab_per_partition = logits.size(-1)
        vocab_start = tp_rank * vocab_per_partition
        target_mask = (target >= vocab_start) & (target < vocab_start + vocab_per_partition)
        masked_target = target.clone() - vocab_start
        masked_target = masked_target.masked_fill(~target_mask, 0)

        predicted = torch.gather(
            exp_logits,
            dim=-1,
            index=masked_target.unsqueeze(-1),
        ).squeeze(-1)
        predicted = predicted * target_mask.to(predicted.dtype)
        if tp_size > 1:
            dist.all_reduce(predicted, op=dist.ReduceOp.SUM, group=group)

        loss = (sum_exp.log() - predicted.clamp_min(1e-20).log()).mean()

        # Softmax partition for backward: exp / sum_exp
        softmax = exp_logits / sum_exp.unsqueeze(-1)
        ctx.save_for_backward(softmax, target_mask, masked_target)
        ctx.vocab_parallel_logits_shape = vocab_parallel_logits.shape
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        softmax, target_mask, masked_target = ctx.saved_tensors
        grad = softmax
        # Subtract 1 from the target class probability on the owning partition.
        grad_2d = grad.reshape(-1, grad.size(-1))
        mask_1d = target_mask.reshape(-1)
        idx_1d = masked_target.reshape(-1)
        rows = torch.arange(grad_2d.size(0), device=grad.device)
        selected = mask_1d.nonzero(as_tuple=False).squeeze(-1)
        if selected.numel() > 0:
            grad_2d[rows[selected], idx_1d[selected]] -= 1.0
        grad = grad_2d.view_as(softmax)
        # Mean reduction over batch*seq elements.
        numel = softmax.shape[:-1].numel()
        grad = grad * (grad_output.to(grad.dtype) / max(numel, 1))
        return grad, None


def vocab_parallel_cross_entropy(vocab_parallel_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)


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


def shard_vocab_weight(full_weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Shard embedding/LM-head weight [vocab, hidden] along vocab dim."""
    vocab = full_weight.size(0)
    assert vocab % tp_size == 0
    local = vocab // tp_size
    start = tp_rank * local
    return full_weight[start : start + local, :].contiguous()


def parameter_count(module: nn.Module) -> int:
    return sum(math.prod(param.shape) for param in module.parameters())
