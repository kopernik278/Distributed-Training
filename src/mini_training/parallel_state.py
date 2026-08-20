from __future__ import annotations

import torch
import torch.distributed as dist

_TENSOR_MODEL_PARALLEL_GROUP: dist.ProcessGroup | None = None
_PIPELINE_MODEL_PARALLEL_GROUP: dist.ProcessGroup | None = None
_DATA_PARALLEL_GROUP: dist.ProcessGroup | None = None
_TENSOR_MODEL_PARALLEL_RANK = 0
_TENSOR_MODEL_PARALLEL_SIZE = 1
_PIPELINE_MODEL_PARALLEL_RANK = 0
_PIPELINE_MODEL_PARALLEL_SIZE = 1
_DATA_PARALLEL_RANK = 0
_DATA_PARALLEL_SIZE = 1
_MODEL_PARALLEL_IS_INITIALIZED = False


def model_parallel_is_initialized() -> bool:
    return _MODEL_PARALLEL_IS_INITIALIZED


def get_tensor_model_parallel_group() -> dist.ProcessGroup | None:
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group() -> dist.ProcessGroup | None:
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group() -> dist.ProcessGroup | None:
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_rank() -> int:
    return _TENSOR_MODEL_PARALLEL_RANK


def get_tensor_model_parallel_world_size() -> int:
    return _TENSOR_MODEL_PARALLEL_SIZE


def get_pipeline_model_parallel_rank() -> int:
    return _PIPELINE_MODEL_PARALLEL_RANK


def get_pipeline_model_parallel_world_size() -> int:
    return _PIPELINE_MODEL_PARALLEL_SIZE


def get_data_parallel_rank() -> int:
    return _DATA_PARALLEL_RANK


def get_data_parallel_world_size() -> int:
    return _DATA_PARALLEL_SIZE


def is_pipeline_first_stage() -> bool:
    return _PIPELINE_MODEL_PARALLEL_RANK == 0


def is_pipeline_last_stage() -> bool:
    return _PIPELINE_MODEL_PARALLEL_RANK == (_PIPELINE_MODEL_PARALLEL_SIZE - 1)


def destroy_model_parallel() -> None:
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_RANK
    global _TENSOR_MODEL_PARALLEL_SIZE
    global _PIPELINE_MODEL_PARALLEL_RANK
    global _PIPELINE_MODEL_PARALLEL_SIZE
    global _DATA_PARALLEL_RANK
    global _DATA_PARALLEL_SIZE
    global _MODEL_PARALLEL_IS_INITIALIZED

    _TENSOR_MODEL_PARALLEL_GROUP = None
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP = None
    _TENSOR_MODEL_PARALLEL_RANK = 0
    _TENSOR_MODEL_PARALLEL_SIZE = 1
    _PIPELINE_MODEL_PARALLEL_RANK = 0
    _PIPELINE_MODEL_PARALLEL_SIZE = 1
    _DATA_PARALLEL_RANK = 0
    _DATA_PARALLEL_SIZE = 1
    _MODEL_PARALLEL_IS_INITIALIZED = False


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> None:
    """Initialize TP / PP / DP groups (Megatron rank ordering).

    rank = dp_rank * (pp * tp) + pp_rank * tp + tp_rank

    Example world_size=4, tp=2, pp=1 (Phase-3 compatible):
      TP groups: {0,1}, {2,3}
      DP groups: {0,2}, {1,3}

    Example world_size=2, tp=1, pp=2:
      PP group: {0,1}
      stage0=rank0, stage1=rank1
    """
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_RANK
    global _TENSOR_MODEL_PARALLEL_SIZE
    global _PIPELINE_MODEL_PARALLEL_RANK
    global _PIPELINE_MODEL_PARALLEL_SIZE
    global _DATA_PARALLEL_RANK
    global _DATA_PARALLEL_SIZE
    global _MODEL_PARALLEL_IS_INITIALIZED

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")

    if tensor_model_parallel_size < 1:
        raise ValueError("tensor_model_parallel_size must be >= 1")
    if pipeline_model_parallel_size < 1:
        raise ValueError("pipeline_model_parallel_size must be >= 1")

    if not dist.is_initialized():
        if tensor_model_parallel_size != 1 or pipeline_model_parallel_size != 1:
            raise RuntimeError("distributed process group must be initialized before TP>1 or PP>1")
        destroy_model_parallel()
        _MODEL_PARALLEL_IS_INITIALIZED = True
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tp*pp="
            f"{tensor_model_parallel_size}*{pipeline_model_parallel_size}"
        )

    destroy_model_parallel()

    tp_size = tensor_model_parallel_size
    pp_size = pipeline_model_parallel_size
    dp_size = world_size // model_parallel_size

    tp_rank = rank % tp_size
    pp_rank = (rank // tp_size) % pp_size
    dp_rank = rank // model_parallel_size

    # TP groups: same (dp, pp), contiguous tp ranks.
    for dp_idx in range(dp_size):
        for pp_idx in range(pp_size):
            ranks = [
                dp_idx * model_parallel_size + pp_idx * tp_size + tp_idx
                for tp_idx in range(tp_size)
            ]
            group = dist.new_group(ranks)
            if dp_idx == dp_rank and pp_idx == pp_rank:
                _TENSOR_MODEL_PARALLEL_GROUP = group

    # PP groups: same (dp, tp), stride tp across pipeline stages.
    for dp_idx in range(dp_size):
        for tp_idx in range(tp_size):
            ranks = [
                dp_idx * model_parallel_size + pp_idx * tp_size + tp_idx
                for pp_idx in range(pp_size)
            ]
            group = dist.new_group(ranks)
            if dp_idx == dp_rank and tp_idx == tp_rank:
                _PIPELINE_MODEL_PARALLEL_GROUP = group

    # DP groups: same (pp, tp) across data-parallel replicas.
    for pp_idx in range(pp_size):
        for tp_idx in range(tp_size):
            ranks = [
                dp_idx * model_parallel_size + pp_idx * tp_size + tp_idx
                for dp_idx in range(dp_size)
            ]
            group = dist.new_group(ranks)
            if pp_idx == pp_rank and tp_idx == tp_rank:
                _DATA_PARALLEL_GROUP = group

    _TENSOR_MODEL_PARALLEL_SIZE = tp_size
    _TENSOR_MODEL_PARALLEL_RANK = tp_rank
    _PIPELINE_MODEL_PARALLEL_SIZE = pp_size
    _PIPELINE_MODEL_PARALLEL_RANK = pp_rank
    _DATA_PARALLEL_SIZE = dp_size
    _DATA_PARALLEL_RANK = dp_rank
    _MODEL_PARALLEL_IS_INITIALIZED = True


def get_data_parallel_src_rank() -> int:
    """Global rank of the DP-group leader (dp_rank == 0) for this (pp, tp) shard."""
    return _PIPELINE_MODEL_PARALLEL_RANK * _TENSOR_MODEL_PARALLEL_SIZE + _TENSOR_MODEL_PARALLEL_RANK


def get_pipeline_model_parallel_prev_rank() -> int | None:
    if _PIPELINE_MODEL_PARALLEL_RANK == 0:
        return None
    model_parallel_size = _TENSOR_MODEL_PARALLEL_SIZE * _PIPELINE_MODEL_PARALLEL_SIZE
    return (
        _DATA_PARALLEL_RANK * model_parallel_size
        + (_PIPELINE_MODEL_PARALLEL_RANK - 1) * _TENSOR_MODEL_PARALLEL_SIZE
        + _TENSOR_MODEL_PARALLEL_RANK
    )


def get_pipeline_model_parallel_next_rank() -> int | None:
    if _PIPELINE_MODEL_PARALLEL_RANK == _PIPELINE_MODEL_PARALLEL_SIZE - 1:
        return None
    model_parallel_size = _TENSOR_MODEL_PARALLEL_SIZE * _PIPELINE_MODEL_PARALLEL_SIZE
    return (
        _DATA_PARALLEL_RANK * model_parallel_size
        + (_PIPELINE_MODEL_PARALLEL_RANK + 1) * _TENSOR_MODEL_PARALLEL_SIZE
        + _TENSOR_MODEL_PARALLEL_RANK
    )


def get_pipeline_model_parallel_last_rank() -> int:
    """Global rank of the last PP stage for this (dp, tp) peer."""
    model_parallel_size = _TENSOR_MODEL_PARALLEL_SIZE * _PIPELINE_MODEL_PARALLEL_SIZE
    return (
        _DATA_PARALLEL_RANK * model_parallel_size
        + (_PIPELINE_MODEL_PARALLEL_SIZE - 1) * _TENSOR_MODEL_PARALLEL_SIZE
        + _TENSOR_MODEL_PARALLEL_RANK
    )


def broadcast_parameters_within_dp(module: torch.nn.Module) -> None:
    """Broadcast parameters from dp_rank=0 so matching shards stay identical across DP."""
    if not dist.is_initialized() or _DATA_PARALLEL_SIZE <= 1:
        return
    group = _DATA_PARALLEL_GROUP
    src = get_data_parallel_src_rank()
    for param in module.parameters():
        dist.broadcast(param.data, src=src, group=group)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=src, group=group)


def ensure_divisible(value: int, divisor: int, name: str) -> None:
    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by divisor={divisor}")
