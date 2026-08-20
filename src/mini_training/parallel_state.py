from __future__ import annotations

import torch
import torch.distributed as dist

_TENSOR_MODEL_PARALLEL_GROUP: dist.ProcessGroup | None = None
_DATA_PARALLEL_GROUP: dist.ProcessGroup | None = None
_TENSOR_MODEL_PARALLEL_RANK = 0
_TENSOR_MODEL_PARALLEL_SIZE = 1
_DATA_PARALLEL_RANK = 0
_DATA_PARALLEL_SIZE = 1
_MODEL_PARALLEL_IS_INITIALIZED = False


def model_parallel_is_initialized() -> bool:
    return _MODEL_PARALLEL_IS_INITIALIZED


def get_tensor_model_parallel_group() -> dist.ProcessGroup | None:
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group() -> dist.ProcessGroup | None:
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_rank() -> int:
    return _TENSOR_MODEL_PARALLEL_RANK


def get_tensor_model_parallel_world_size() -> int:
    return _TENSOR_MODEL_PARALLEL_SIZE


def get_data_parallel_rank() -> int:
    return _DATA_PARALLEL_RANK


def get_data_parallel_world_size() -> int:
    return _DATA_PARALLEL_SIZE


def destroy_model_parallel() -> None:
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_RANK
    global _TENSOR_MODEL_PARALLEL_SIZE
    global _DATA_PARALLEL_RANK
    global _DATA_PARALLEL_SIZE
    global _MODEL_PARALLEL_IS_INITIALIZED

    _TENSOR_MODEL_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP = None
    _TENSOR_MODEL_PARALLEL_RANK = 0
    _TENSOR_MODEL_PARALLEL_SIZE = 1
    _DATA_PARALLEL_RANK = 0
    _DATA_PARALLEL_SIZE = 1
    _MODEL_PARALLEL_IS_INITIALIZED = False


def initialize_model_parallel(tensor_model_parallel_size: int = 1) -> None:
    """Initialize contiguous TP groups and strided DP groups.

    Rank layout example for world_size=4, tp_size=2:
      ranks [0,1] -> TP group 0 (dp_rank=0)
      ranks [2,3] -> TP group 1 (dp_rank=1)
      ranks [0,2] -> DP group for tp_rank=0
      ranks [1,3] -> DP group for tp_rank=1
    """
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_RANK
    global _TENSOR_MODEL_PARALLEL_SIZE
    global _DATA_PARALLEL_RANK
    global _DATA_PARALLEL_SIZE
    global _MODEL_PARALLEL_IS_INITIALIZED

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")

    if tensor_model_parallel_size < 1:
        raise ValueError("tensor_model_parallel_size must be >= 1")

    if not dist.is_initialized():
        if tensor_model_parallel_size != 1:
            raise RuntimeError("distributed process group must be initialized before TP>1")
        destroy_model_parallel()
        _MODEL_PARALLEL_IS_INITIALIZED = True
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % tensor_model_parallel_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tensor_model_parallel_size={tensor_model_parallel_size}"
        )

    destroy_model_parallel()

    tp_size = tensor_model_parallel_size
    dp_size = world_size // tp_size
    tp_rank = rank % tp_size
    dp_rank = rank // tp_size

    # Build TP groups: consecutive ranks.
    for dp_idx in range(dp_size):
        ranks = list(range(dp_idx * tp_size, (dp_idx + 1) * tp_size))
        group = dist.new_group(ranks)
        if dp_idx == dp_rank:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build DP groups: same tp_rank across DP replicas.
    for tp_idx in range(tp_size):
        ranks = list(range(tp_idx, world_size, tp_size))
        group = dist.new_group(ranks)
        if tp_idx == tp_rank:
            _DATA_PARALLEL_GROUP = group

    _TENSOR_MODEL_PARALLEL_SIZE = tp_size
    _TENSOR_MODEL_PARALLEL_RANK = tp_rank
    _DATA_PARALLEL_SIZE = dp_size
    _DATA_PARALLEL_RANK = dp_rank
    _MODEL_PARALLEL_IS_INITIALIZED = True


def get_data_parallel_src_rank() -> int:
    """Global rank of the DP-group leader (dp_rank == 0) for this TP shard."""
    return _TENSOR_MODEL_PARALLEL_RANK


def broadcast_parameters_within_dp(module: torch.nn.Module) -> None:
    """Broadcast parameters from dp_rank=0 so matching TP shards stay identical across DP."""
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
        raise ValueError(f"{name}={value} must be divisible by tensor_parallel_size={divisor}")
