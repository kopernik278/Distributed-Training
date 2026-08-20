from __future__ import annotations

import os
import random
import time

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def infer_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def init_distributed(backend: str) -> tuple[torch.device, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = infer_device()
    actual_backend = backend if device.type == "cuda" else "gloo"
    if world_size > 1 and not dist.is_initialized():
        init_kwargs = {
            "backend": actual_backend,
            "timeout": torch.distributed.constants.default_pg_timeout,
        }
        # PyTorch >= 2.x recommends passing device_id for NCCL to avoid ambiguous rank-device mapping.
        if device.type == "cuda" and device.index is not None:
            init_kwargs["device_id"] = device
        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            # Older torch without device_id support.
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)
    return device, world_size, actual_backend


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    """Set Python/Torch RNG to an absolute seed (caller chooses DP/TP offsets)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return tensor.item()


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


class Timer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.start_time = 0.0

    def __enter__(self) -> "Timer":
        synchronize_device(self.device)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *_args: object) -> None:
        synchronize_device(self.device)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.start_time
