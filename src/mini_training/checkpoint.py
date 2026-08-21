from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import TrainingConfig
from .distributed import barrier, get_rank, is_distributed, is_main_process
from .parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

CHECKPOINT_FORMAT_VERSION = 1


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def shard_filename(pp_rank: int, tp_rank: int) -> str:
    return f"mp_rank_pp{pp_rank:03d}_tp{tp_rank:03d}.pt"


def checkpoint_step_dir(checkpoint_dir: str | Path, step: int) -> Path:
    return Path(checkpoint_dir) / f"step_{step:06d}"


def build_param_specs(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Describe which parameters are TP-sharded and along which dimension."""
    specs: dict[str, dict[str, Any]] = {}
    for name, param in unwrap_model(model).named_parameters():
        is_tp = bool(getattr(param, "tensor_model_parallel", False))
        partition_dim = getattr(param, "partition_dim", None) if is_tp else None
        specs[name] = {
            "tensor_model_parallel": is_tp,
            "partition_dim": partition_dim,
            "shape": list(param.shape),
        }
    return specs


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: TrainingConfig,
) -> Path:
    """Save a sharded checkpoint. Only dp_rank==0 writes each (pp, tp) shard."""
    step_dir = checkpoint_step_dir(checkpoint_dir, step)
    if is_main_process():
        step_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    pp_rank = get_pipeline_model_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()
    pp_size = get_pipeline_model_parallel_world_size()
    tp_size = get_tensor_model_parallel_world_size()
    core = unwrap_model(model)

    if get_data_parallel_rank() == 0:
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": int(step),
            "pp_rank": int(pp_rank),
            "tp_rank": int(tp_rank),
            "pp_size": int(pp_size),
            "tp_size": int(tp_size),
            "model": _cpu_state_dict(core),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "param_specs": build_param_specs(core),
        }
        torch.save(payload, step_dir / shard_filename(pp_rank, tp_rank))

    if is_main_process():
        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": int(step),
            "config": asdict(config),
            "parallel": {
                "data_parallel_size": get_data_parallel_world_size(),
                "pipeline_parallel_size": pp_size,
                "tensor_parallel_size": tp_size,
            },
        }
        (step_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        latest = Path(checkpoint_dir) / "latest"
        latest.write_text(str(step_dir.name) + "\n", encoding="utf-8")

    barrier()
    return step_dir


def load_metadata(step_dir: str | Path) -> dict[str, Any]:
    path = Path(step_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Accept either a step dir or a parent dir containing ``latest``."""
    path = Path(path)
    if (path / "metadata.json").is_file():
        return path
    latest = path / "latest"
    if latest.is_file():
        name = latest.read_text(encoding="utf-8").strip()
        candidate = path / name
        if (candidate / "metadata.json").is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint metadata under {path}")


def _torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_shard(step_dir: Path, pp_rank: int, tp_rank: int, map_location: str | torch.device) -> dict[str, Any]:
    path = step_dir / shard_filename(pp_rank, tp_rank)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint shard: {path}")
    return _torch_load(path, map_location)


def consolidate_tp_state_dicts(
    shard_states: list[dict[str, torch.Tensor]],
    param_specs: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Merge TP shards into full tensors using partition_dim metadata."""
    if not shard_states:
        raise ValueError("shard_states must be non-empty")
    full: dict[str, torch.Tensor] = {}
    keys = shard_states[0].keys()
    for key in keys:
        spec = param_specs.get(key, {"tensor_model_parallel": False, "partition_dim": None})
        pieces = [state[key].cpu() for state in shard_states]
        if not spec.get("tensor_model_parallel"):
            full[key] = pieces[0].contiguous()
            continue
        dim = spec.get("partition_dim")
        if dim is None:
            raise ValueError(f"TP parameter {key} missing partition_dim in param_specs")
        full[key] = torch.cat(pieces, dim=int(dim)).contiguous()
    return full


def shard_full_state_dict(
    full_state: dict[str, torch.Tensor],
    param_specs: dict[str, dict[str, Any]],
    *,
    tp_rank: int,
    tp_size: int,
) -> dict[str, torch.Tensor]:
    """Split consolidated tensors into the local TP shard."""
    local: dict[str, torch.Tensor] = {}
    for key, tensor in full_state.items():
        spec = param_specs.get(key, {"tensor_model_parallel": False, "partition_dim": None})
        if not spec.get("tensor_model_parallel"):
            local[key] = tensor.contiguous()
            continue
        dim = int(spec["partition_dim"])
        size = tensor.size(dim)
        if size % tp_size != 0:
            raise ValueError(f"{key} dim{dim}={size} not divisible by tp_size={tp_size}")
        local_size = size // tp_size
        start = tp_rank * local_size
        local[key] = tensor.narrow(dim, start, local_size).contiguous().clone()
    return local


def _param_specs_from_model_or_shards(
    model: nn.Module,
    shard_payloads: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    # Prefer live model tags; fall back to saved specs (needed when consolidating offline).
    live = build_param_specs(model)
    if live:
        return live
    for payload in shard_payloads:
        if payload.get("param_specs"):
            return payload["param_specs"]
    raise RuntimeError("Unable to resolve param_specs for reshard")


def load_model_state_with_optional_reshard(
    step_dir: str | Path,
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model weights, consolidating/splitting TP shards when tp_size differs.

    Returns metadata dict. Optimizer is not loaded here.
    """
    step_dir = Path(step_dir)
    metadata = load_metadata(step_dir)
    saved_tp = int(metadata["parallel"]["tensor_parallel_size"])
    saved_pp = int(metadata["parallel"]["pipeline_parallel_size"])
    cur_tp = get_tensor_model_parallel_world_size()
    cur_pp = get_pipeline_model_parallel_world_size()
    pp_rank = get_pipeline_model_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()

    if saved_pp != cur_pp:
        raise ValueError(
            f"PP reshard not supported in this milestone: checkpoint pp={saved_pp}, current pp={cur_pp}"
        )

    core = unwrap_model(model)

    if saved_tp == cur_tp:
        payload = _load_shard(step_dir, pp_rank, tp_rank, map_location)
        core.load_state_dict(payload["model"], strict=True)
        return metadata

    # TP reshard: consolidate all source TP shards for this pp_rank, then split.
    payloads = [_load_shard(step_dir, pp_rank, src_tp, map_location) for src_tp in range(saved_tp)]
    specs = _param_specs_from_model_or_shards(core, payloads)
    # When loading into a different tp size, prefer destination model specs for split,
    # but consolidation must use source specs (saved).
    source_specs = payloads[0].get("param_specs") or specs
    full = consolidate_tp_state_dicts([p["model"] for p in payloads], source_specs)
    # Destination specs: mark TP params from the live model.
    dest_specs = build_param_specs(core)
    # Keys that exist in full but are TP in dest need partition info; ensure dest_specs cover them.
    local = shard_full_state_dict(full, dest_specs, tp_rank=tp_rank, tp_size=cur_tp)
    missing_unexpected = core.load_state_dict(local, strict=True)
    _ = missing_unexpected
    return metadata


def load_optimizer_if_compatible(
    step_dir: str | Path,
    optimizer: torch.optim.Optimizer,
    *,
    map_location: str | torch.device = "cpu",
) -> bool:
    """Restore optimizer only when TP/PP topology matches the checkpoint."""
    step_dir = Path(step_dir)
    metadata = load_metadata(step_dir)
    saved_tp = int(metadata["parallel"]["tensor_parallel_size"])
    saved_pp = int(metadata["parallel"]["pipeline_parallel_size"])
    if saved_tp != get_tensor_model_parallel_world_size() or saved_pp != get_pipeline_model_parallel_world_size():
        return False
    payload = _load_shard(
        step_dir,
        get_pipeline_model_parallel_rank(),
        get_tensor_model_parallel_rank(),
        map_location,
    )
    opt_state = payload.get("optimizer")
    if opt_state is None:
        return False
    optimizer.load_state_dict(opt_state)
    return True


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> tuple[int, dict[str, Any], bool]:
    """Load checkpoint for current rank.

    Returns ``(step, metadata, optimizer_loaded)``.
    """
    step_dir = resolve_checkpoint_dir(checkpoint_path)
    if map_location is None:
        map_location = "cpu"
    metadata = load_model_state_with_optional_reshard(step_dir, model, map_location=map_location)
    optimizer_loaded = False
    if optimizer is not None:
        optimizer_loaded = load_optimizer_if_compatible(step_dir, optimizer, map_location=map_location)
    step = int(metadata["step"])
    if is_distributed():
        barrier()
    return step, metadata, optimizer_loaded
