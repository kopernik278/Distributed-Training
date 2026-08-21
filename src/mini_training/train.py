from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from .config import TrainingConfig
from .data import RandomTokenDataset
from .distributed import (
    Timer,
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    reduce_mean,
    set_seed,
)
from .model import MiniTransformerLM, PipelineStage, cross_entropy_loss, model_parameter_count
from .parallel_state import (
    broadcast_parameters_within_dp,
    destroy_model_parallel,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    initialize_model_parallel,
)
from .profiler import maybe_profile, profiler_step, record_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini distributed training engine")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Microbatch size when PP>1")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument(
        "--num-microbatches",
        type=int,
        default=None,
        help="PP microbatches per step (default: max(pp_size, 1); ignored when pp=1)",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=None,
        help="Optional. Must equal world_size // (pp * tp) when set.",
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--warmup-discard", type=int, default=1, help="Drop first N steps from summary averages")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Directory for sharded checkpoints (step_XXXXXX/)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save every N steps when --checkpoint-dir is set (0 disables periodic saves; final step still saved)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from a step dir or a checkpoint root containing latest",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="",
        help="If set, export Chrome traces + a text table under this directory (slow; not for published tokens/s)",
    )
    parser.add_argument("--profile-wait", type=int, default=1, help="Profiler: steps to skip before warmup")
    parser.add_argument("--profile-warmup", type=int, default=1, help="Profiler: warmup steps")
    parser.add_argument("--profile-active", type=int, default=2, help="Profiler: recorded steps")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    pp = args.pipeline_parallel_size
    num_microbatches = args.num_microbatches
    if num_microbatches is None:
        num_microbatches = max(pp, 1)
    if pp == 1:
        num_microbatches = 1
    return TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=pp,
        num_microbatches=num_microbatches,
        data_parallel_size=args.data_parallel_size,
        log_interval=args.log_interval,
    )


def linear_warmup_lr(step: int, config: TrainingConfig) -> float:
    if config.warmup_steps <= 0:
        return config.lr
    scale = min((step + 1) / config.warmup_steps, 1.0)
    return config.lr * scale


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_environment(device: torch.device, backend: str) -> dict[str, object]:
    env: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "backend": backend,
        "world_size": get_world_size(),
        "rank": get_rank(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "tensor_parallel_size": get_tensor_model_parallel_world_size(),
        "tensor_parallel_rank": get_tensor_model_parallel_rank(),
        "pipeline_parallel_size": get_pipeline_model_parallel_world_size(),
        "pipeline_parallel_rank": get_pipeline_model_parallel_rank(),
        "data_parallel_size": get_data_parallel_world_size(),
        "data_parallel_rank": get_data_parallel_rank(),
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(device)
        env["gpu_count"] = torch.cuda.device_count()
        if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
            env["nccl_version"] = ".".join(str(part) for part in torch.cuda.nccl.version())
    return env


def summarize_metrics(metrics: list[dict[str, float | int]], warmup_discard: int) -> dict[str, float | int]:
    usable = metrics[warmup_discard:] if len(metrics) > warmup_discard else metrics
    if not usable:
        return {"num_steps": 0}

    def mean(key: str) -> float:
        return sum(float(item[key]) for item in usable) / len(usable)

    return {
        "num_steps": len(usable),
        "warmup_discard": warmup_discard,
        "avg_loss": mean("loss"),
        "avg_step_time_ms": mean("step_time_ms"),
        "avg_forward_ms": mean("forward_ms"),
        "avg_backward_ms": mean("backward_ms"),
        "avg_optimizer_ms": mean("optimizer_ms"),
        "avg_rank_tokens_per_second": mean("rank_tokens_per_second"),
        "avg_global_tokens_per_second": mean("global_tokens_per_second"),
    }


def validate_parallel_sizes(world_size: int, config: TrainingConfig) -> int:
    if config.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if config.pipeline_parallel_size < 1:
        raise ValueError("pipeline_parallel_size must be >= 1")
    model_parallel = config.tensor_parallel_size * config.pipeline_parallel_size
    if world_size % model_parallel != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tp*pp="
            f"{config.tensor_parallel_size}*{config.pipeline_parallel_size}"
        )
    if config.pipeline_parallel_size > 1 and config.num_layers % config.pipeline_parallel_size != 0:
        raise ValueError(
            f"num_layers={config.num_layers} must be divisible by "
            f"pipeline_parallel_size={config.pipeline_parallel_size}"
        )
    if config.pipeline_parallel_size > 1 and config.num_microbatches < 1:
        raise ValueError("num_microbatches must be >= 1 when pipeline parallel is enabled")
    inferred_dp = world_size // model_parallel
    if config.data_parallel_size is not None and config.data_parallel_size != inferred_dp:
        raise ValueError(
            f"data_parallel_size={config.data_parallel_size} conflicts with "
            f"world_size={world_size} / (pp*tp) (inferred dp={inferred_dp})"
        )
    return inferred_dp


def _wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not (is_distributed() and get_data_parallel_world_size() > 1):
        return model
    ddp_kwargs = {
        "device_ids": [device.index] if device.type == "cuda" else None,
        "process_group": get_data_parallel_group(),
    }
    if ddp_kwargs["device_ids"] is None:
        ddp_kwargs.pop("device_ids")
    return DDP(model, **ddp_kwargs)


def _run_non_pipeline_step(
    model: torch.nn.Module,
    dataset: RandomTokenDataset,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    forward_s = 0.0
    backward_s = 0.0
    loss_value = 0.0
    for _micro_step in range(config.grad_accum_steps):
        inputs, targets = dataset.next_batch()
        with Timer(device) as forward_timer:
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets) / config.grad_accum_steps
        forward_s += forward_timer.elapsed_s
        with Timer(device) as backward_timer:
            loss.backward()
        backward_s += backward_timer.elapsed_s
        loss_value += loss.item()
    return loss_value, forward_s, backward_s


def _run_pipeline_step(
    engine: PipelineEngine,
    dataset: RandomTokenDataset,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """One optimizer step of 1F1B. Timing is coarse (whole schedule)."""
    forward_s = 0.0
    backward_s = 0.0
    loss_value = 0.0
    for _accum in range(config.grad_accum_steps):
        microbatches: list[MicrobatchIO] = []
        for _ in range(config.num_microbatches):
            inputs, targets = dataset.next_batch()
            microbatches.append(MicrobatchIO(input_ids=inputs, targets=targets))
        with Timer(device) as pipe_timer:
            step_loss = engine.run(microbatches, device)
        # Attribute whole schedule time to forward bucket for simplicity; PP mixes F/B.
        forward_s += pipe_timer.elapsed_s
        loss_value += step_loss / config.grad_accum_steps
    return loss_value, forward_s, backward_s


def _sync_pipeline_loss(loss_value: float, device: torch.device) -> float:
    """Broadcast loss from last PP stage so all ranks share the same scalar for logging."""
    tensor = torch.tensor([loss_value], device=device, dtype=torch.float64)
    if is_distributed() and get_pipeline_model_parallel_world_size() > 1:
        import torch.distributed as dist

        dist.broadcast(
            tensor,
            src=get_pipeline_model_parallel_last_rank(),
            group=get_pipeline_model_parallel_group(),
        )
    return float(tensor.item())


def main() -> None:
    args = parse_args()
    config = build_config(args)
    device, world_size, actual_backend = init_distributed(config.backend)
    inferred_dp = validate_parallel_sizes(world_size, config)

    initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_parallel_size,
        pipeline_model_parallel_size=config.pipeline_parallel_size,
    )
    if get_data_parallel_world_size() != inferred_dp:
        raise RuntimeError(
            f"parallel_state dp_size={get_data_parallel_world_size()} != inferred {inferred_dp}"
        )

    # Matching TP shards across DP replicas must share init; TP ranks differ.
    # PP stages own different layers, so also offset by pp_rank.
    set_seed(config.seed + get_tensor_model_parallel_rank() + 1_000 * get_pipeline_model_parallel_rank())
    use_pipeline = config.pipeline_parallel_size > 1
    if use_pipeline:
        model: torch.nn.Module = PipelineStage(config).to(device)
    else:
        model = MiniTransformerLM(config).to(device)
    broadcast_parameters_within_dp(model)
    parameter_count = model_parameter_count(model)
    model = _wrap_ddp(model, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_step = 0
    resumed_optimizer = False
    if args.resume:
        loaded_step, ckpt_meta, resumed_optimizer = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        # Keep DP replicas aligned after load/reshard.
        broadcast_parameters_within_dp(unwrap_model(model))
        start_step = loaded_step + 1
        if is_main_process():
            print(
                json.dumps(
                    {
                        "event": "resume",
                        "path": args.resume,
                        "loaded_step": loaded_step,
                        "start_step": start_step,
                        "optimizer_loaded": resumed_optimizer,
                        "checkpoint_parallel": ckpt_meta.get("parallel"),
                    }
                )
            )

    # All PP ranks in a DP replica share the same microbatch stream (first uses
    # tokens, last uses targets; shapes must match for P2P).
    dataset = RandomTokenDataset(
        config,
        device,
        seed=config.seed + 10_000 + get_data_parallel_rank() + 1_000_000 * start_step,
    )
    set_seed(config.seed + 20_000 + get_data_parallel_rank() + 1_000_000 * start_step)

    engine: PipelineEngine | None = None
    if use_pipeline:
        # Keep DDP wrapper when present so DP gradient hooks still run.
        param = next(model.parameters())
        engine = PipelineEngine(
            model,
            num_microbatches=config.num_microbatches,
            hidden_size=config.hidden_size,
            dtype=param.dtype,
        )

    environment = collect_environment(device, actual_backend)
    metrics: list[dict[str, float | int]] = []
    barrier()

    if is_main_process():
        print(
            json.dumps(
                {
                    "event": "startup",
                    "world_size": get_world_size(),
                    "device": str(device),
                    "backend": actual_backend,
                    "config": asdict(config),
                    "parameter_count": parameter_count,
                    "environment": environment,
                    "profiling": bool(args.profile_dir),
                }
            )
        )

    end_step = start_step + config.steps
    with maybe_profile(
        enabled=bool(args.profile_dir),
        profile_dir=args.profile_dir or ".",
        device=device,
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        active=args.profile_active,
        rank=get_rank(),
    ) as prof:
        for step in range(start_step, end_step):
            optimizer.zero_grad(set_to_none=True)

            with record_range("train_step"):
                with Timer(device) as step_timer:
                    if engine is not None:
                        with record_range("pipeline_1f1b"):
                            loss_value, forward_s, backward_s = _run_pipeline_step(
                                engine, dataset, config, device
                            )
                        loss_value = _sync_pipeline_loss(loss_value, device)
                    else:
                        loss_value, forward_s, backward_s = _run_non_pipeline_step(
                            model, dataset, config, device
                        )

                    lr = linear_warmup_lr(step, config)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                    with Timer(device) as optimizer_timer:
                        with record_range("optimizer_step"):
                            optimizer.step()
                    optimizer_s = optimizer_timer.elapsed_s

            profiler_step(prof)

            step_time_s = step_timer.elapsed_s
            local_tokens = config.tokens_per_step_per_rank
            global_tokens = float(local_tokens * get_data_parallel_world_size())
            avg_loss = reduce_mean(loss_value, device)
            avg_step_time_s = reduce_mean(step_time_s, device)
            avg_forward_s = reduce_mean(forward_s, device)
            avg_backward_s = reduce_mean(backward_s, device)
            avg_optimizer_s = reduce_mean(optimizer_s, device)
            global_tokens_per_second = global_tokens / avg_step_time_s
            rank_tokens_per_second = local_tokens / step_time_s

            entry: dict[str, float | int] = {
                "step": step,
                "rank": get_rank(),
                "world_size": get_world_size(),
                "tensor_parallel_size": get_tensor_model_parallel_world_size(),
                "pipeline_parallel_size": get_pipeline_model_parallel_world_size(),
                "data_parallel_size": get_data_parallel_world_size(),
                "num_microbatches": config.num_microbatches if use_pipeline else 1,
                "loss": avg_loss,
                "lr": lr,
                "step_time_ms": avg_step_time_s * 1000.0,
                "forward_ms": avg_forward_s * 1000.0,
                "backward_ms": avg_backward_s * 1000.0,
                "optimizer_ms": avg_optimizer_s * 1000.0,
                "rank_tokens_per_second": rank_tokens_per_second,
                "global_tokens_per_second": global_tokens_per_second,
            }
            metrics.append(entry)

            if is_main_process() and step % config.log_interval == 0:
                print(json.dumps(entry))

            if args.checkpoint_dir:
                should_save = step == end_step - 1
                if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                    should_save = True
                if should_save:
                    ckpt_path = save_checkpoint(
                        args.checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        config=config,
                    )
                    if is_main_process():
                        print(json.dumps({"event": "checkpoint", "path": str(ckpt_path), "step": step}))

    summary = summarize_metrics(metrics, args.warmup_discard)
    if is_main_process():
        print(json.dumps({"event": "summary", **summary}))

    if args.metrics_path and is_main_process():
        with open(args.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": asdict(config),
                    "world_size": get_world_size(),
                    "backend": actual_backend,
                    "parameter_count": parameter_count,
                    "environment": environment,
                    "summary": summary,
                    "metrics": metrics,
                },
                handle,
                indent=2,
            )

    destroy_model_parallel()
    cleanup_distributed()


if __name__ == "__main__":
    main()
