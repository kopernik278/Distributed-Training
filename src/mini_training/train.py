from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

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
    reduce_sum,
    set_seed,
)
from .model import MiniTransformerLM, cross_entropy_loss, model_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 DDP training baseline")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
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
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--metrics-path", type=str, default="")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
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
        log_interval=args.log_interval,
    )


def linear_warmup_lr(step: int, config: TrainingConfig) -> float:
    if config.warmup_steps <= 0:
        return config.lr
    scale = min((step + 1) / config.warmup_steps, 1.0)
    return config.lr * scale


def main() -> None:
    args = parse_args()
    config = build_config(args)
    device, _, actual_backend = init_distributed(config.backend)
    set_seed(config.seed)

    model = MiniTransformerLM(config).to(device)
    parameter_count = model_parameter_count(model)
    if is_distributed():
        ddp_kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    dataset = RandomTokenDataset(config, device)

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
                }
            )
        )

    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)

        with Timer(device) as timer:
            loss_value = 0.0
            for micro_step in range(config.grad_accum_steps):
                inputs, targets = dataset.next_batch()
                logits = model(inputs)
                loss = cross_entropy_loss(logits, targets) / config.grad_accum_steps
                loss.backward()
                loss_value += loss.item()

            lr = linear_warmup_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()

        step_time_s = timer.elapsed_s
        local_tokens = config.batch_size * config.seq_len * config.grad_accum_steps
        global_tokens = reduce_sum(float(local_tokens), device)
        avg_loss = reduce_mean(loss_value, device)
        avg_step_time_s = reduce_mean(step_time_s, device)
        global_tokens_per_second = global_tokens / avg_step_time_s
        rank_tokens_per_second = local_tokens / step_time_s

        entry: dict[str, float | int] = {
            "step": step,
            "rank": get_rank(),
            "world_size": get_world_size(),
            "loss": avg_loss,
            "lr": lr,
            "step_time_ms": avg_step_time_s * 1000.0,
            "rank_tokens_per_second": rank_tokens_per_second,
            "global_tokens_per_second": global_tokens_per_second,
        }
        metrics.append(entry)

        if is_main_process() and step % config.log_interval == 0:
            print(json.dumps(entry))

    if args.metrics_path and is_main_process():
        with open(args.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": asdict(config),
                    "world_size": get_world_size(),
                    "backend": actual_backend,
                    "parameter_count": parameter_count,
                    "metrics": metrics,
                },
                handle,
                indent=2,
            )

    cleanup_distributed()


if __name__ == "__main__":
    main()
