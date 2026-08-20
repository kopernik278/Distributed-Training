from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    vocab_size: int = 4096
    seq_len: int = 256
    batch_size: int = 8
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    steps: int = 20
    warmup_steps: int = 5
    grad_accum_steps: int = 1
    seed: int = 42
    log_interval: int = 1
    backend: str = "nccl"
    tensor_parallel_size: int = 1
    # Optional explicit DP size. When None, inferred as world_size // tensor_parallel_size.
    data_parallel_size: int | None = None

    @property
    def tokens_per_step_per_rank(self) -> int:
        return self.batch_size * self.seq_len * self.grad_accum_steps
