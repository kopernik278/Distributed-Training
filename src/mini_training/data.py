from __future__ import annotations

import torch

from .config import TrainingConfig


class RandomTokenDataset:
    """Synthetic token generator for infra-focused training experiments."""

    def __init__(self, config: TrainingConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(self.config.batch_size, self.config.seq_len + 1),
            device=self.device,
            dtype=torch.long,
        )
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
        return inputs, targets
