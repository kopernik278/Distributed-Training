from __future__ import annotations

import torch

from .config import TrainingConfig


class RandomTokenDataset:
    """Synthetic token generator for infra-focused training experiments.

    Pass an explicit ``seed`` so DP replicas diverge while TP peers
    (same ``dp_rank``) consume identical microbatches.
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self._generator: torch.Generator | None = None
        if seed is not None:
            # CPU generator keeps DP/TP batch identity independent of CUDA device index.
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int(seed))

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        size = (self.config.batch_size, self.config.seq_len + 1)
        if self._generator is None:
            tokens = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=size,
                device=self.device,
                dtype=torch.long,
            )
        else:
            tokens = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=size,
                device="cpu",
                dtype=torch.long,
                generator=self._generator,
            ).to(self.device)
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
        return inputs, targets
