from __future__ import annotations

import math

import torch
from torch import nn

from .config import TrainingConfig


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + attn_output

        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((config.seq_len, config.seq_len), float("-inf")),
                diagonal=1,
            ),
            persistent=False,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.dropout(x)
        mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.final_norm(x)
        return self.lm_head(x)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.reshape(-1))


def model_parameter_count(model: nn.Module) -> int:
    return sum(math.prod(param.shape) for param in model.parameters())
