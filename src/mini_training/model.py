from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import TrainingConfig
from .layers import ColumnParallelLinear, RowParallelLinear
from .parallel_state import (
    ensure_divisible,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)


class TensorParallelAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        ensure_divisible(hidden_size, tp_size, "hidden_size")
        ensure_divisible(num_heads, tp_size, "num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_per_partition = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.hidden_size_per_partition = self.num_heads_per_partition * self.head_dim

        self.qkv = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            gather_output=False,
        )
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads_per_partition, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + attn_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size_per_partition)
        return self.out_proj(context)


class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        ff_size = hidden_size * mlp_ratio
        ensure_divisible(ff_size, tp_size, "mlp_hidden_size")

        self.fc1 = ColumnParallelLinear(hidden_size, ff_size, bias=True, gather_output=False)
        self.fc2 = RowParallelLinear(ff_size, hidden_size, bias=True, input_is_parallel=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = TensorParallelAttention(hidden_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = TensorParallelMLP(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()
        ensure_divisible(config.hidden_size, tp_size, "hidden_size")
        ensure_divisible(config.num_heads, tp_size, "num_heads")
        ensure_divisible(config.hidden_size * config.mlp_ratio, tp_size, "mlp_hidden_size")

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
        # Keep LM head replicated in this milestone for simpler correctness/testing.
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
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear) and not hasattr(module.weight, "tensor_model_parallel"):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
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


class PipelineStage(nn.Module):
    """One pipeline stage: optional embed, a contiguous block range, optional LM head."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        pp_size = get_pipeline_model_parallel_world_size()
        pp_rank = get_pipeline_model_parallel_rank()
        ensure_divisible(config.num_layers, pp_size, "num_layers")
        tp_size = get_tensor_model_parallel_world_size()
        ensure_divisible(config.hidden_size, tp_size, "hidden_size")
        ensure_divisible(config.num_heads, tp_size, "num_heads")
        ensure_divisible(config.hidden_size * config.mlp_ratio, tp_size, "mlp_hidden_size")

        layers_per_stage = config.num_layers // pp_size
        self.layer_start = pp_rank * layers_per_stage
        self.layer_end = self.layer_start + layers_per_stage
        self.is_first = is_pipeline_first_stage()
        self.is_last = is_pipeline_last_stage()

        if self.is_first:
            self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embeddings = nn.Embedding(config.seq_len, config.hidden_size)
            self.embed_dropout = nn.Dropout(config.dropout)
        else:
            self.token_embeddings = None
            self.position_embeddings = None
            self.embed_dropout = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(layers_per_stage)
            ]
        )

        if self.is_last:
            self.final_norm = nn.LayerNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            # Weight tying only when embed lives on this stage (pp_size == 1).
            if self.is_first and self.token_embeddings is not None:
                self.lm_head.weight = self.token_embeddings.weight
        else:
            self.final_norm = None
            self.lm_head = None

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
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear) and not hasattr(module.weight, "tensor_model_parallel"):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_first:
            if input_ids is None:
                raise ValueError("first pipeline stage requires input_ids")
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            assert self.token_embeddings is not None
            assert self.position_embeddings is not None
            assert self.embed_dropout is not None
            x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
            x = self.embed_dropout(x)
        else:
            if hidden_states is None:
                raise ValueError("non-first pipeline stage requires hidden_states")
            x = hidden_states
            seq_len = x.size(1)

        mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        if self.is_last:
            assert self.final_norm is not None
            assert self.lm_head is not None
            x = self.final_norm(x)
            return self.lm_head(x)
        return x


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.reshape(-1))


def model_parameter_count(model: nn.Module) -> int:
    # Local parameter count on this rank (sharded params counted locally).
    return sum(math.prod(param.shape) for param in model.parameters())


def load_tp_shards_from_full_linear(
    full: nn.Linear,
    column: ColumnParallelLinear | None = None,
    row: RowParallelLinear | None = None,
) -> None:
    """Helper used by tests to copy dense weights into TP shards."""
    from .layers import shard_column_weight, shard_row_weight

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    with torch.no_grad():
        if column is not None:
            column.weight.copy_(shard_column_weight(full.weight.data, tp_rank, tp_size))
            if column.bias is not None and full.bias is not None:
                out = full.bias.numel()
                local = out // tp_size
                start = tp_rank * local
                column.bias.copy_(full.bias.data[start : start + local])
        if row is not None:
            row.weight.copy_(shard_row_weight(full.weight.data, tp_rank, tp_size))
            if row.bias is not None and full.bias is not None:
                row.bias.copy_(full.bias.data)
