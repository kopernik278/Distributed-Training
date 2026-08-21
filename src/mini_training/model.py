from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import TrainingConfig
from .layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    vocab_parallel_cross_entropy,
)
from .mappings import (
    finalize_tensor_parallel_output,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from .overlap import flush_pending_tp_reduces
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
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        *,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        ensure_divisible(hidden_size, tp_size, "hidden_size")
        ensure_divisible(num_heads, tp_size, "num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_per_partition = num_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.hidden_size_per_partition = self.num_heads_per_partition * self.head_dim
        self.sequence_parallel = sequence_parallel

        self.qkv = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            gather_output=False,
            sequence_parallel=sequence_parallel,
        )
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
            sequence_parallel=sequence_parallel,
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # With SP, ColumnParallel AllGathers sequence so QKV sees full S (local heads).
        qkv = self.qkv(x)
        batch_size, seq_len, _ = qkv.shape
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
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: int,
        dropout: float,
        *,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        ff_size = hidden_size * mlp_ratio
        ensure_divisible(ff_size, tp_size, "mlp_hidden_size")
        self.sequence_parallel = sequence_parallel

        self.fc1 = ColumnParallelLinear(
            hidden_size,
            ff_size,
            bias=True,
            gather_output=False,
            sequence_parallel=sequence_parallel,
        )
        self.fc2 = RowParallelLinear(
            ff_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
            sequence_parallel=sequence_parallel,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        *,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = TensorParallelAttention(
            hidden_size, num_heads, dropout, sequence_parallel=sequence_parallel
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = TensorParallelMLP(
            hidden_size, mlp_ratio, dropout, sequence_parallel=sequence_parallel
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # LN / residual stay in SP layout when sequence_parallel is on.
        residual = x
        attn_out = self.attn(self.ln1(x), attn_mask)
        attn_out = finalize_tensor_parallel_output(attn_out, self.attn.out_proj.bias)
        x = residual + attn_out

        residual = x
        mlp_out = self.mlp(self.ln2(x))
        mlp_out = finalize_tensor_parallel_output(mlp_out, self.mlp.fc2.bias)
        mlp_out = self.mlp.dropout(mlp_out)
        x = residual + mlp_out
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()
        ensure_divisible(config.hidden_size, tp_size, "hidden_size")
        ensure_divisible(config.num_heads, tp_size, "num_heads")
        ensure_divisible(config.hidden_size * config.mlp_ratio, tp_size, "mlp_hidden_size")

        self.sequence_parallel = bool(config.sequence_parallel) and tp_size > 1
        self.vocab_parallel = bool(config.vocab_parallel) and tp_size > 1
        if self.sequence_parallel:
            ensure_divisible(config.seq_len, tp_size, "seq_len")
        if self.vocab_parallel:
            ensure_divisible(config.vocab_size, tp_size, "vocab_size")

        if self.vocab_parallel:
            self.token_embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
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
                    sequence_parallel=self.sequence_parallel,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        if self.vocab_parallel:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                gather_output=False,
                sequence_parallel=self.sequence_parallel,
            )
            self.lm_head.weight = self.token_embeddings.weight
        else:
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
        if isinstance(module, (nn.Embedding, VocabParallelEmbedding)):
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
        if self.sequence_parallel:
            x = scatter_to_sequence_parallel_region(x)

        mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        flush_pending_tp_reduces()
        if self.vocab_parallel:
            # ColumnParallel LM head AllGathers seq when SP is on.
            x = self.final_norm(x)
            return self.lm_head(x)

        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x)
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

        self.sequence_parallel = bool(config.sequence_parallel) and tp_size > 1
        self.vocab_parallel = bool(config.vocab_parallel) and tp_size > 1
        if self.sequence_parallel:
            ensure_divisible(config.seq_len, tp_size, "seq_len")
        if self.vocab_parallel:
            ensure_divisible(config.vocab_size, tp_size, "vocab_size")

        layers_per_stage = config.num_layers // pp_size
        self.layer_start = pp_rank * layers_per_stage
        self.layer_end = self.layer_start + layers_per_stage
        self.is_first = is_pipeline_first_stage()
        self.is_last = is_pipeline_last_stage()

        if self.is_first:
            if self.vocab_parallel:
                self.token_embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            else:
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
                    sequence_parallel=self.sequence_parallel,
                )
                for _ in range(layers_per_stage)
            ]
        )

        if self.is_last:
            self.final_norm = nn.LayerNorm(config.hidden_size)
            if self.vocab_parallel:
                self.lm_head = ColumnParallelLinear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                    gather_output=False,
                    sequence_parallel=self.sequence_parallel,
                )
                if self.is_first and self.token_embeddings is not None:
                    self.lm_head.weight = self.token_embeddings.weight
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
        if isinstance(module, (nn.Embedding, VocabParallelEmbedding)):
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
            if self.sequence_parallel:
                x = scatter_to_sequence_parallel_region(x)
        else:
            if hidden_states is None:
                raise ValueError("non-first pipeline stage requires hidden_states")
            x = hidden_states
            # Full sequence length for the causal mask (SP shards only local S/tp).
            seq_len = self.config.seq_len if self.sequence_parallel else x.size(1)

        mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        if self.is_last:
            assert self.final_norm is not None
            assert self.lm_head is not None
            flush_pending_tp_reduces()
            if self.vocab_parallel:
                x = self.final_norm(x)
                return self.lm_head(x)
            if self.sequence_parallel:
                x = gather_from_sequence_parallel_region(x)
            x = self.final_norm(x)
            return self.lm_head(x)
        flush_pending_tp_reduces()
        return x


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.reshape(-1))


def compute_language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    vocab_parallel: bool,
) -> torch.Tensor:
    if vocab_parallel:
        return vocab_parallel_cross_entropy(logits, targets)
    return cross_entropy_loss(logits, targets)


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
