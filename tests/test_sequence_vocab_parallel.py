from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
from torch import nn

from mini_training.config import TrainingConfig
from mini_training.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    shard_column_weight,
    shard_row_weight,
    shard_vocab_weight,
    vocab_parallel_cross_entropy,
)
from mini_training.mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from mini_training.model import MiniTransformerLM, compute_language_model_loss
from mini_training.parallel_state import destroy_model_parallel, initialize_model_parallel


def _maybe_init_dist() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29541")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


class SequenceParallelUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        _maybe_init_dist()
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=1)

    def tearDown(self) -> None:
        destroy_model_parallel()

    def test_seq_mapping_roundtrip_tp1(self) -> None:
        x = torch.randn(2, 8, 16)
        y = gather_from_sequence_parallel_region(scatter_to_sequence_parallel_region(x))
        self.assertTrue(torch.allclose(x, y))
        z = reduce_scatter_to_sequence_parallel_region(x)
        self.assertTrue(torch.allclose(z, x))


def _sp_mlp_worker(rank: int, world_size: int, result_queue) -> None:
    """Column+Row with SP must match non-SP TP after gathering sequence."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29542"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=2)

    torch.manual_seed(11)
    dense1 = nn.Linear(8, 16)
    dense2 = nn.Linear(16, 8)
    torch.manual_seed(11)
    dense1.reset_parameters()
    dense2.reset_parameters()
    for tensor in list(dense1.parameters()) + list(dense2.parameters()):
        dist.broadcast(tensor.data, src=0)

    x = torch.randn(2, 8, 8)  # seq=8 divisible by tp=2
    dist.broadcast(x, src=0)

    def _build(sp: bool):
        col = ColumnParallelLinear(8, 16, bias=True, gather_output=False, sequence_parallel=sp)
        row = RowParallelLinear(16, 8, bias=True, input_is_parallel=True, sequence_parallel=sp)
        with torch.no_grad():
            col.weight.copy_(shard_column_weight(dense1.weight.data, rank, 2))
            col.bias.copy_(dense1.bias.data[rank * 8 : (rank + 1) * 8])
            row.weight.copy_(shard_row_weight(dense2.weight.data, rank, 2))
            row.bias.copy_(dense2.bias.data)
        return col, row

    col_s, row_s = _build(False)
    y_sync = row_s(torch.nn.functional.gelu(col_s(x.clone())))

    col_sp, row_sp = _build(True)
    x_sp = scatter_to_sequence_parallel_region(x.clone())
    y_sp_local = row_sp(torch.nn.functional.gelu(col_sp(x_sp)))
    y_sp = gather_from_sequence_parallel_region(y_sp_local)

    y_ref = dense2(torch.nn.functional.gelu(dense1(x)))
    err_sync = (y_ref - y_sync).abs().max().item()
    err_sp = (y_ref - y_sp).abs().max().item()
    err_pair = (y_sync - y_sp).abs().max().item()

    y_sp.square().mean().backward()
    result_queue.put((rank, err_sync, err_sp, err_pair, col_sp.weight.grad is not None))
    destroy_model_parallel()
    dist.destroy_process_group()


def _vocab_ce_worker(rank: int, world_size: int, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29543"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=2)

    torch.manual_seed(3)
    logits_full = torch.randn(2, 4, 16, dtype=torch.float32)
    targets = torch.randint(0, 16, (2, 4))
    dist.broadcast(logits_full, src=0)
    dist.broadcast(targets, src=0)

    local = 8
    start = rank * local
    logits_local = logits_full[..., start : start + local].contiguous().detach().requires_grad_(True)
    loss_vp = vocab_parallel_cross_entropy(logits_local, targets)
    loss_ref = torch.nn.functional.cross_entropy(
        logits_full.reshape(-1, 16), targets.reshape(-1)
    )
    err = (loss_vp - loss_ref).abs().item()
    loss_vp.backward()
    result_queue.put((rank, err, logits_local.grad is not None))
    destroy_model_parallel()
    dist.destroy_process_group()


def _vocab_embed_worker(rank: int, world_size: int, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29544"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=2)

    torch.manual_seed(5)
    dense = nn.Embedding(16, 8)
    torch.manual_seed(5)
    dense.reset_parameters()
    dist.broadcast(dense.weight.data, src=0)

    ids = torch.randint(0, 16, (2, 6))
    dist.broadcast(ids, src=0)

    emb = VocabParallelEmbedding(16, 8)
    with torch.no_grad():
        emb.weight.copy_(shard_vocab_weight(dense.weight.data, rank, 2))

    y_vp = emb(ids)
    y_ref = dense(ids)
    err = (y_vp - y_ref).abs().max().item()
    y_vp.sum().backward()
    result_queue.put((rank, err, emb.weight.grad is not None))
    destroy_model_parallel()
    dist.destroy_process_group()


def _model_sp_vp_worker(rank: int, world_size: int, result_queue) -> None:
    """Full MiniTransformerLM: SP and/or VP forward loss is finite and matches shapes."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29545"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=2)

    config = TrainingConfig(
        vocab_size=64,
        seq_len=8,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        tensor_parallel_size=2,
        sequence_parallel=True,
        vocab_parallel=True,
    )
    torch.manual_seed(21)
    model = MiniTransformerLM(config)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    input_ids = torch.randint(0, config.vocab_size, (2, config.seq_len))
    targets = torch.randint(0, config.vocab_size, (2, config.seq_len))
    dist.broadcast(input_ids, src=0)
    dist.broadcast(targets, src=0)

    logits = model(input_ids)
    # VP logits are local vocab slice; with SP Column LM head gathers seq → full S.
    assert logits.shape[0] == 2
    assert logits.shape[1] == config.seq_len
    assert logits.shape[2] == config.vocab_size // 2
    loss = compute_language_model_loss(logits, targets, vocab_parallel=True)
    loss.backward()
    result_queue.put((rank, float(loss.detach()), torch.isfinite(loss).item()))
    destroy_model_parallel()
    dist.destroy_process_group()


class SequenceVocabMultiProcessTests(unittest.TestCase):
    def _run(self, worker, world_size: int = 2, timeout: int = 120):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        processes = []
        for rank in range(world_size):
            process = ctx.Process(target=worker, args=(rank, world_size, result_queue))
            process.start()
            processes.append(process)
        results = [result_queue.get() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=timeout)
            self.assertEqual(process.exitcode, 0)
        return results

    def test_sp_mlp_matches_dense_tp2(self) -> None:
        for rank, err_sync, err_sp, err_pair, has_grad in self._run(_sp_mlp_worker):
            self.assertLess(err_sync, 1e-4, msg=f"rank {rank} sync {err_sync}")
            self.assertLess(err_sp, 1e-4, msg=f"rank {rank} sp {err_sp}")
            self.assertLess(err_pair, 1e-5, msg=f"rank {rank} pair {err_pair}")
            self.assertTrue(has_grad)

    def test_vocab_parallel_ce_matches_dense_tp2(self) -> None:
        for rank, err, has_grad in self._run(_vocab_ce_worker):
            self.assertLess(err, 1e-5, msg=f"rank {rank} ce err={err}")
            self.assertTrue(has_grad)

    def test_vocab_parallel_embedding_matches_dense_tp2(self) -> None:
        for rank, err, has_grad in self._run(_vocab_embed_worker):
            self.assertLess(err, 1e-5, msg=f"rank {rank} embed err={err}")
            self.assertTrue(has_grad)

    def test_model_sp_vp_smoke_tp2(self) -> None:
        for rank, loss, finite in self._run(_model_sp_vp_worker):
            self.assertTrue(finite, msg=f"rank {rank} loss={loss}")


if __name__ == "__main__":
    unittest.main()
