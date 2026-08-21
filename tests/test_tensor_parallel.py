from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
from torch import nn

from mini_training.config import TrainingConfig
from mini_training.layers import ColumnParallelLinear, RowParallelLinear, shard_column_weight, shard_row_weight
from mini_training.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from mini_training.model import MiniTransformerLM
from mini_training.parallel_state import destroy_model_parallel, initialize_model_parallel


def _maybe_init_dist() -> None:
    if dist.is_initialized():
        return
    # Single-process group so TP utilities can run with tp_size=1 or be reused in spawn tests.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


class ParallelLinearUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        _maybe_init_dist()
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=1)

    def tearDown(self) -> None:
        destroy_model_parallel()

    def test_column_row_match_dense_mlp_tp1(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16)
        dense1 = nn.Linear(16, 32)
        dense2 = nn.Linear(32, 16)
        col = ColumnParallelLinear(16, 32, bias=True, gather_output=False)
        row = RowParallelLinear(32, 16, bias=True, input_is_parallel=True)

        with torch.no_grad():
            col.weight.copy_(dense1.weight)
            col.bias.copy_(dense1.bias)
            row.weight.copy_(dense2.weight)
            row.bias.copy_(dense2.bias)

        y_ref = dense2(torch.nn.functional.gelu(dense1(x)))
        y_tp = row(torch.nn.functional.gelu(col(x)))
        self.assertTrue(torch.allclose(y_ref, y_tp, atol=1e-5, rtol=1e-5))

    def test_mapping_roundtrip_tp1(self) -> None:
        x = torch.randn(2, 3, 8)
        y = gather_from_tensor_model_parallel_region(scatter_to_tensor_model_parallel_region(x))
        self.assertTrue(torch.allclose(x, y))
        z = reduce_from_tensor_model_parallel_region(copy_to_tensor_model_parallel_region(x))
        self.assertTrue(torch.allclose(x, z))


class TensorParallelModelSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        _maybe_init_dist()
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=1)

    def tearDown(self) -> None:
        destroy_model_parallel()

    def test_model_forward_shape_tp1(self) -> None:
        config = TrainingConfig(
            vocab_size=128,
            seq_len=16,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            tensor_parallel_size=1,
        )
        model = MiniTransformerLM(config)
        input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
        logits = model(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 16, 128))


def _tp2_worker(rank: int, world_size: int, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29532"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=2)

    torch.manual_seed(123)
    # Build a dense reference MLP once from shared initial weights.
    dense1 = nn.Linear(8, 16)
    dense2 = nn.Linear(16, 8)
    torch.manual_seed(123)
    dense1.reset_parameters()
    dense2.reset_parameters()

    # Broadcast dense weights from rank0 so both ranks share the same reference.
    for tensor in list(dense1.parameters()) + list(dense2.parameters()):
        dist.broadcast(tensor.data, src=0)

    col = ColumnParallelLinear(8, 16, bias=True, gather_output=False)
    row = RowParallelLinear(16, 8, bias=True, input_is_parallel=True)
    with torch.no_grad():
        col.weight.copy_(shard_column_weight(dense1.weight.data, rank, 2))
        col.bias.copy_(dense1.bias.data[rank * 8 : (rank + 1) * 8])
        row.weight.copy_(shard_row_weight(dense2.weight.data, rank, 2))
        row.bias.copy_(dense2.bias.data)

    x = torch.randn(2, 3, 8)
    dist.broadcast(x, src=0)

    y_ref = dense2(torch.nn.functional.gelu(dense1(x)))
    y_tp = row(torch.nn.functional.gelu(col(x)))
    max_err = (y_ref - y_tp).abs().max().item()

    # Also check gradients flow.
    loss = y_tp.square().mean()
    loss.backward()

    result_queue.put((rank, max_err, col.weight.grad is not None, row.weight.grad is not None))
    destroy_model_parallel()
    dist.destroy_process_group()


def _tp2_overlap_worker(rank: int, world_size: int, result_queue) -> None:
    """Same MLP as _tp2_worker but with comm/compute overlap enabled; compare grads to sync path."""
    from mini_training.overlap import set_overlap_enabled

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29533"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=2)

    torch.manual_seed(123)
    dense1 = nn.Linear(8, 16)
    dense2 = nn.Linear(16, 8)
    torch.manual_seed(123)
    dense1.reset_parameters()
    dense2.reset_parameters()
    for tensor in list(dense1.parameters()) + list(dense2.parameters()):
        dist.broadcast(tensor.data, src=0)

    x = torch.randn(2, 3, 8)
    dist.broadcast(x, src=0)

    def _build_tp():
        col = ColumnParallelLinear(8, 16, bias=True, gather_output=False)
        row = RowParallelLinear(16, 8, bias=True, input_is_parallel=True)
        with torch.no_grad():
            col.weight.copy_(shard_column_weight(dense1.weight.data, rank, 2))
            col.bias.copy_(dense1.bias.data[rank * 8 : (rank + 1) * 8])
            row.weight.copy_(shard_row_weight(dense2.weight.data, rank, 2))
            row.bias.copy_(dense2.bias.data)
        return col, row

    set_overlap_enabled(False)
    col_sync, row_sync = _build_tp()
    y_sync = row_sync(torch.nn.functional.gelu(col_sync(x.clone())))
    y_sync.square().mean().backward()

    set_overlap_enabled(True)
    col_ov, row_ov = _build_tp()
    y_ov = row_ov(torch.nn.functional.gelu(col_ov(x.clone())))
    y_ov.square().mean().backward()

    y_err = (y_sync.detach() - y_ov.detach()).abs().max().item()
    w_err = (col_sync.weight.grad - col_ov.weight.grad).abs().max().item()
    x_err = 0.0
    set_overlap_enabled(False)
    result_queue.put((rank, y_err, w_err, x_err))
    destroy_model_parallel()
    dist.destroy_process_group()


def _tp2_delayed_row_worker(rank: int, world_size: int, result_queue) -> None:
    """skip_bias_add + delayed reduce must match sync finalize path; pending clears on flush."""
    from mini_training.mappings import finalize_tensor_parallel_output
    from mini_training.overlap import has_pending_tp_reduces, set_overlap_enabled

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29534"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=2)

    torch.manual_seed(7)
    dense = nn.Linear(16, 8)
    torch.manual_seed(7)
    dense.reset_parameters()
    for tensor in dense.parameters():
        dist.broadcast(tensor.data, src=0)

    x = torch.randn(2, 3, 16)
    dist.broadcast(x, src=0)
    # RowParallel with input_is_parallel expects already-split last dim.
    local = 8
    x_local = x[..., rank * local : (rank + 1) * local].contiguous()

    def _build(skip_bias_add: bool) -> RowParallelLinear:
        row = RowParallelLinear(16, 8, bias=True, input_is_parallel=True, skip_bias_add=skip_bias_add)
        with torch.no_grad():
            row.weight.copy_(shard_row_weight(dense.weight.data, rank, 2))
            row.bias.copy_(dense.bias.data)
        return row

    set_overlap_enabled(False)
    row_sync = _build(skip_bias_add=False)
    y_sync = row_sync(x_local.clone())
    y_sync.square().mean().backward()

    set_overlap_enabled(True)
    row_ov = _build(skip_bias_add=True)
    y_pending = row_ov(x_local.clone())
    had_pending = has_pending_tp_reduces()
    y_ov = finalize_tensor_parallel_output(y_pending, row_ov.bias)
    still_pending = has_pending_tp_reduces()
    y_ov.square().mean().backward()

    y_err = (y_sync.detach() - y_ov.detach()).abs().max().item()
    w_err = (row_sync.weight.grad - row_ov.weight.grad).abs().max().item()
    set_overlap_enabled(False)
    result_queue.put((rank, y_err, w_err, had_pending, still_pending))
    destroy_model_parallel()
    dist.destroy_process_group()


def _tp2_block_overlap_worker(rank: int, world_size: int, result_queue) -> None:
    """Full MiniTransformerLM forward/backward: overlap path matches sync."""
    from mini_training.overlap import set_overlap_enabled

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29535"
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
    )
    input_ids = torch.randint(0, config.vocab_size, (2, config.seq_len))
    dist.broadcast(input_ids, src=0)

    def _run(overlap: bool):
        set_overlap_enabled(overlap)
        torch.manual_seed(99)
        model = MiniTransformerLM(config)
        # Align shards: broadcast all parameters from rank0 then re-shard is hard;
        # instead broadcast full state after building on each rank with same seed.
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        logits = model(input_ids)
        loss = logits.float().pow(2).mean()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return logits.detach(), grads

    logits_sync, grads_sync = _run(False)
    logits_ov, grads_ov = _run(True)
    set_overlap_enabled(False)

    y_err = (logits_sync - logits_ov).abs().max().item()
    g_err = 0.0
    for name, g in grads_sync.items():
        g_err = max(g_err, (g - grads_ov[name]).abs().max().item())
    result_queue.put((rank, y_err, g_err))
    destroy_model_parallel()
    dist.destroy_process_group()


class TensorParallelMultiProcessTests(unittest.TestCase):
    def test_column_row_mlp_matches_dense_tp2(self) -> None:
        import torch.multiprocessing as mp

        world_size = 2
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        processes = []
        for rank in range(world_size):
            process = ctx.Process(target=_tp2_worker, args=(rank, world_size, result_queue))
            process.start()
            processes.append(process)

        results = [result_queue.get() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)

        for rank, max_err, col_grad, row_grad in results:
            self.assertLess(max_err, 1e-4, msg=f"rank {rank} max_err={max_err}")
            self.assertTrue(col_grad)
            self.assertTrue(row_grad)

    def test_overlap_matches_sync_tp2(self) -> None:
        import torch.multiprocessing as mp

        world_size = 2
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        processes = []
        for rank in range(world_size):
            process = ctx.Process(target=_tp2_overlap_worker, args=(rank, world_size, result_queue))
            process.start()
            processes.append(process)

        results = [result_queue.get() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=120)
            self.assertEqual(process.exitcode, 0)

        for rank, y_err, w_err, _x_err in results:
            self.assertLess(y_err, 1e-5, msg=f"rank {rank} y_err={y_err}")
            self.assertLess(w_err, 1e-5, msg=f"rank {rank} w_err={w_err}")

    def test_delayed_row_reduce_matches_sync_tp2(self) -> None:
        import torch.multiprocessing as mp

        world_size = 2
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        processes = []
        for rank in range(world_size):
            process = ctx.Process(target=_tp2_delayed_row_worker, args=(rank, world_size, result_queue))
            process.start()
            processes.append(process)

        results = [result_queue.get() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=120)
            self.assertEqual(process.exitcode, 0)

        for rank, y_err, w_err, had_pending, still_pending in results:
            self.assertTrue(had_pending, msg=f"rank {rank} expected pending AllReduce before flush")
            self.assertFalse(still_pending, msg=f"rank {rank} pending should clear after finalize")
            self.assertLess(y_err, 1e-5, msg=f"rank {rank} y_err={y_err}")
            self.assertLess(w_err, 1e-5, msg=f"rank {rank} w_err={w_err}")

    def test_block_overlap_matches_sync_tp2(self) -> None:
        import torch.multiprocessing as mp

        world_size = 2
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        processes = []
        for rank in range(world_size):
            process = ctx.Process(target=_tp2_block_overlap_worker, args=(rank, world_size, result_queue))
            process.start()
            processes.append(process)

        results = [result_queue.get() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=180)
            self.assertEqual(process.exitcode, 0)

        for rank, y_err, g_err in results:
            self.assertLess(y_err, 1e-5, msg=f"rank {rank} y_err={y_err}")
            self.assertLess(g_err, 1e-5, msg=f"rank {rank} g_err={g_err}")


if __name__ == "__main__":
    unittest.main()
