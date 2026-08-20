from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist

from mini_training.config import TrainingConfig
from mini_training.data import RandomTokenDataset
from mini_training.model import MiniTransformerLM, cross_entropy_loss
from mini_training.parallel_state import (
    broadcast_parameters_within_dp,
    destroy_model_parallel,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_src_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    initialize_model_parallel,
)


def _dp_tp_worker(rank: int, world_size: int, dp_size: int, tp_size: int, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29541"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=tp_size)

    assert get_tensor_model_parallel_world_size() == tp_size
    assert get_data_parallel_world_size() == dp_size
    assert get_tensor_model_parallel_rank() == rank % tp_size
    assert get_data_parallel_rank() == rank // tp_size
    assert get_data_parallel_src_rank() == get_tensor_model_parallel_rank()

    # Model init seed depends on tp_rank, then DP broadcast aligns replicas.
    torch.manual_seed(7 + get_tensor_model_parallel_rank())
    config = TrainingConfig(
        vocab_size=64,
        seq_len=8,
        batch_size=2,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
    )
    model = MiniTransformerLM(config)
    # Deliberately perturb non-leader DP ranks before broadcast.
    if get_data_parallel_rank() != 0:
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.123)
    broadcast_parameters_within_dp(model)

    # Within a DP group, parameters must match after broadcast.
    for param in model.parameters():
        gathered = [torch.empty_like(param.data) for _ in range(dp_size)]
        dist.all_gather(gathered, param.data.contiguous(), group=get_data_parallel_group())
        for other in gathered[1:]:
            if not torch.allclose(gathered[0], other):
                result_queue.put((rank, False, "param_mismatch", 0.0, False, False))
                destroy_model_parallel()
                dist.destroy_process_group()
                return

    dataset = RandomTokenDataset(config, device=torch.device("cpu"), seed=1000 + get_data_parallel_rank())
    inputs, targets = dataset.next_batch()

    # TP peers must share the same microbatch; different DP ranks must differ.
    batch_payload = inputs.to(torch.float64).reshape(-1)
    world_batches = [torch.empty_like(batch_payload) for _ in range(world_size)]
    dist.all_gather(world_batches, batch_payload)

    my_tp = get_tensor_model_parallel_rank()
    my_dp = get_data_parallel_rank()
    for other_rank, other_batch in enumerate(world_batches):
        other_tp = other_rank % tp_size
        other_dp = other_rank // tp_size
        same = torch.allclose(batch_payload, other_batch)
        if other_dp == my_dp and other_tp != my_tp:
            if not same:
                result_queue.put((rank, False, "tp_batch_mismatch", 0.0, False, False))
                destroy_model_parallel()
                dist.destroy_process_group()
                return
        if other_dp != my_dp:
            if same:
                result_queue.put((rank, False, "dp_batch_identical", 0.0, False, False))
                destroy_model_parallel()
                dist.destroy_process_group()
                return

    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    finite = bool(torch.isfinite(loss).item())
    loss.backward()

    # Gradients of matching shards should be allreduced by DDP in training;
    # here we manually average within DP to validate group membership.
    synced = True
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.data.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=get_data_parallel_group())
        grad /= float(dp_size)
        # After manual average, every DP peer should hold the same averaged grad.
        check = grad.clone()
        dist.broadcast(check, src=get_data_parallel_src_rank(), group=get_data_parallel_group())
        if not torch.allclose(grad, check):
            synced = False
            break

    result_queue.put((rank, True, "ok", float(loss.item()), finite, synced))
    destroy_model_parallel()
    dist.destroy_process_group()


class DataParallelTensorParallelTests(unittest.TestCase):
    def test_dp2_tp2_groups_batches_and_grads(self) -> None:
        import torch.multiprocessing as mp

        dp_size, tp_size = 2, 2
        world_size = dp_size * tp_size
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        processes = []
        for rank in range(world_size):
            process = ctx.Process(
                target=_dp_tp_worker,
                args=(rank, world_size, dp_size, tp_size, result_queue),
            )
            process.start()
            processes.append(process)

        results = [result_queue.get() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=120)
            self.assertEqual(process.exitcode, 0)

        for rank, ok, reason, loss_value, finite, synced in results:
            self.assertTrue(ok, msg=f"rank {rank} failed: {reason}")
            self.assertTrue(finite, msg=f"rank {rank} non-finite loss={loss_value}")
            self.assertTrue(synced, msg=f"rank {rank} DP grad sync failed")
            self.assertTrue(loss_value == loss_value)  # not NaN


class DatasetSeedTests(unittest.TestCase):
    def test_same_seed_same_batch(self) -> None:
        config = TrainingConfig(vocab_size=32, seq_len=4, batch_size=2)
        a = RandomTokenDataset(config, torch.device("cpu"), seed=11)
        b = RandomTokenDataset(config, torch.device("cpu"), seed=11)
        x1, y1 = a.next_batch()
        x2, y2 = b.next_batch()
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))

    def test_different_seed_different_batch(self) -> None:
        config = TrainingConfig(vocab_size=32, seq_len=4, batch_size=2)
        a = RandomTokenDataset(config, torch.device("cpu"), seed=11)
        b = RandomTokenDataset(config, torch.device("cpu"), seed=12)
        x1, _ = a.next_batch()
        x2, _ = b.next_batch()
        self.assertFalse(torch.equal(x1, x2))


if __name__ == "__main__":
    unittest.main()
