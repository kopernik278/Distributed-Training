from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist

from mini_training.config import TrainingConfig
from mini_training.model import PipelineStage, cross_entropy_loss
from mini_training.parallel_state import (
    destroy_model_parallel,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_world_size,
    initialize_model_parallel,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from mini_training.pipeline import MicrobatchIO, PipelineEngine


def _pp_worker(rank: int, world_size: int, pp_size: int, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29551"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=pp_size)

    assert get_pipeline_model_parallel_world_size() == pp_size
    assert get_pipeline_model_parallel_rank() == rank
    assert get_tensor_model_parallel_world_size() == 1
    if rank == 0:
        assert is_pipeline_first_stage()
        assert get_pipeline_model_parallel_prev_rank() is None
        assert get_pipeline_model_parallel_next_rank() == 1
    if rank == pp_size - 1:
        assert is_pipeline_last_stage()
        assert get_pipeline_model_parallel_next_rank() is None

    torch.manual_seed(0)
    config = TrainingConfig(
        vocab_size=64,
        seq_len=8,
        batch_size=2,
        hidden_size=32,
        num_layers=4,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        pipeline_parallel_size=pp_size,
        num_microbatches=4,
        tensor_parallel_size=1,
    )
    stage = PipelineStage(config)
    engine = PipelineEngine(
        stage,
        num_microbatches=config.num_microbatches,
        hidden_size=config.hidden_size,
        dtype=torch.float32,
    )

    # Shared microbatches across the pipeline (same seed on every rank).
    g = torch.Generator()
    g.manual_seed(123)
    microbatches: list[MicrobatchIO] = []
    for _ in range(config.num_microbatches):
        tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len + 1), generator=g)
        microbatches.append(
            MicrobatchIO(input_ids=tokens[:, :-1].contiguous(), targets=tokens[:, 1:].contiguous())
        )

    optimizer = torch.optim.SGD(stage.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    loss = engine.run(microbatches, device=torch.device("cpu"))
    optimizer.step()

    finite = True
    if is_pipeline_last_stage():
        finite = bool(loss == loss) and abs(loss) < 1e6

    # Gradients should exist on local stage params after 1F1B.
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in stage.parameters())

    result_queue.put((rank, True, finite, has_grad, float(loss) if is_pipeline_last_stage() else 0.0))
    destroy_model_parallel()
    dist.destroy_process_group()


class PipelineParallelTests(unittest.TestCase):
    def test_pp2_1f1b_smoke(self) -> None:
        import torch.multiprocessing as mp

        pp_size = 2
        world_size = pp_size
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        processes = []
        for rank in range(world_size):
            process = ctx.Process(target=_pp_worker, args=(rank, world_size, pp_size, result_queue))
            process.start()
            processes.append(process)

        results = [result_queue.get() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=120)
            self.assertEqual(process.exitcode, 0)

        for rank, ok, finite, has_grad, loss in results:
            self.assertTrue(ok, msg=f"rank {rank} failed")
            self.assertTrue(has_grad, msg=f"rank {rank} missing grads")
            if rank == pp_size - 1:
                self.assertTrue(finite, msg=f"last-stage loss not finite: {loss}")

    def test_pipeline_stage_layer_split(self) -> None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29552")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        destroy_model_parallel()
        initialize_model_parallel(1, 1)
        config = TrainingConfig(num_layers=4, hidden_size=32, num_heads=4, mlp_ratio=2, vocab_size=32, seq_len=8)
        stage = PipelineStage(config)
        self.assertTrue(stage.is_first and stage.is_last)
        self.assertEqual(len(stage.blocks), 4)
        ids = torch.randint(0, 32, (2, 8))
        logits = stage(input_ids=ids)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        loss = cross_entropy_loss(logits, ids)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
