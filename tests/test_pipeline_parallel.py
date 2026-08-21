from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist

from mini_training.config import TrainingConfig
from mini_training.model import MiniTransformerLM, PipelineStage, cross_entropy_loss
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


def _copy_dense_into_pipeline_stage(dense: MiniTransformerLM, stage: PipelineStage) -> None:
    """Copy matching dense weights into a pipeline stage (pp split of the same model)."""
    with torch.no_grad():
        if stage.is_first:
            assert stage.token_embeddings is not None
            assert stage.position_embeddings is not None
            stage.token_embeddings.weight.copy_(dense.token_embeddings.weight)
            stage.position_embeddings.weight.copy_(dense.position_embeddings.weight)
        for local_i, block in enumerate(stage.blocks):
            dense_block = dense.blocks[stage.layer_start + local_i]
            block.load_state_dict(dense_block.state_dict())
        if stage.is_last:
            assert stage.final_norm is not None
            assert stage.lm_head is not None
            stage.final_norm.load_state_dict(dense.final_norm.state_dict())
            # Dense ties lm_head ↔ token_embeddings; PP last stage has a separate head.
            stage.lm_head.weight.copy_(dense.lm_head.weight)


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


def _pp_vs_dense_worker(rank: int, world_size: int, pp_size: int, result_queue) -> None:
    """Compare PP=2 1F1B loss/grads against a single-process dense MiniTransformerLM."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29553"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

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

    # Build dense reference once, broadcast all parameters so every rank can shard locally.
    # Untie lm_head ↔ embeddings: PP places them on different stages, so a tied dense
    # reference would mix embedding-path grads into the head (and vice versa).
    destroy_model_parallel()
    initialize_model_parallel(1, 1)
    torch.manual_seed(7)
    dense = MiniTransformerLM(config)
    with torch.no_grad():
        dense.lm_head.weight = torch.nn.Parameter(dense.token_embeddings.weight.detach().clone())
    for param in dense.parameters():
        dist.broadcast(param.data, src=0)
    dense_state = {k: v.detach().clone() for k, v in dense.state_dict().items()}

    # Rebuild PP groups and local stage; load matching dense shard.
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=pp_size)

    dense_ref = MiniTransformerLM(config)
    with torch.no_grad():
        dense_ref.lm_head.weight = torch.nn.Parameter(dense_ref.token_embeddings.weight.detach().clone())
    dense_ref.load_state_dict(dense_state)

    stage = PipelineStage(config)
    _copy_dense_into_pipeline_stage(dense_ref, stage)

    g = torch.Generator()
    g.manual_seed(321)
    microbatches: list[MicrobatchIO] = []
    for _ in range(config.num_microbatches):
        tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len + 1), generator=g)
        microbatches.append(
            MicrobatchIO(input_ids=tokens[:, :-1].contiguous(), targets=tokens[:, 1:].contiguous())
        )

    # Dense reference: same microbatches, mean CE (matches engine's /M scaling sum).
    dense_ref.zero_grad(set_to_none=True)
    ref_loss_sum = 0.0
    for mb in microbatches:
        logits = dense_ref(mb.input_ids)
        loss = cross_entropy_loss(logits, mb.targets) / config.num_microbatches
        loss.backward()
        ref_loss_sum += float(loss.detach().item())

    engine = PipelineEngine(
        stage,
        num_microbatches=config.num_microbatches,
        hidden_size=config.hidden_size,
        dtype=torch.float32,
    )
    stage.zero_grad(set_to_none=True)
    pp_loss = engine.run(microbatches, device=torch.device("cpu"))

    loss_err = 0.0
    max_grad_err = 0.0
    if is_pipeline_last_stage():
        loss_err = abs(pp_loss - ref_loss_sum)

    # Compare overlapping parameter grads against the untied dense reference.
    if stage.is_first:
        assert stage.token_embeddings is not None
        assert stage.position_embeddings is not None
        assert dense_ref.token_embeddings.weight.grad is not None
        assert stage.token_embeddings.weight.grad is not None
        max_grad_err = max(
            max_grad_err,
            (stage.token_embeddings.weight.grad - dense_ref.token_embeddings.weight.grad)
            .abs()
            .max()
            .item(),
        )
        max_grad_err = max(
            max_grad_err,
            (stage.position_embeddings.weight.grad - dense_ref.position_embeddings.weight.grad)
            .abs()
            .max()
            .item(),
        )
        for local_i, block in enumerate(stage.blocks):
            dense_block = dense_ref.blocks[stage.layer_start + local_i]
            for p1, p2 in zip(block.parameters(), dense_block.parameters()):
                if p1.grad is None or p2.grad is None:
                    continue
                max_grad_err = max(max_grad_err, (p1.grad - p2.grad).abs().max().item())

    if stage.is_last:
        assert stage.final_norm is not None
        assert stage.lm_head is not None
        assert dense_ref.final_norm.weight.grad is not None
        assert stage.final_norm.weight.grad is not None
        assert dense_ref.lm_head.weight.grad is not None
        assert stage.lm_head.weight.grad is not None
        max_grad_err = max(
            max_grad_err,
            (stage.final_norm.weight.grad - dense_ref.final_norm.weight.grad).abs().max().item(),
        )
        max_grad_err = max(
            max_grad_err,
            (stage.final_norm.bias.grad - dense_ref.final_norm.bias.grad).abs().max().item(),
        )
        max_grad_err = max(
            max_grad_err,
            (stage.lm_head.weight.grad - dense_ref.lm_head.weight.grad).abs().max().item(),
        )
        for local_i, block in enumerate(stage.blocks):
            dense_block = dense_ref.blocks[stage.layer_start + local_i]
            for p1, p2 in zip(block.parameters(), dense_block.parameters()):
                if p1.grad is None or p2.grad is None:
                    continue
                max_grad_err = max(max_grad_err, (p1.grad - p2.grad).abs().max().item())

    result_queue.put(
        (
            rank,
            float(pp_loss) if is_pipeline_last_stage() else 0.0,
            float(ref_loss_sum),
            float(loss_err),
            float(max_grad_err),
        )
    )
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

    def test_pp2_matches_dense_loss_and_grads(self) -> None:
        import torch.multiprocessing as mp

        pp_size = 2
        world_size = pp_size
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        processes = []
        for rank in range(world_size):
            process = ctx.Process(
                target=_pp_vs_dense_worker,
                args=(rank, world_size, pp_size, result_queue),
            )
            process.start()
            processes.append(process)

        results = [result_queue.get() for _ in range(world_size)]
        for process in processes:
            process.join(timeout=180)
            self.assertEqual(process.exitcode, 0)

        for rank, pp_loss, ref_loss, loss_err, grad_err in results:
            self.assertLess(grad_err, 1e-4, msg=f"rank {rank} max_grad_err={grad_err}")
            if rank == pp_size - 1:
                self.assertLess(loss_err, 1e-5, msg=f"loss_err={loss_err} pp={pp_loss} ref={ref_loss}")

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
