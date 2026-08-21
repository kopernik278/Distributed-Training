from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

from mini_training.checkpoint import (
    build_param_specs,
    consolidate_tp_state_dicts,
    load_checkpoint,
    load_metadata,
    save_checkpoint,
    shard_full_state_dict,
    _load_shard,
)
from mini_training.config import TrainingConfig
from mini_training.model import MiniTransformerLM
from mini_training.parallel_state import destroy_model_parallel, initialize_model_parallel


def _init_single(port: str = "29561") -> None:
    if dist.is_initialized():
        destroy_model_parallel()
        return
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    destroy_model_parallel()
    initialize_model_parallel(1, 1)


class CheckpointSameLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        _init_single("29561")
        initialize_model_parallel(1, 1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        destroy_model_parallel()
        self.tmp.cleanup()

    def test_save_load_roundtrip_tp1(self) -> None:
        config = TrainingConfig(
            vocab_size=64,
            seq_len=8,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            steps=1,
        )
        torch.manual_seed(0)
        model = MiniTransformerLM(config)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, config.vocab_size, (2, 8))
        model(x).sum().backward()
        opt.step()

        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        step_dir = save_checkpoint(self.root, model=model, optimizer=opt, step=3, config=config)
        self.assertTrue((step_dir / "metadata.json").is_file())
        self.assertTrue((step_dir / "mp_rank_pp000_tp000.pt").is_file())

        torch.manual_seed(99)
        model2 = MiniTransformerLM(config)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        loaded_step, meta, opt_loaded = load_checkpoint(step_dir, model=model2, optimizer=opt2)
        self.assertEqual(loaded_step, 3)
        self.assertTrue(opt_loaded)
        self.assertEqual(meta["parallel"]["tensor_parallel_size"], 1)
        for key, tensor in before.items():
            self.assertTrue(torch.equal(tensor, model2.state_dict()[key]), msg=key)


def _tp_reshard_worker(rank: int, world_size: int, ckpt_dir: str, port: str, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    destroy_model_parallel()
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    config = TrainingConfig(
        vocab_size=64,
        seq_len=8,
        batch_size=2,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        tensor_parallel_size=2,
    )
    model = MiniTransformerLM(config)
    load_checkpoint(ckpt_dir, model=model, optimizer=None, map_location="cpu")

    meta = load_metadata(Path(ckpt_dir))
    saved_tp = int(meta["parallel"]["tensor_parallel_size"])
    payloads = [_load_shard(Path(ckpt_dir), 0, t, "cpu") for t in range(saved_tp)]
    source_specs = payloads[0]["param_specs"]
    full = consolidate_tp_state_dicts([p["model"] for p in payloads], source_specs)
    dest_specs = build_param_specs(model)
    expected_local = shard_full_state_dict(full, dest_specs, tp_rank=rank, tp_size=2)

    max_err = 0.0
    for key, tensor in model.state_dict().items():
        err = (tensor.cpu() - expected_local[key]).abs().max().item()
        max_err = max(max_err, err)

    result_queue.put((rank, float(max_err)))
    destroy_model_parallel()
    dist.barrier()
    dist.destroy_process_group()


class CheckpointTPReshardTests(unittest.TestCase):
    def test_tp1_checkpoint_reshards_to_tp2(self) -> None:
        import socket
        import torch.multiprocessing as mp

        # Parent must not hold an active process group while spawning children.
        if dist.is_initialized():
            destroy_model_parallel()
            dist.destroy_process_group()

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
        sock.close()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(int(port) + 1)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        initialize_model_parallel(1, 1)

        config = TrainingConfig(
            vocab_size=64,
            seq_len=8,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            tensor_parallel_size=1,
        )
        torch.manual_seed(1)
        model = MiniTransformerLM(config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step_dir = save_checkpoint(root, model=model, optimizer=None, step=7, config=config)

            destroy_model_parallel()
            dist.destroy_process_group()

            ctx = mp.get_context("spawn")
            result_queue = ctx.SimpleQueue()
            processes = []
            for rank in range(2):
                process = ctx.Process(
                    target=_tp_reshard_worker,
                    args=(rank, 2, str(step_dir), port, result_queue),
                )
                process.start()
                processes.append(process)
            results = [result_queue.get() for _ in range(2)]
            for process in processes:
                process.join(timeout=120)
                self.assertEqual(process.exitcode, 0)
            for rank, max_err in results:
                self.assertLess(max_err, 1e-6, msg=f"rank {rank} max_err={max_err}")

    def test_consolidate_shard_helpers_tp2(self) -> None:
        _init_single("29564")
        initialize_model_parallel(1, 1)
        full = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        specs = {"w": {"tensor_model_parallel": True, "partition_dim": 0}}
        shards = [
            shard_full_state_dict({"w": full}, specs, tp_rank=0, tp_size=2),
            shard_full_state_dict({"w": full}, specs, tp_rank=1, tp_size=2),
        ]
        merged = consolidate_tp_state_dicts(shards, specs)
        self.assertTrue(torch.equal(merged["w"], full))


if __name__ == "__main__":
    unittest.main()
