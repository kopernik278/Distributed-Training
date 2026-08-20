from __future__ import annotations

import unittest

import torch

from mini_training.config import TrainingConfig
from mini_training.data import RandomTokenDataset
from mini_training.model import MiniTransformerLM, cross_entropy_loss, model_parameter_count
from mini_training.train import summarize_metrics


class ModelCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TrainingConfig(
            vocab_size=128,
            seq_len=16,
            batch_size=2,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            steps=3,
        )
        self.model = MiniTransformerLM(self.config)
        self.dataset = RandomTokenDataset(self.config, device=torch.device("cpu"))

    def test_output_shape(self) -> None:
        inputs, _targets = self.dataset.next_batch()
        logits = self.model(inputs)
        self.assertEqual(tuple(logits.shape), (self.config.batch_size, self.config.seq_len, self.config.vocab_size))

    def test_loss_is_finite(self) -> None:
        inputs, targets = self.dataset.next_batch()
        logits = self.model(inputs)
        loss = cross_entropy_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())

    def test_parameter_count_positive(self) -> None:
        self.assertGreater(model_parameter_count(self.model), 0)

    def test_short_run_loss_decreases(self) -> None:
        torch.manual_seed(0)
        model = MiniTransformerLM(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        inputs, targets = self.dataset.next_batch()
        losses: list[float] = []
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        self.assertLess(losses[-1], losses[0])


class MetricsSummaryTests(unittest.TestCase):
    def test_summary_discards_warmup(self) -> None:
        metrics = [
            {
                "loss": 10.0,
                "step_time_ms": 100.0,
                "forward_ms": 40.0,
                "backward_ms": 50.0,
                "optimizer_ms": 10.0,
                "rank_tokens_per_second": 10.0,
                "global_tokens_per_second": 10.0,
            },
            {
                "loss": 2.0,
                "step_time_ms": 20.0,
                "forward_ms": 8.0,
                "backward_ms": 10.0,
                "optimizer_ms": 2.0,
                "rank_tokens_per_second": 100.0,
                "global_tokens_per_second": 200.0,
            },
            {
                "loss": 4.0,
                "step_time_ms": 40.0,
                "forward_ms": 16.0,
                "backward_ms": 20.0,
                "optimizer_ms": 4.0,
                "rank_tokens_per_second": 50.0,
                "global_tokens_per_second": 100.0,
            },
        ]
        summary = summarize_metrics(metrics, warmup_discard=1)
        self.assertEqual(summary["num_steps"], 2)
        self.assertEqual(summary["avg_loss"], 3.0)
        self.assertEqual(summary["avg_global_tokens_per_second"], 150.0)


if __name__ == "__main__":
    unittest.main()
