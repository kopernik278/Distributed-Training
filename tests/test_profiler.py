from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mini_training.profiler import maybe_profile, profiler_step, record_range


class ProfilerTests(unittest.TestCase):
    def test_disabled_is_noop(self) -> None:
        with maybe_profile(
            enabled=False,
            profile_dir="/tmp/unused",
            device=torch.device("cpu"),
            wait=0,
            warmup=0,
            active=1,
            rank=0,
        ) as prof:
            self.assertIsNone(prof)
            with record_range("train_step"):
                x = torch.ones(2, 2)
                _ = x @ x
            profiler_step(prof)

    def test_enabled_writes_trace_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prof"
            with maybe_profile(
                enabled=True,
                profile_dir=out,
                device=torch.device("cpu"),
                wait=0,
                warmup=1,
                active=1,
                rank=0,
            ) as prof:
                self.assertIsNotNone(prof)
                with record_range("train_step"):
                    x = torch.randn(8, 8, requires_grad=True)
                    y = x @ x
                    y.sum().backward()
                profiler_step(prof)
                # Need one extra step so schedule (wait=0,warmup=1,active=1) exports.
                with record_range("train_step"):
                    x = torch.randn(8, 8, requires_grad=True)
                    (x @ x).sum().backward()
                profiler_step(prof)

            trace = out / "rank0.json"
            summary = out / "rank0_summary.txt"
            self.assertTrue(trace.is_file(), msg="chrome trace missing")
            self.assertGreater(trace.stat().st_size, 100)
            self.assertTrue(summary.is_file())
            text = summary.read_text(encoding="utf-8")
            self.assertTrue(len(text) > 10)


if __name__ == "__main__":
    unittest.main()
