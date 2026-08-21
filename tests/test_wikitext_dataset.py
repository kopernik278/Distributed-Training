from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mini_training.config import TrainingConfig
from mini_training.data import WikiText2Dataset, _build_vocab, _tokenize, build_dataset, ensure_wikitext2


class WikiTextHelpersTests(unittest.TestCase):
    def test_tokenize_and_vocab(self) -> None:
        words = _tokenize("Hello World hello")
        self.assertEqual(words, ["hello", "world", "hello"])
        vocab = _build_vocab(words, vocab_size=8)
        self.assertIn("<unk>", vocab)
        self.assertEqual(vocab["hello"], 2)  # after unk/pad


class WikiTextDatasetIntegrationTests(unittest.TestCase):
    def test_download_and_batch_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = ensure_wikitext2(tmp)
            self.assertTrue(path.is_file())
            config = TrainingConfig(
                vocab_size=512,
                seq_len=32,
                batch_size=2,
                dataset="wikitext2",
                data_dir=tmp,
            )
            ds = build_dataset(
                config,
                torch.device("cpu"),
                dataset_name="wikitext2",
                data_dir=tmp,
                seed=0,
                dp_rank=0,
                dp_size=1,
            )
            assert isinstance(ds, WikiText2Dataset)
            x, y = ds.next_batch()
            self.assertEqual(tuple(x.shape), (2, 32))
            self.assertEqual(tuple(y.shape), (2, 32))
            self.assertGreater(ds.num_tokens, 1000)


if __name__ == "__main__":
    unittest.main()
