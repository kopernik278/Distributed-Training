from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

import torch

from .config import TrainingConfig

_WIKITEXT2_BASE = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
)
_WIKITEXT2_FILES = ("train.txt", "valid.txt", "test.txt")


class RandomTokenDataset:
    """Synthetic token generator for infra-focused training experiments.

    Pass an explicit ``seed`` so DP replicas diverge while TP peers
    (same ``dp_rank``) consume identical microbatches.
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int(seed))

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        size = (self.config.batch_size, self.config.seq_len + 1)
        if self._generator is None:
            tokens = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=size,
                device=self.device,
                dtype=torch.long,
            )
        else:
            tokens = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=size,
                device="cpu",
                dtype=torch.long,
                generator=self._generator,
            ).to(self.device)
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
        return inputs, targets


def _tokenize(text: str) -> list[str]:
    # Keep WikiText markers; lowercase for a compact word vocab.
    return re.findall(r"\S+", text.lower())


def ensure_wikitext2(data_dir: str | Path) -> Path:
    """Download WikiText-2 text splits if missing; return path to ``train.txt``."""
    root = Path(data_dir) / "wikitext-2"
    train_path = root / "train.txt"
    if train_path.is_file() and train_path.stat().st_size > 0:
        return train_path

    root.mkdir(parents=True, exist_ok=True)
    for name in _WIKITEXT2_FILES:
        dest = root / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        url = f"{_WIKITEXT2_BASE}/{name}"
        print(f"[data] downloading {url} → {dest}")
        req = urllib.request.Request(url, headers={"User-Agent": "mini-training/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
            out.write(resp.read())
    if not train_path.is_file():
        raise FileNotFoundError(f"expected {train_path} after downloading WikiText-2")
    return train_path


def _build_vocab(tokens: list[str], vocab_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    # Reserve 0=unk, 1=pad (unused in LM packing but kept stable).
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab: dict[str, int] = {"<unk>": 0, "<pad>": 1}
    for word, _ in ordered:
        if word in vocab:
            continue
        if len(vocab) >= vocab_size:
            break
        vocab[word] = len(vocab)
    return vocab


class WikiText2Dataset:
    """Packed WikiText-2 language-modeling stream (word-level vocab).

    Tokens are packed into a 1D stream; each ``next_batch`` returns contiguous
    windows. DP ranks use disjoint cursors so replicas see different data while
    TP peers (same seed/dp_rank) stay aligned.
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
        *,
        data_dir: str | Path,
        seed: int | None = None,
        dp_rank: int = 0,
        dp_size: int = 1,
    ) -> None:
        self.config = config
        self.device = device
        train_path = ensure_wikitext2(data_dir)
        text = train_path.read_text(encoding="utf-8", errors="ignore")
        words = _tokenize(text)
        if not words:
            raise RuntimeError("WikiText-2 train split produced zero tokens")

        # Vocab is deterministic from the corpus + vocab_size (shared across ranks).
        self.vocab = _build_vocab(words, config.vocab_size)
        ids = [self.vocab.get(w, 0) for w in words]
        self._tokens = torch.tensor(ids, dtype=torch.long)

        # Cursor: DP shards stride through the stream; seed adds a small jitter.
        base = int(seed or 0)
        shard = (base * 1_000_003 + dp_rank * 97) % max(len(self._tokens) - 1, 1)
        # Ensure TP peers with same dp_rank+seed start together.
        self._cursor = int(shard)
        self._dp_size = max(1, int(dp_size))
        self.num_tokens = int(self._tokens.numel())
        self.vocab_coverage = len(self.vocab)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = self.config.batch_size
        seq = self.config.seq_len
        need = bsz * (seq + 1)
        # Gather a contiguous slice with wrap-around.
        if self._cursor + need > self.num_tokens:
            piece = torch.cat(
                [
                    self._tokens[self._cursor :],
                    self._tokens[: (self._cursor + need) % self.num_tokens],
                ]
            )
        else:
            piece = self._tokens[self._cursor : self._cursor + need]
        # Advance by full windows; DP ranks skip ahead so they don't collide soon.
        self._cursor = (self._cursor + need * self._dp_size) % max(self.num_tokens, 1)
        windows = piece.view(bsz, seq + 1)
        inputs = windows[:, :-1].to(self.device).contiguous()
        targets = windows[:, 1:].to(self.device).contiguous()
        return inputs, targets


def build_dataset(
    config: TrainingConfig,
    device: torch.device,
    *,
    dataset_name: str,
    data_dir: str,
    seed: int | None,
    dp_rank: int = 0,
    dp_size: int = 1,
) -> RandomTokenDataset | WikiText2Dataset:
    name = (dataset_name or "random").lower()
    if name in {"random", "synthetic"}:
        return RandomTokenDataset(config, device, seed=seed)
    if name in {"wikitext2", "wikitext-2", "wt2"}:
        return WikiText2Dataset(
            config,
            device,
            data_dir=data_dir,
            seed=seed,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
    raise ValueError(f"unknown dataset={dataset_name!r}; use random|wikitext2")


def dataset_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
