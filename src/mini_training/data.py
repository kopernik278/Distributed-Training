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

# WikiText-103 is published on HuggingFace as parquet shards (text column).
_WIKITEXT103_PARQUET_URLS = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet",
)


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


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {url} → {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "mini-training/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


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
        _download(url, dest)
    if not train_path.is_file():
        raise FileNotFoundError(f"expected {train_path} after downloading WikiText-2")
    return train_path


def ensure_wikitext103(data_dir: str | Path) -> Path:
    """Download WikiText-103-raw train text; return path to ``wiki.train.raw``."""
    root = Path(data_dir) / "wikitext-103-raw"
    train_path = root / "wiki.train.raw"
    if train_path.is_file() and train_path.stat().st_size > 0:
        return train_path

    root.mkdir(parents=True, exist_ok=True)
    parquet_dir = root / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    local_parts: list[Path] = []
    for url in _WIKITEXT103_PARQUET_URLS:
        name = url.rsplit("/", 1)[-1]
        dest = parquet_dir / name
        if not dest.is_file() or dest.stat().st_size < 1_000_000:
            _download(url, dest)
        local_parts.append(dest)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "WikiText-103 download requires pyarrow; pip install pyarrow"
        ) from exc

    print(f"[data] converting {len(local_parts)} parquet shards → {train_path}")
    with train_path.open("w", encoding="utf-8") as out:
        for part in local_parts:
            table = pq.read_table(part, columns=["text"])
            for value in table.column("text").to_pylist():
                if not value:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                out.write(text)
                out.write("\n")
    if not train_path.is_file() or train_path.stat().st_size == 0:
        raise FileNotFoundError(f"expected {train_path} after converting WikiText-103")
    return train_path


def ensure_text_corpus(dataset_name: str, data_dir: str | Path) -> Path:
    name = dataset_name.lower()
    if name in {"wikitext2", "wikitext-2", "wt2"}:
        return ensure_wikitext2(data_dir)
    if name in {"wikitext103", "wikitext-103", "wt103"}:
        return ensure_wikitext103(data_dir)
    raise ValueError(f"unknown text corpus {dataset_name!r}")


def _build_vocab(tokens: list[str], vocab_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab: dict[str, int] = {"<unk>": 0, "<pad>": 1}
    for word, _ in ordered:
        if word in vocab:
            continue
        if len(vocab) >= vocab_size:
            break
        vocab[word] = len(vocab)
    return vocab


def _build_vocab_streaming(path: Path, vocab_size: int, max_scan_chars: int | None) -> dict[str, int]:
    """Build vocab by scanning the corpus (optionally truncated for speed)."""
    counts: dict[str, int] = {}
    scanned = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            scanned += len(chunk)
            for w in _tokenize(chunk):
                counts[w] = counts.get(w, 0) + 1
            if max_scan_chars is not None and scanned >= max_scan_chars:
                break
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab: dict[str, int] = {"<unk>": 0, "<pad>": 1}
    for word, _ in ordered:
        if word in vocab:
            continue
        if len(vocab) >= vocab_size:
            break
        vocab[word] = len(vocab)
    return vocab


class PackedTextDataset:
    """Packed language-modeling stream over a real text corpus (word-level vocab).

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
        corpus: str,
        seed: int | None = None,
        dp_rank: int = 0,
        dp_size: int = 1,
        max_vocab_scan_chars: int | None = None,
        max_pack_chars: int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.corpus = corpus
        train_path = ensure_text_corpus(corpus, data_dir)

        # For WT-103, full in-memory word lists are huge; stream vocab + pack.
        default_scan = None if "103" not in corpus.replace("-", "") else 200_000_000
        scan_chars = max_vocab_scan_chars if max_vocab_scan_chars is not None else default_scan
        self.vocab = _build_vocab_streaming(train_path, config.vocab_size, scan_chars)

        pack_limit = max_pack_chars
        if pack_limit is None and "103" in corpus.replace("-", ""):
            # Enough tokens for long throughput runs without multi-GB id tensors.
            pack_limit = 80_000_000

        ids: list[int] = []
        packed_chars = 0
        with train_path.open("r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                packed_chars += len(chunk)
                for w in _tokenize(chunk):
                    ids.append(self.vocab.get(w, 0))
                if pack_limit is not None and packed_chars >= pack_limit:
                    break
        if not ids:
            raise RuntimeError(f"{corpus} train split produced zero tokens")

        self._tokens = torch.tensor(ids, dtype=torch.long)
        base = int(seed or 0)
        shard = (base * 1_000_003 + dp_rank * 97) % max(len(self._tokens) - 1, 1)
        self._cursor = int(shard)
        self._dp_size = max(1, int(dp_size))
        self.num_tokens = int(self._tokens.numel())
        self.vocab_coverage = len(self.vocab)
        self.train_path = train_path

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = self.config.batch_size
        seq = self.config.seq_len
        need = bsz * (seq + 1)
        if self._cursor + need > self.num_tokens:
            piece = torch.cat(
                [
                    self._tokens[self._cursor :],
                    self._tokens[: (self._cursor + need) % self.num_tokens],
                ]
            )
        else:
            piece = self._tokens[self._cursor : self._cursor + need]
        self._cursor = (self._cursor + need * self._dp_size) % max(self.num_tokens, 1)
        windows = piece.view(bsz, seq + 1)
        inputs = windows[:, :-1].to(self.device).contiguous()
        targets = windows[:, 1:].to(self.device).contiguous()
        return inputs, targets


class WikiText2Dataset(PackedTextDataset):
    """Back-compat alias for WikiText-2 packed LM stream."""

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
        super().__init__(
            config,
            device,
            data_dir=data_dir,
            corpus="wikitext2",
            seed=seed,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )


def build_dataset(
    config: TrainingConfig,
    device: torch.device,
    *,
    dataset_name: str,
    data_dir: str,
    seed: int | None,
    dp_rank: int = 0,
    dp_size: int = 1,
) -> RandomTokenDataset | PackedTextDataset:
    name = (dataset_name or "random").lower()
    if name in {"random", "synthetic"}:
        return RandomTokenDataset(config, device, seed=seed)
    if name in {"wikitext2", "wikitext-2", "wt2"}:
        return PackedTextDataset(
            config,
            device,
            data_dir=data_dir,
            corpus="wikitext2",
            seed=seed,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
    if name in {"wikitext103", "wikitext-103", "wt103"}:
        return PackedTextDataset(
            config,
            device,
            data_dir=data_dir,
            corpus="wikitext103",
            seed=seed,
            dp_rank=dp_rank,
            dp_size=dp_size,
        )
    raise ValueError(f"unknown dataset={dataset_name!r}; use random|wikitext2|wikitext103")


def dataset_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
