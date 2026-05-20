from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir


LOGGER = logging.getLogger("pairwise_tuner")


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(to_jsonable(v) for v in value)
    if hasattr(value, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def read_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected exactly 2 columns, got {len(row)}"
                )
            left = row[0].strip()
            right = row[1].strip()
            pairs.append((left, right))

    return pairs


def split_pairs(
    pairs: list[tuple[str, str]],
    test_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in the open interval (0, 1)")

    if len(pairs) < 2:
        raise ValueError("Need at least two pairs to perform a split")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    test_count = max(1, int(math.floor(len(shuffled) * test_fraction)))
    test_count = min(test_count, len(shuffled) - 1)

    test_pairs = shuffled[:test_count]
    train_pairs = shuffled[test_count:]

    if not train_pairs:
        raise ValueError("Training split is empty after splitting")

    return train_pairs, test_pairs


def pairwise_nll_loss(preferred_nll: torch.Tensor, rejected_nll: torch.Tensor) -> torch.Tensor:
    return F.softplus(preferred_nll - rejected_nll)


def normalize_token_ids(raw: Any) -> list[int] | None:
    if raw is None:
        return None

    if hasattr(raw, "input_ids"):
        raw = raw.input_ids

    if isinstance(raw, dict):
        if "input_ids" in raw:
            raw = raw["input_ids"]
        elif raw:
            raw = next(iter(raw.values()))
        else:
            return None

    if torch.is_tensor(raw):
        raw = raw.detach().cpu().tolist()

    if hasattr(raw, "tolist") and not isinstance(raw, (list, tuple, str, bytes)):
        try:
            raw = raw.tolist()
        except Exception:
            pass

    if isinstance(raw, tuple):
        raw = list(raw)

    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise ValueError("Tokenizer returned multiple sequences for one sentence")
        raw = raw[0]

    if isinstance(raw, list) and all(isinstance(x, int) for x in raw):
        return [int(x) for x in raw]

    return None


def encode_sentence(tokenizer: Any, sentence: str) -> list[int]:
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((sentence,), {"prepend": "<|bos|>"}),
        ((sentence,), {"add_special_tokens": True}),
        ((sentence,), {}),
    ]

    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            encoded = tokenizer(*args, **kwargs)
            token_ids = normalize_token_ids(encoded)
            if token_ids is not None:
                return token_ids
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc

    raise TypeError(
        f"Could not encode sentence with tokenizer type {type(tokenizer)!r}"
    ) from last_error


class SentenceEncodingCache:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._cache: dict[str, list[int]] = {}

    def encode(self, sentence: str) -> list[int]:
        cached = self._cache.get(sentence)
        if cached is not None:
            return cached

        ids = encode_sentence(self._tokenizer, sentence)
        self._cache[sentence] = ids
        return ids

    def warmup(self, sentences: Iterable[str]) -> None:
        for sentence in sentences:
            if sentence not in self._cache:
                self._cache[sentence] = encode_sentence(self._tokenizer, sentence)


@dataclass(slots=True)
class TrainRunConfig:
    train_pairs_path: Path = Path("data/datasets/devel.tsv")
    test_pairs_path: Path | None = None

    source: str = "base"
    hf_path: str | None = None
    model_tag: str | None = "d4"
    step: int | None = None

    test_fraction: float = 0.05
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0

    max_steps: int = 300
    eval_every: int = 50
    save_every: int = 100

    seed: int = 1337
    device_type: str = "auto"

    save_dir: Path = field(
        default_factory=lambda: Path(get_base_dir()) / "base_checkpoints" / "fine_tuned"
    )

    def __post_init__(self) -> None:
        self.train_pairs_path = Path(self.train_pairs_path)
        if self.test_pairs_path is not None:
            self.test_pairs_path = Path(self.test_pairs_path)
        self.save_dir = Path(self.save_dir)


@dataclass(slots=True)
class LoadedBackend:
    model: Any
    tokenizer: Any
    name: str
    kind: str
    meta: dict[str, Any]


def parse_args() -> TrainRunConfig:
    parser = argparse.ArgumentParser(
        description="Pairwise preference training on sentence pairs."
    )
    parser.add_argument(
        "--train-pairs-path",
        type=Path,
        default=Path("data/datasets/devel.tsv"),
        help="TSV file containing training pairs, one pair per line.",
    )
    parser.add_argument(
        "--test-pairs-path",
        type=Path,
        default=None,
        help=(
            "Optional TSV file containing evaluation pairs. "
            "If set, no train/test split is performed."
        ),
    )
    parser.add_argument("--source", type=str, default="base")
    parser.add_argument("--hf-path", type=str, default=None)
    parser.add_argument("--model-tag", type=str, default="d4")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device-type",
        type=str,
        default="auto",
        help="cuda, cpu, or auto.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(get_base_dir()) / "base_checkpoints" / "fine_tuned",
    )

    ns = parser.parse_args()
    return TrainRunConfig(
        train_pairs_path=ns.train_pairs_path,
        test_pairs_path=ns.test_pairs_path,
        source=ns.source,
        hf_path=ns.hf_path,
        model_tag=ns.model_tag,
        step=ns.step,
        test_fraction=ns.test_fraction,
        batch_size=ns.batch_size,
        lr=ns.lr,
        weight_decay=ns.weight_decay,
        max_steps=ns.max_steps,
        eval_every=ns.eval_every,
        save_every=ns.save_every,
        seed=ns.seed,
        device_type=ns.device_type,
        save_dir=ns.save_dir,
    )


def validate_config(cfg: TrainRunConfig) -> None:
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.lr <= 0:
        raise ValueError("lr must be positive")
    if cfg.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if cfg.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if cfg.save_every <= 0:
        raise ValueError("save_every must be positive")
    if cfg.device_type.strip().lower() not in {"auto", "cpu", "cuda", ""}:
        raise ValueError("device_type must be one of: auto, cpu, cuda")
    if cfg.test_pairs_path is None and not 0.0 < cfg.test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1) when splitting is used")


def resolve_device_type(requested: str) -> str:
    if requested.strip().lower() in {"", "auto"}:
        return autodetect_device_type()
    return requested


def load_backend(cfg: TrainRunConfig, device: Any) -> LoadedBackend:
    if cfg.hf_path is not None:
        from scripts.base_eval import load_hf_model

        model, tokenizer = load_hf_model(cfg.hf_path, device)
        return LoadedBackend(
            model=model,
            tokenizer=tokenizer,
            name=cfg.hf_path,
            kind="hf",
            meta={"hf_path": cfg.hf_path},
        )

    model, tokenizer, meta = load_model(
        cfg.source,
        device,
        phase="train",
        model_tag=cfg.model_tag,
        step=cfg.step,
    )

    loaded_step = meta.get("step", cfg.step or "latest") if isinstance(meta, dict) else (
        cfg.step or "latest"
    )

    return LoadedBackend(
        model=model,
        tokenizer=tokenizer,
        name=f"{cfg.source}_model (step {loaded_step})",
        kind="nanochat",
        meta=meta if isinstance(meta, dict) else {},
    )


def unwrap_trainable(model: Any) -> Any:
    return model.model if hasattr(model, "model") else model


def model_device(model: Any, fallback: Any) -> Any:
    if hasattr(model, "get_device"):
        try:
            return model.get_device()
        except Exception:
            pass

    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return fallback
        except Exception:
            pass

    return fallback


def save_tokenizer(tokenizer: Any, output_dir: Path) -> None:
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "save_pretrained"):
        tokenizer.tokenizer.save_pretrained(output_dir)
        return
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)
        return
    raise AttributeError("Tokenizer does not support save_pretrained")


class PairwisePreferenceTrainer:
    def __init__(self, cfg: TrainRunConfig) -> None:
        self.cfg = cfg
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)

        self.device_type = resolve_device_type(cfg.device_type)
        _ddp, _rank, _local_rank, world_size, device = compute_init(self.device_type)
        if world_size != 1:
            raise NotImplementedError("This script currently supports single-process training only")

        self.device = device
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        self.backend = load_backend(cfg, device)
        self.model = unwrap_trainable(self.backend.model)
        self.model = self.model.to(device) if hasattr(self.model, "to") else self.model
        self.runtime_device = model_device(self.model, device)
        self.model.train()

        self.train_pairs, self.eval_pairs = self._load_data()
        self.encoding = SentenceEncodingCache(self.backend.tokenizer)
        self.encoding.warmup(self._all_sentences())

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        self.train_order = list(range(len(self.train_pairs)))
        self.cursor = 0
        self._reshuffle()

        self.best_eval_acc = float("-inf")

    def _load_data(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        train_pairs = read_tsv_pairs(self.cfg.train_pairs_path)
        if not train_pairs:
            raise ValueError(f"No training pairs found in {self.cfg.train_pairs_path}")

        if self.cfg.test_pairs_path is not None:
            eval_pairs = read_tsv_pairs(self.cfg.test_pairs_path)
            return train_pairs, eval_pairs

        return split_pairs(train_pairs, self.cfg.test_fraction, self.cfg.seed)

    def _all_sentences(self) -> list[str]:
        seen: set[str] = set()
        sentences: list[str] = []

        for left, right in self.train_pairs + self.eval_pairs:
            if left not in seen:
                seen.add(left)
                sentences.append(left)
            if right not in seen:
                seen.add(right)
                sentences.append(right)

        return sentences

    def _reshuffle(self) -> None:
        self.rng.shuffle(self.train_order)
        self.cursor = 0

    def _next_batch(self) -> list[tuple[str, str]]:
        batch_indices: list[int] = []
        for _ in range(self.cfg.batch_size):
            if self.cursor >= len(self.train_order):
                self._reshuffle()
            batch_indices.append(self.train_order[self.cursor])
            self.cursor += 1
        return [self.train_pairs[i] for i in batch_indices]

    def sentence_nll(self, sentence: str) -> torch.Tensor:
        ids = self.encoding.encode(sentence)

        if len(ids) < 2:
            return torch.tensor(float("inf"), dtype=torch.float32, device=self.runtime_device)

        x = torch.tensor(ids[:-1], dtype=torch.long, device=self.runtime_device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=self.runtime_device).unsqueeze(0)

        out = self.model(x, y)
        if not torch.is_tensor(out):
            out = torch.tensor(out, dtype=torch.float32, device=self.runtime_device)

        return out.float().mean()

    def batch_metrics(self, pairs: list[tuple[str, str]]) -> tuple[torch.Tensor, int, int]:
        if not pairs:
            raise ValueError("Empty batch")

        preferred = torch.stack([self.sentence_nll(a) for a, _ in pairs])
        rejected = torch.stack([self.sentence_nll(b) for _, b in pairs])

        loss = pairwise_nll_loss(preferred, rejected).mean()
        correct = int((preferred < rejected).sum().item())
        return loss, correct, len(pairs)

    def evaluate(self) -> tuple[float, float]:
        if not self.eval_pairs:
            return 0.0, 0.0

        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        with torch.inference_mode():
            for start in range(0, len(self.eval_pairs), self.cfg.batch_size):
                batch = self.eval_pairs[start : start + self.cfg.batch_size]
                loss, correct, count = self.batch_metrics(batch)
                total_loss += loss.item() * count
                total_correct += correct
                total_count += count

        if was_training:
            self.model.train()

        avg_loss = total_loss / total_count if total_count else 0.0
        acc = total_correct / total_count if total_count else 0.0
        return avg_loss, acc

    def _checkpoint_meta(self, step: int, output_name: str) -> dict[str, Any]:
        model_config = getattr(self.backend.model, "config", None)
        if model_config is None:
            model_config = getattr(self.model, "config", None)

        meta = {
            "step": step,
            "source": self.cfg.source,
            "hf_path": self.cfg.hf_path,
            "model_tag": self.cfg.model_tag,
            "step_in": self.cfg.step,
            "pairs_path": str(self.cfg.train_pairs_path),
            "test_pairs_path": str(self.cfg.test_pairs_path) if self.cfg.test_pairs_path else None,
            "test_fraction": self.cfg.test_fraction,
            "batch_size": self.cfg.batch_size,
            "lr": self.cfg.lr,
            "weight_decay": self.cfg.weight_decay,
            "model_kind": self.backend.kind,
            "user_config": to_jsonable(self.cfg),
        }

        if self.backend.kind != "hf" and model_config is not None:
            meta["model_config"] = to_jsonable(model_config)

        return meta

    def save(self, step: int) -> None:
        """
        Match the original script's checkpoint behavior exactly.

        NanoChat:
            save_dir/model_000123.pt
            save_dir/meta_000123.json

        HF:
            save_dir/  (save_pretrained output)
            save_dir/meta.json
        """
        output_dir = self.cfg.save_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.backend.kind == "hf":
            self.model.save_pretrained(output_dir)
            save_tokenizer(self.backend.tokenizer, output_dir)
            with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(self._checkpoint_meta(step, "hf"), f, indent=2, ensure_ascii=False)
            return

        torch.save(self.model.state_dict(), output_dir / f"model_{step:06d}.pt")
        with (output_dir / f"meta_{step:06d}.json").open("w", encoding="utf-8") as f:
            json.dump(self._checkpoint_meta(step, f"model_{step:06d}"), f, indent=2, ensure_ascii=False)

    def train(self) -> None:
        LOGGER.info("Model: %s", self.backend.name)
        LOGGER.info("Train pairs: %d | Eval pairs: %d", len(self.train_pairs), len(self.eval_pairs))

        from tqdm import trange
        for step in trange(1, self.cfg.max_steps + 1):
            batch = self._next_batch()

            self.optimizer.zero_grad(set_to_none=True)
            loss, correct, count = self.batch_metrics(batch)
            loss.backward()
            self.optimizer.step()

            if step == 1 or step % 10 == 0:
                LOGGER.info(
                    "step %05d | train_loss=%.4f | train_acc=%.4f",
                    step,
                    loss.item(),
                    correct / count,
                )

            if self.eval_pairs and step % self.cfg.eval_every == 0:
                eval_loss, eval_acc = self.evaluate()
                LOGGER.info(
                    "step %05d | eval_loss=%.4f | eval_acc=%.4f",
                    step,
                    eval_loss,
                    eval_acc,
                )
                if eval_acc > self.best_eval_acc:
                    self.best_eval_acc = eval_acc
                    self.save(step)

            if step % self.cfg.save_every == 0:
                self.save(step)

        self.save(self.cfg.max_steps)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = parse_args()
    validate_config(cfg)

    trainer = PairwisePreferenceTrainer(cfg)
    try:
        trainer.train()
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()