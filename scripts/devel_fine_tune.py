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


@dataclass(slots=True)
class ExecutionSettings:
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
    save_dir: Path = field(default_factory=lambda: Path(get_base_dir()) / "base_checkpoints" / "fine_tuned")

    def __post_init__(self) -> None:
        self.train_pairs_path = Path(self.train_pairs_path)
        if self.test_pairs_path is not None:
            self.test_pairs_path = Path(self.test_pairs_path)
        self.save_dir = Path(self.save_dir)


@dataclass(slots=True)
class ModelContainer:
    model: Any
    tokenizer: Any
    name: str
    kind: str
    meta: dict[str, Any]


class IdCache:
    def __init__(self, tokenizer: Any) -> None:
        self.tok = tokenizer
        self.store: dict[str, list[int]] = {}

    def fetch(self, text: str) -> list[int]:
        if text not in self.store:
            self.store[text] = run_tokenizer(self.tok, text)
        return self.store[text]

    def populate(self, texts: Iterable[str]) -> None:
        for t in texts:
            if t not in self.store:
                self.store[t] = run_tokenizer(self.tok, t)


def make_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return make_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = [make_serializable(v) for v in obj]
        return sorted(items) if isinstance(obj, set) else items
    if hasattr(obj, "__dict__"):
        return {k: make_serializable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def load_tsv(file_path: Path) -> list[tuple[str, str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    pairs = []
    with file_path.open("r", encoding="utf-8", newline="") as f:
        for idx, row in enumerate(csv.reader(f, delimiter="\t"), 1):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) != 2:
                raise ValueError(f"{file_path}:{idx}: expected exactly 2 columns, got {len(row)}")
            pairs.append((row[0].strip(), row[1].strip()))
    return pairs


def split_dataset(
    data: list[tuple[str, str]], ratio: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("test_fraction must be in the open interval (0, 1)")
    if len(data) < 2:
        raise ValueError("Need at least two pairs to perform a split")

    items = list(data)
    random.Random(seed).shuffle(items)

    n_test = min(max(1, int(math.floor(len(items) * ratio))), len(items) - 1)
    train, test = items[n_test:], items[:n_test]

    if not train:
        raise ValueError("Training split is empty after splitting")
    return train, test


def clean_tokens(res: Any) -> list[int] | None:
    if res is None:
        return None
    if hasattr(res, "input_ids"):
        res = res.input_ids
    if isinstance(res, dict):
        res = res.get("input_ids") if "input_ids" in res else (next(iter(res.values())) if res else None)
    if torch.is_tensor(res):
        res = res.detach().cpu().tolist()
    if hasattr(res, "tolist") and not isinstance(res, (list, tuple, str, bytes)):
        try:
            res = res.tolist()
        except Exception:
            pass
    if isinstance(res, tuple):
        res = list(res)
    if isinstance(res, list) and res and isinstance(res[0], list):
        if len(res) != 1:
            raise ValueError("Tokenizer returned multiple sequences for one sentence")
        res = res[0]
    if isinstance(res, list) and all(isinstance(v, int) for v in res):
        return [int(v) for v in res]
    return None


def run_tokenizer(tok: Any, text: str) -> list[int]:
    err = None
    candidates = [((text,), {"prepend": "<|bos|>"}), ((text,), {"add_special_tokens": True}), ((text,), {})]
    for args, kwargs in candidates:
        try:
            tokens = clean_tokens(tok(*args, **kwargs))
            if tokens is not None:
                return tokens
        except (TypeError, Exception) as e:
            err = e
    raise TypeError(f"Could not encode sentence with tokenizer type {type(tok)!r}") from err


def gather_arguments() -> ExecutionSettings:
    p = argparse.ArgumentParser(description="Pairwise preference training on sentence pairs.")
    p.add_argument("--train-pairs-path", type=Path, default=Path("data/datasets/devel.tsv"))
    p.add_argument("--test-pairs-path", type=Path, default=None)
    p.add_argument("--source", type=str, default="base")
    p.add_argument("--hf-path", type=str, default=None)
    p.add_argument("--model-tag", type=str, default="d4")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device-type", type=str, default="auto")
    p.add_argument("--save-dir", type=Path, default=Path(get_base_dir()) / "base_checkpoints" / "fine_tuned")

    ns = p.parse_args()
    return ExecutionSettings(**vars(ns))


def check_settings(cfg: ExecutionSettings) -> None:
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


def setup_backend(cfg: ExecutionSettings, dev: Any) -> ModelContainer:
    if cfg.hf_path is not None:
        from scripts.base_eval import load_hf_model

        m, t = load_hf_model(cfg.hf_path, dev)
        return ModelContainer(model=m, tokenizer=t, name=cfg.hf_path, kind="hf", meta={"hf_path": cfg.hf_path})

    m, t, meta = load_model(cfg.source, dev, phase="train", model_tag=cfg.model_tag, step=cfg.step)
    step_val = meta.get("step", cfg.step or "latest") if isinstance(meta, dict) else (cfg.step or "latest")
    return ModelContainer(
        model=m,
        tokenizer=t,
        name=f"{cfg.source}_model (step {step_val})",
        kind="nanochat",
        meta=meta if isinstance(meta, dict) else {},
    )


def extract_underlying(m: Any) -> Any:
    return m.model if hasattr(m, "model") else m


def locate_device(m: Any, fallback: Any) -> Any:
    if hasattr(m, "get_device"):
        try:
            return m.get_device()
        except Exception:
            pass
    if hasattr(m, "parameters"):
        try:
            return next(m.parameters()).device
        except (StopIteration, Exception):
            pass
    return fallback


def export_tokenizer(t: Any, out_dir: Path) -> None:
    if hasattr(t, "tokenizer") and hasattr(t.tokenizer, "save_pretrained"):
        t.tokenizer.save_pretrained(out_dir)
    elif hasattr(t, "save_pretrained"):
        t.save_pretrained(out_dir)
    else:
        raise AttributeError("Tokenizer does not support save_pretrained")


class PreferenceEngine:
    def __init__(self, cfg: ExecutionSettings) -> None:
        self.cfg = cfg
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)

        dtype = cfg.device_type.strip().lower()
        dev_str = autodetect_device_type() if dtype in ("", "auto") else cfg.device_type
        _, _, _, world_size, target_device = compute_init(dev_str)
        if world_size != 1:
            raise NotImplementedError("This script currently supports single-process training only")

        self.dev = target_device
        self.rand = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        self.suite = setup_backend(cfg, self.dev)
        self.net = extract_underlying(self.suite.model)
        if hasattr(self.net, "to"):
            self.net = self.net.to(self.dev)
        self.active_device = locate_device(self.net, self.dev)
        self.net.train()

        train_data = load_tsv(cfg.train_pairs_path)
        if not train_data:
            raise ValueError(f"No training pairs found in {cfg.train_pairs_path}")

        if cfg.test_pairs_path is not None:
            self.train_set, self.eval_set = train_data, load_tsv(cfg.test_pairs_path)
        else:
            self.train_set, self.eval_set = split_dataset(train_data, cfg.test_fraction, cfg.seed)

        self.cache = IdCache(self.suite.tokenizer)
        seen_strings: list[str] = []
        for pair in self.train_set + self.eval_set:
            for item in pair:
                if item not in seen_strings:
                    seen_strings.append(item)
        self.cache.populate(seen_strings)

        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.indices = list(range(len(self.train_set)))
        self.pos = 0
        self._shuffle_indices()
        self.top_acc = float("-inf")

    def _shuffle_indices(self) -> None:
        self.rand.shuffle(self.indices)
        self.pos = 0

    def _fetch_batch(self) -> list[tuple[str, str]]:
        out = []
        for _ in range(self.cfg.batch_size):
            if self.pos >= len(self.indices):
                self._shuffle_indices()
            out.append(self.train_set[self.indices[self.pos]])
            self.pos += 1
        return out

    def _nll(self, text: str) -> torch.Tensor:
        tokens = self.cache.fetch(text)
        if len(tokens) < 2:
            return torch.tensor(float("inf"), dtype=torch.float32, device=self.active_device)
        x = torch.tensor(tokens[:-1], dtype=torch.long, device=self.active_device).unsqueeze(0)
        y = torch.tensor(tokens[1:], dtype=torch.long, device=self.active_device).unsqueeze(0)
        logits = self.net(x, y)
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32, device=self.active_device)
        return logits.float().mean()

    def _compute_batch(self, batch: list[tuple[str, str]]) -> tuple[torch.Tensor, int]:
        if not batch:
            raise ValueError("Empty batch")
        p_nll = torch.stack([self._nll(a) for a, _ in batch])
        r_nll = torch.stack([self._nll(b) for _, b in batch])
        loss = F.softplus(p_nll - r_nll).mean()
        corrects = int((p_nll < r_nll).sum().item())
        return loss, corrects

    def run_eval(self) -> tuple[float, float]:
        if not self.eval_set:
            return 0.0, 0.0
        mode = self.net.training
        self.net.eval()
        loss_sum, match_sum, total = 0.0, 0, 0
        with torch.inference_mode():
            for i in range(0, len(self.eval_set), self.cfg.batch_size):
                chunk = self.eval_set[i : i + self.cfg.batch_size]
                l, c = self._compute_batch(chunk)
                loss_sum += l.item() * len(chunk)
                match_sum += c
                total += len(chunk)
        if mode:
            self.net.train()
        return (loss_sum / total if total else 0.0), (match_sum / total if total else 0.0)

    def _create_snapshot(self, step: int) -> dict[str, Any]:
        conf = getattr(self.suite.model, "config", getattr(self.net, "config", None))
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
            "model_kind": self.suite.kind,
            "user_config": make_serializable(self.cfg),
        }
        if self.suite.kind != "hf" and conf is not None:
            meta["model_config"] = make_serializable(conf)
        return meta

    def save_checkpoint(self, step: int) -> None:
        path = self.cfg.save_dir
        path.mkdir(parents=True, exist_ok=True)
        if self.suite.kind == "hf":
            self.net.save_pretrained(path)
            export_tokenizer(self.suite.tokenizer, path)
            with (path / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(self._create_snapshot(step), f, indent=2, ensure_ascii=False)
            return

        torch.save(self.net.state_dict(), path / f"model_{step:06d}.pt")
        with (path / f"meta_{step:06d}.json").open("w", encoding="utf-8") as f:
            json.dump(self._create_snapshot(step), f, indent=2, ensure_ascii=False)

    def execute(self) -> None:
        LOGGER.info("==== Core Engine Initialized ====")
        LOGGER.info(f"Model Architecture: {self.suite.name}")
        LOGGER.info(f"Dataset Stats -> Train pairs: {len(self.train_set)} | Eval pairs: {len(self.eval_set)}")

        from tqdm import trange

        for step in trange(1, self.cfg.max_steps + 1):
            samples = self._fetch_batch()
            self.opt.zero_grad(set_to_none=True)
            loss, correct = self._compute_batch(samples)
            loss.backward()
            self.opt.step()

            if step == 1 or step % 10 == 0:
                LOGGER.info(f"[Step {step:05d}] Loss: {loss.item():.4f} | Accuracy: {correct / len(samples):.4f}")

            if self.eval_set and step % self.cfg.eval_every == 0:
                v_loss, v_acc = self.run_eval()
                LOGGER.info(f"[Validation {step:05d}] Loss: {v_loss:.4f} | Accuracy: {v_acc:.4f}")
                if v_acc > self.top_acc:
                    self.top_acc = v_acc
                    self.save_checkpoint(step)

            if step % self.cfg.save_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(self.cfg.max_steps)


def start_optimization() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    )
    settings = gather_arguments()
    check_settings(settings)

    engine = PreferenceEngine(settings)
    try:
        engine.execute()
    finally:
        compute_cleanup()


if __name__ == "__main__":
    start_optimization()