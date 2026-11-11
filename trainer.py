"""DDP trainer for ThinkLoop."""

from __future__ import annotations

import contextlib
import inspect
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
import yaml
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import ThoughtDataset, load_thoughts, thought_collator
from src.metrics import report_predictions, score_predictions
from src.model import ThinkLoop


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    random.seed(torch.initial_seed() % 2**32)


def load_config(argv: list[str]) -> dict:
    if not argv:
        raise SystemExit("usage: trainer.py configs/weibo.yaml [key=value ...]")
    with Path(argv[0]).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for item in argv[1:]:
        key, value = item.split("=", 1)
        if value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
        else:
            try:
                config[key] = int(value)
            except ValueError:
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = value
    return config


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        world = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = world > 1
        if self.distributed:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.rank = dist.get_rank()
        else:
            self.local_rank = 0
            self.rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main = self.rank == 0
        set_seed(int(config.get("seed", 0)))
        self.num_loops = int(config.get("num_loops", 3))
        self.epochs = int(config.get("epochs", 10))
        self.patience = int(config.get("patience", 3))
        self.accum_steps = max(int(config.get("grad_accum_steps", 1)), 1)
        self.output_dir = Path(config.get("output_dir", "outputs/weibo"))
        if not self.output_dir.is_absolute():
            self.output_dir = ROOT / self.output_dir
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._build_model()
        self.train_loader, self.val_loader, self.test_loader, self.train_sampler = self._build_loaders()
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=float(config["lr"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=float(config.get("min_lr", float(config["lr"]) * 0.1)),
        )
        self.loss_fn = CrossEntropyLoss()

    def _build_model(self) -> torch.nn.Module:
        model = ThinkLoop(
            model_name=str(self.config["model_name"]),
            lora_r=int(self.config.get("lora_r", 16)),
            lora_alpha=int(self.config.get("lora_alpha", 32)),
            lora_dropout=float(self.config.get("lora_dropout", 0.05)),
            num_classes=int(self.config.get("num_classes", 2)),
            num_loops=self.num_loops,
        ).to(self.device)
        if self.is_main:
            total = sum(parameter.numel() for parameter in model.parameters())
            trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            print(f"ThinkLoop: {trainable}/{total} trainable, loops={self.num_loops}, device={self.device}")
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        return model

    def _loader(self, dataset: ThoughtDataset, shuffle: bool, sampler) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(int(self.config.get("seed", 0)))
        kwargs = {
            "batch_size": int(self.config["batch_size"]),
            "shuffle": shuffle and sampler is None,
            "sampler": sampler,
            "num_workers": int(self.config.get("num_workers", 0)),
            "collate_fn": thought_collator,
            "pin_memory": True,
            "generator": generator,
            "worker_init_fn": _seed_worker,
        }
        if "in_order" in inspect.signature(DataLoader).parameters:
            kwargs["in_order"] = True
        return DataLoader(dataset, **kwargs)

    def _thoughts(self, name: str) -> Path:
        directory = Path(self.config.get("thoughts_dir", self.output_dir / "thoughts"))
        if not directory.is_absolute():
            directory = ROOT / directory
        path = directory / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing stored thoughts at {path}. Run collect.py before training.")
        return path

    def _build_loaders(self):
        train = load_thoughts(self._thoughts("train"))
        val = load_thoughts(self._thoughts("val"))
        test = load_thoughts(self._thoughts("test"))
        if self.is_main:
            print(f"stored thoughts train {len(train)} val {len(val)} test {len(test)}")
        train_set = ThoughtDataset(train)
        sampler = (
            DistributedSampler(train_set, shuffle=True, seed=int(self.config.get("seed", 0)))
            if self.distributed
            else None
        )
        train_loader = self._loader(train_set, True, sampler)
        val_loader = self._loader(ThoughtDataset(val), False, None) if self.is_main else None
        test_loader = self._loader(ThoughtDataset(test), False, None) if self.is_main else None
        return train_loader, val_loader, test_loader, sampler

    def _unwrap(self) -> ThinkLoop:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _logits(self, batch: dict) -> torch.Tensor:
        return self.model(
            prefix_ids=batch["prefix_ids"],
            prefix_mask=batch["prefix_mask"],
            last_ids=batch["last_ids"],
            last_pos=batch["last_pos"],
            num_loops=self.num_loops,
        )

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_count = 0
        for step, batch in enumerate(tqdm(self.train_loader, disable=not self.is_main, leave=False)):
            batch = move_batch(batch, self.device)
            sync = (step + 1) % self.accum_steps == 0 or (step + 1) == len(self.train_loader)
            context = self.model.no_sync() if self.distributed and not sync else contextlib.nullcontext()
            with context:
                logits = self._logits(batch)
                loss = self.loss_fn(logits, batch["labels"])
                (loss / self.accum_steps).backward()
            total_loss += float(loss.detach().item()) * int(batch["labels"].size(0))
            total_count += int(batch["labels"].size(0))
            if not sync:
                continue
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("max_grad_norm", 1.0)))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        if self.distributed:
            packed = torch.tensor([total_loss, total_count], dtype=torch.float64, device=self.device)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            total_loss, total_count = float(packed[0]), int(packed[1])
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[dict[str, float], str]:
        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        for batch in loader:
            batch = move_batch(batch, self.device)
            logits = self._unwrap()(
                prefix_ids=batch["prefix_ids"],
                prefix_mask=batch["prefix_mask"],
                last_ids=batch["last_ids"],
                last_pos=batch["last_pos"],
                num_loops=self.num_loops,
            )
            y_true.extend(batch["labels"].detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        return score_predictions(y_true, y_pred), report_predictions(y_true, y_pred)

    def _save(self, name: str, epoch: int, metrics: dict | None) -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "model": self._unwrap().trainable_state(),
                "epoch": epoch,
                "metrics": metrics,
                "config": self.config,
            },
            self.output_dir / name,
        )

    def train(self) -> None:
        best_score = -1.0
        stale = 0
        best_epoch = None
        for epoch in range(1, self.epochs + 1):
            loss = self._train_one_epoch(epoch)
            self.scheduler.step()
            stop = False
            if self.is_main:
                message = f"epoch {epoch}/{self.epochs} loss {loss:.4f} lr {self.scheduler.get_last_lr()[0]:.2e}"
                metrics, report = self.evaluate(self.val_loader)
                message += f" acc {metrics['accuracy']:.4f} macro_f1 {metrics['macro_f1']:.4f}"
                if metrics["macro_f1"] > best_score:
                    best_score = metrics["macro_f1"]
                    stale = 0
                    best_epoch = epoch
                    self._save("best.pt", epoch, metrics)
                    print(message)
                    print(report)
                else:
                    stale += 1
                    if self.patience > 0:
                        message += f" stale {stale}/{self.patience}"
                    print(message)
                self._save("last.pt", epoch, metrics)
                stop = self.patience > 0 and stale >= self.patience
            if self.distributed:
                flag = torch.tensor([1 if stop else 0], device=self.device)
                dist.broadcast(flag, src=0)
                stop = bool(flag.item())
            if stop:
                if self.is_main:
                    print(f"early stop at epoch {epoch}, best val macro_f1 {best_score:.4f}")
                break
        if self.is_main:
            checkpoint = torch.load(self.output_dir / "best.pt", map_location=self.device, weights_only=False)
            self._unwrap().load_trainable(checkpoint["model"])
            print(f"test checkpoint epoch {checkpoint.get('epoch', best_epoch)}")
            metrics, report = self.evaluate(self.test_loader)
            print("test", " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
            print(report)
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    Trainer(load_config(sys.argv[1:])).train()
