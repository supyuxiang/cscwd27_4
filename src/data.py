"""Text-only Weibo loader. Images are ignored; label 1 is fake."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .sequences import pack_token_rows


SYSTEM = "你是新闻核查员。阅读新闻后写出核查推理，只写推理过程，不要给出真或假的结论。"


def load_jsonl_samples(path: str | Path) -> list[dict]:
    samples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                {
                    "id": str(row.get("id", len(samples))),
                    "text": str(row.get("text") or ""),
                    "label": int(row["label"]),
                }
            )
    if not samples:
        raise RuntimeError(f"No samples in {path}")
    return samples


def stratified_split(samples: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if val_ratio <= 0:
        return list(samples), []
    groups: dict[int, list[dict]] = {}
    for sample in samples:
        groups.setdefault(int(sample["label"]), []).append(sample)
    rng = random.Random(seed)
    train, val = [], []
    for group in groups.values():
        rng.shuffle(group)
        n_val = min(int(round(len(group) * val_ratio)), len(group) - 1)
        n_val = max(n_val, 0)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_thoughts(path: str | Path) -> list[dict]:
    samples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            token_ids = [int(token) for token in row["token_ids"]]
            if len(token_ids) < 2:
                continue
            samples.append({"id": str(row.get("id", len(samples))), "label": int(row["label"]), "token_ids": token_ids})
    if not samples:
        raise RuntimeError(f"No stored thoughts in {path}")
    return samples


class ThoughtDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {"token_ids": sample["token_ids"], "label": int(sample["label"])}


def build_prompt_ids(tokenizer, text: str, max_text_tokens: int) -> list[int]:
    text = text.strip() or "。"
    token_ids = tokenizer(text, add_special_tokens=False).input_ids[:max_text_tokens]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]
    return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def thought_collator(batch: list[dict]) -> dict:
    prefix_ids, prefix_mask, last_ids, last_pos = pack_token_rows([sample["token_ids"] for sample in batch])
    return {
        "prefix_ids": prefix_ids,
        "prefix_mask": prefix_mask,
        "last_ids": last_ids,
        "last_pos": last_pos,
        "labels": torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.long),
    }
