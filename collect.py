"""Sample frozen thoughts with vLLM and store the token ids for loop training."""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from trainer import load_config
from src.data import build_prompt_ids, load_jsonl_samples, stratified_split


def thoughts_dir(config: dict) -> Path:
    directory = Path(config.get("thoughts_dir", Path(config.get("output_dir", "outputs/weibo")) / "thoughts"))
    if not directory.is_absolute():
        directory = ROOT / directory
    return directory


def _splits(config: dict) -> dict[str, list[dict]]:
    train = load_jsonl_samples(config["train_jsonl"])
    val_path = str(config.get("val_jsonl") or "").strip()
    if val_path:
        val = load_jsonl_samples(val_path)
    else:
        train, val = stratified_split(train, float(config.get("val_ratio", 0.1)), int(config.get("seed", 0)))
    test = load_jsonl_samples(config["test_jsonl"])
    return {"train": train, "val": val, "test": test}


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect(config: dict) -> None:
    directory = thoughts_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "done.json"
    if marker.is_file() and not bool(config.get("recollect", False)):
        print(f"thoughts already stored in {directory}")
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=False)
    splits = _splits(config)
    prompts = []
    owners = []
    max_text = int(config.get("max_text_tokens", 256))
    for name, samples in splits.items():
        for sample in samples:
            prompts.append(TokensPrompt(prompt_token_ids=build_prompt_ids(tokenizer, sample["text"], max_text)))
            owners.append((name, sample))
    engine = LLM(
        model=str(config["model_name"]),
        tensor_parallel_size=int(config.get("collect_tp", 4)),
        dtype="bfloat16",
        seed=int(config.get("seed", 0)),
        gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.85)),
        disable_custom_all_reduce=True,
        trust_remote_code=False,
    )
    params = SamplingParams(
        temperature=float(config.get("temperature", 0.7)),
        top_p=float(config.get("top_p", 0.95)),
        max_tokens=int(config.get("max_new_tokens", 64)),
        seed=int(config.get("seed", 0)),
    )
    outputs = engine.generate(prompts, params, use_tqdm=True)
    stored = {name: [] for name in splits}
    for (name, sample), output in zip(owners, outputs):
        completion = [int(token) for token in output.outputs[0].token_ids]
        token_ids = list(output.prompt_token_ids) + completion
        if len(token_ids) < 2:
            continue
        stored[name].append(
            {
                "id": sample["id"],
                "label": int(sample["label"]),
                "token_ids": token_ids,
                "thought": tokenizer.decode(completion, skip_special_tokens=True),
            }
        )
    for name, rows in stored.items():
        _write(directory / f"{name}.jsonl", rows)
        print(f"{name} {len(rows)}")
    marker.write_text(
        json.dumps(
            {
                "counts": {name: len(rows) for name, rows in stored.items()},
                "temperature": float(config.get("temperature", 0.7)),
                "top_p": float(config.get("top_p", 0.95)),
                "max_new_tokens": int(config.get("max_new_tokens", 64)),
                "seed": int(config.get("seed", 0)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del outputs
    del engine
    from vllm.distributed.parallel_state import destroy_model_parallel

    destroy_model_parallel()
    gc.collect()


if __name__ == "__main__":
    collect(load_config(sys.argv[1:]))
