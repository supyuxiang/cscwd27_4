"""Turn a frozen generation into the prefix cache inputs and the token that will be looped."""

from __future__ import annotations

import torch


def pack_thoughts(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    sequences: torch.Tensor,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split each row into a left-padded prefix and the last generated token.

    ``sequences`` is the generate() result: the left-padded prompt followed by
    the completion. The completion ends at the first EOS, and that EOS is the
    last token. Prompts are recovered from ``prompt_mask``, so left padding is
    not treated as text.
    """
    if sequences.size(1) < prompt_ids.size(1):
        raise RuntimeError("Generated sequences are shorter than the prompt.")
    completion = sequences[:, prompt_ids.size(1) :]
    prefixes: list[list[int]] = []
    lasts: list[int] = []
    for index in range(prompt_ids.size(0)):
        prompt = prompt_ids[index, prompt_mask[index].bool()].tolist()
        extra = completion[index]
        hits = (extra == eos_id).nonzero(as_tuple=False)
        take = int(hits[0]) + 1 if len(hits) else int(extra.numel())
        full = prompt + extra[:take].tolist()
        if len(full) < 2:
            raise RuntimeError("ThinkLoop needs at least one prefix token and one last token.")
        prefixes.append(full[:-1])
        lasts.append(full[-1])
    width = max(len(row) for row in prefixes)
    prefix_ids = prompt_ids.new_zeros(len(prefixes), width)
    prefix_mask = prompt_mask.new_zeros(len(prefixes), width)
    for index, row in enumerate(prefixes):
        prefix_ids[index, width - len(row) :] = torch.tensor(row, dtype=prompt_ids.dtype, device=prompt_ids.device)
        prefix_mask[index, width - len(row) :] = 1
    last_ids = torch.tensor(lasts, dtype=prompt_ids.dtype, device=prompt_ids.device).unsqueeze(1)
    last_pos = prefix_mask.sum(dim=1).to(dtype=torch.long)
    return prefix_ids, prefix_mask, last_ids, last_pos


def pack_token_rows(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad stored thoughts. The final id of each row is the loop token."""
    if any(len(row) < 2 for row in rows):
        raise RuntimeError("Each stored thought needs a prefix and a last token.")
    prefixes = [row[:-1] for row in rows]
    width = max(len(row) for row in prefixes)
    prefix_ids = torch.zeros(len(rows), width, dtype=torch.long)
    prefix_mask = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(prefixes):
        prefix_ids[index, width - len(row) :] = torch.tensor(row, dtype=torch.long)
        prefix_mask[index, width - len(row) :] = 1
    last_ids = torch.tensor([row[-1] for row in rows], dtype=torch.long).unsqueeze(1)
    last_pos = prefix_mask.sum(dim=1)
    return prefix_ids, prefix_mask, last_ids, last_pos


def position_ids(mask: torch.Tensor) -> torch.Tensor:
    """Positions of real tokens in a left-padded mask. Pads stay at position 0."""
    return (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
