"""ThinkLoop for text fake-news detection.

Thoughts are sampled once by a frozen model and stored as token ids. Training
only runs the loop: the first pass reads the last token's embedding, and every
later pass feeds the previous hidden state back at the same position. Those
passes attend to a frozen prefix cache. LoRA and the linear head are the only
trained parameters. The head reads the final hidden state.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM

from .cache import cache_from_snapshot, snapshot_cache
from .sequences import position_ids


class ThinkLoop(nn.Module):
    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_classes: int = 2,
        num_loops: int = 3,
    ):
        super().__init__()
        if num_loops < 1:
            raise ValueError("num_loops must be at least 1 so the LoRA loop runs.")
        self.num_loops = num_loops
        policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        policy.requires_grad_(False)
        self.llm = get_peft_model(
            policy,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        hidden = int(policy.config.hidden_size)
        self.head = nn.Linear(hidden, num_classes)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {key: value.detach().cpu() for key, value in self.state_dict().items() if key in names}

    def load_trainable(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad = [key for key in missing if "lora_" in key or key.startswith("head.")]
        if bad or unexpected:
            raise RuntimeError(f"Checkpoint did not match the loop head. missing={bad} unexpected={unexpected}")

    def forward(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        last_ids: torch.Tensor,
        last_pos: torch.Tensor,
        num_loops: int | None = None,
    ) -> torch.Tensor:
        loops = self.num_loops if num_loops is None else num_loops
        hidden = self._loop(prefix_ids, prefix_mask, last_ids, last_pos, loops)
        return self.head(hidden.float())

    def _loop(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        last_ids: torch.Tensor,
        last_pos: torch.Tensor,
        num_loops: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            with self.llm.disable_adapter():
                prefix = self.llm(
                    input_ids=prefix_ids,
                    attention_mask=prefix_mask,
                    position_ids=position_ids(prefix_mask),
                    use_cache=True,
                )
            frozen = snapshot_cache(prefix.past_key_values)
        query_mask = torch.cat([prefix_mask, torch.ones_like(prefix_mask[:, :1])], dim=1)
        query_pos = last_pos.unsqueeze(1)
        hidden = None
        for step in range(num_loops):
            past = cache_from_snapshot(frozen)
            if step == 0:
                output = self.llm(
                    input_ids=last_ids,
                    attention_mask=query_mask,
                    position_ids=query_pos,
                    past_key_values=past,
                    use_cache=True,
                )
            else:
                output = self.llm(
                    inputs_embeds=hidden.unsqueeze(1),
                    attention_mask=query_mask,
                    position_ids=query_pos,
                    past_key_values=past,
                    use_cache=True,
                )
            hidden = output.last_hidden_state[:, 0, :]
        return hidden
