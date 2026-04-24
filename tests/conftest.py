from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size

    def _encode_one(self, text: str, max_length: int | None = None) -> list[int]:
        ids = [2 + (ord(ch) % max(2, self.vocab_size - 2)) for ch in text]
        if not ids:
            ids = [self.eos_token_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def __call__(self, text, return_tensors="pt", truncation=True, max_length: int | None = None, padding=False):
        if isinstance(text, str):
            rows = [self._encode_one(text, max_length=max_length)]
        else:
            rows = [self._encode_one(t, max_length=max_length) for t in text]

        max_len = max(len(r) for r in rows)
        if padding:
            target = max_len
        else:
            target = max_len

        input_ids = []
        attention_mask = []
        for r in rows:
            pad_n = target - len(r)
            input_ids.append(r + [self.pad_token_id] * pad_n)
            attention_mask.append([1] * len(r) + [0] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ToySelfAttn(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        return self.o_proj(torch.tanh(mixed))


class ToyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.sigmoid(self.gate_proj(x)) * torch.relu(self.up_proj(x))
        return self.down_proj(gated)


class ToyLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = ToySelfAttn(hidden_size)
        self.mlp = ToyMLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.self_attn(x) + 0.1 * self.mlp(x)


class ToyModel(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([ToyLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            _name_or_path="toy/model",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
        output_attentions: bool = False,
    ):
        x = self.embed(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # causal next-token loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return SimpleNamespace(
            logits=logits,
            loss=loss,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=None,
        )


@pytest.fixture()
def toy_tokenizer() -> ToyTokenizer:
    return ToyTokenizer(vocab_size=128)


@pytest.fixture()
def toy_model() -> ToyModel:
    torch.manual_seed(0)
    return ToyModel(vocab_size=128, hidden_size=32, num_layers=4)


def build_task_batches(tokenizer: ToyTokenizer, prompts: Sequence[str]) -> list[dict[str, torch.Tensor]]:
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=32)
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return [{"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}]
