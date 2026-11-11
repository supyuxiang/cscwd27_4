"""Build a fresh cache from a frozen prefix so each loop starts at the same token."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


def snapshot_cache(past) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs = []
    for layer in past.layers:
        pairs.append((layer.keys.detach(), layer.values.detach()))
    return pairs


def cache_from_snapshot(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    layers = []
    for keys, values in pairs:
        layer = DynamicLayer()
        layer.keys = keys.clone()
        layer.values = values.clone()
        layer.dtype = keys.dtype
        layer.device = keys.device
        layer.is_initialized = True
        layers.append(layer)
    cache = DynamicCache()
    cache.layers = layers
    return cache
