"""Per-sequence generation state container.

Phase 1 keeps S=1 semantics (every field has a batch-leading dimension of size 1),
but every slot is shaped so the batched path can reuse it without structural
changes. All tensors live on the target model's device; acceptance_lengths is a
Python list per sequence for post-hoc statistics.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import DynamicCache


@dataclass
class GenerationState:
    # [S, L] where L = max_length + block_size_2
    output_ids: torch.Tensor
    # [S, L] absolute position indices (same row for every sequence when prompts share length; per-row otherwise)
    position_ids: torch.Tensor

    # [S] current write cursor (= prompt_len initially, advanced by acceptance_length)
    start: torch.Tensor
    # [S] prompt length (== num_input_tokens originally)
    prompt_len: torch.Tensor
    # [S] per-sequence hard cap (prompt_len + max_new_tokens)
    max_length: torch.Tensor
    # [S] True while the sequence has not hit max_length or stop tokens
    active_mask: torch.Tensor

    # target hidden features from the most recently accepted tokens. Shape
    # [S, K, H*num_target_layers] where K is padded to the batch-max of the
    # previous acceptance_length; Phase 1 keeps S=1 so K is just that one
    # sequence's length.
    target_hidden: torch.Tensor

    past_key_values_target: DynamicCache
    past_key_values_draft: DynamicCache
    # DTA cache is only populated by D3Generator; DFlashGenerator leaves this None.
    past_key_values_dta: Optional[DynamicCache] = None

    # per-sequence list of accepted block lengths (Python list of ints per seq)
    acceptance_lengths: List[List[int]] = field(default_factory=list)

    # timing — [S] floats; scalar values are broadcast when needed
    time_to_first_token: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return self.output_ids.shape[0]

    def any_active(self) -> bool:
        return bool(self.active_mask.any().item())


@dataclass
class GenerationResult:
    """What a generator returns for a single input."""
    output_ids: torch.Tensor            # [1, generated_len] (trimmed, stop-aware)
    num_input_tokens: int
    num_output_tokens: int
    time_to_first_token: float
    time_per_output_token: float
    acceptance_lengths: List[int]

    # optional; filled in by benchmark driver after it runs baseline
    baseline_time_per_token: Optional[float] = None
