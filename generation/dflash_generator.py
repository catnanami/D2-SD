"""Single-draft (DFlash-only) speculative generator.

Phase 1: logic is lifted verbatim from benchmark_dflash.py:dflash_generate with
the loop state routed through GenerationState. Batching across sequences (S>1)
will replace the scalar `start` and `.item()` sites without changing the
surrounding control flow.
"""
from __future__ import annotations

import time
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature

from .state import GenerationResult, GenerationState


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


class DFlashGenerator:
    """Orchestrates DFlash draft + target verification for one or more input ids tensors."""

    def __init__(
        self,
        target: AutoModelForCausalLM,
        draft: DFlashDraftModel,
        block_size: int,
    ) -> None:
        self.target = target
        self.draft = draft
        self.block_size = block_size

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,             # [1, prompt_len] -- Phase 1 accepts S=1 only
        max_new_tokens: int,
        stop_token_ids: Optional[List[int]],
        temperature: float = 0.0,
    ) -> GenerationResult:
        assert input_ids.shape[0] == 1, "Phase 1 DFlashGenerator is S=1 only"
        target = self.target
        draft = self.draft
        block_size = self.block_size
        device = target.device

        num_input_tokens = input_ids.shape[1]
        max_length_int = num_input_tokens + max_new_tokens

        state = self._init_state(input_ids, max_length_int, device)
        prefill_start = _cuda_time()

        output = target(
            input_ids,
            position_ids=state.position_ids[:, :num_input_tokens],
            past_key_values=state.past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        state.output_ids[:, :num_input_tokens] = input_ids
        state.output_ids[:, num_input_tokens:num_input_tokens + 1], _ = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)
        state.target_hidden = target_hidden

        time_to_first_token = _cuda_time() - prefill_start

        # Decode timing: reset the clock after the first iter so the draft
        # prefill (which runs once) doesn't skew steady-state numbers.
        decode_start = _cuda_time()
        start = num_input_tokens
        acceptance_lengths: List[int] = []
        draft_prefill = True

        while start < max_length_int:
            block_output_ids = state.output_ids[:, start:start + block_size].clone()
            block_position_ids = state.position_ids[:, start:start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)

            draft_logits = target.lm_head(draft(
                target_hidden=state.target_hidden,
                noise_embedding=noise_embedding,
                position_ids=state.position_ids[:, state.past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=state.past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1:, :])
            state.past_key_values_draft.crop(start)
            block_output_ids[:, 1:], _ = sample(draft_logits)

            if draft_prefill:
                draft_prefill = False
                decode_start = _cuda_time()

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=state.past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior, _ = sample(output.logits, temperature)
            posterior = posterior.unsqueeze(0)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )

            state.output_ids[:, start:start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            state.output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

            acceptance_lengths.append(acceptance_length + 1)
            start += acceptance_length + 1
            state.past_key_values_target.crop(start)
            state.target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)[:, :acceptance_length + 1, :]

            if stop_token_ids is not None and any(
                stop_token_id in state.output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
            ):
                break

        output_ids = state.output_ids[:, :max_length_int]
        output_ids = output_ids[:, output_ids[0] != draft.mask_token_id]
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_idx = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(as_tuple=True)[0]
            if stop_idx.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_idx[0] + 1]

        num_output_tokens = output_ids.shape[1] - num_input_tokens
        total_decode_time = _cuda_time() - decode_start
        time_per_output_token = total_decode_time / max(num_output_tokens, 1)

        return GenerationResult(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            time_to_first_token=time_to_first_token,
            time_per_output_token=time_per_output_token,
            acceptance_lengths=acceptance_lengths,
        )

    @torch.inference_mode()
    def baseline_generate(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Plain autoregressive generation with the target model for timing comparison."""
        target = self.target
        num_input_tokens = input_ids.shape[1]
        past_key_values = DynamicCache()

        output = target(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token, _ = sample(output.logits, temperature)

        decode_start = _cuda_time()
        for _ in range(num_tokens - 1):
            output = target(
                next_token.unsqueeze(0) if next_token.dim() == 1 else next_token,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            next_token, _ = sample(output.logits, temperature)

        total_decode_time = _cuda_time() - decode_start
        time_per_output_token = total_decode_time / max(num_tokens - 1, 1)

        return GenerationResult(
            output_ids=input_ids,  # baseline path does not reconstruct the full sequence; only timing is used
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_tokens,
            time_to_first_token=0.0,
            time_per_output_token=time_per_output_token,
            acceptance_lengths=[],
        )

    def _init_state(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        device: torch.device,
    ) -> GenerationState:
        block_size = self.block_size
        num_input_tokens = input_ids.shape[1]

        output_ids = torch.full(
            (1, max_length + block_size),
            self.draft.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

        return GenerationState(
            output_ids=output_ids,
            position_ids=position_ids,
            start=torch.tensor([num_input_tokens], device=device, dtype=torch.long),
            prompt_len=torch.tensor([num_input_tokens], device=device, dtype=torch.long),
            max_length=torch.tensor([max_length], device=device, dtype=torch.long),
            active_mask=torch.ones(1, dtype=torch.bool, device=device),
            target_hidden=torch.empty(0, device=device),
            past_key_values_target=DynamicCache(),
            past_key_values_draft=DynamicCache(),
            past_key_values_dta=None,
            acceptance_lengths=[[]],
        )
