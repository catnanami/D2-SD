"""Dual-draft (DFlash + DTA) speculative generator.

Phase 1: single-sequence port of the dflash_generate loop in benchmark_d3.py.
The cascade verification call goes through generation.verification so future
batched paths can share it. Control flow (branch selection, second-draft prefill
bookkeeping, cascade vs plain verify split, KV merge after cascade) is kept
byte-for-byte equivalent to the original.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import CascadeGraphRunner, DFlashDraftModel, sample, extract_context_feature

from .state import GenerationResult, GenerationState
from .verification import cascade_verify_forward


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


class D3Generator:
    def __init__(
        self,
        target: AutoModelForCausalLM,
        draft: DFlashDraftModel,
        dta: DFlashDraftModel,
        block_size: int,
        block_size_2: Optional[int] = None,
        dta_cascade_runner: Optional[CascadeGraphRunner] = None,
        verify_cascade_runner: Optional[CascadeGraphRunner] = None,
    ) -> None:
        if block_size_2 is None:
            block_size_2 = block_size
        if block_size_2 < block_size:
            raise ValueError(
                f"block_size_2 ({block_size_2}) must be >= block_size ({block_size}); "
                "the second draft can only extend the first draft, not shrink it."
            )

        self.target = target
        self.draft = draft
        self.dta = dta
        self.block_size = block_size
        self.block_size_2 = block_size_2
        self.dta_cascade_runner = dta_cascade_runner
        self.verify_cascade_runner = verify_cascade_runner

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        stop_token_ids: Optional[List[int]],
        temperature: float = 0.0,
    ) -> GenerationResult:
        assert input_ids.shape[0] == 1, "Phase 1 D3Generator is S=1 only"
        target = self.target
        draft = self.draft
        dta = self.dta
        block_size = self.block_size
        block_size_2 = self.block_size_2
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
        state.target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

        time_to_first_token = _cuda_time() - prefill_start

        # Decode timing: reset the clock after the first iter so the DTA
        # prefill (which runs once) doesn't skew steady-state numbers.
        decode_start = _cuda_time()
        start = num_input_tokens
        acceptance_lengths: List[int] = []
        draft_prefill = True

        while start < max_length_int:
            block_output_ids = state.output_ids[:, start:start + block_size].clone()
            block_position_ids = state.position_ids[:, start:start + block_size_2]
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
            block_output_ids[:, 1:], probs = sample(draft_logits)

            token_ids = block_output_ids[0, 1:]
            conf = probs[torch.arange(block_size - 1, device=probs.device), token_ids]
            cum_prod = torch.cumprod(conf, dim=0)
            rates = torch.empty(block_size - 1, device=conf.device)
            rates[0] = 1.0 - conf[0]
            rates[1:] = cum_prod[:-1] * (1.0 - conf[1:])

            topk = min(4, block_size - 1)
            _, topk_idx = torch.topk(rates, topk)
            selected_pls = sorted(topk_idx.tolist())

            block_ids_list: List[torch.Tensor] = []
            valid_pls: List[int] = []

            if block_size_2 > block_size:
                block_ids_list.append(torch.cat(
                    [block_output_ids,
                     state.output_ids[:, start + block_size: start + block_size_2]],
                    dim=1,
                ))
                valid_pls.append(block_size - 1)

            for pl in selected_pls:
                anchor_pos = pl + 1
                if anchor_pos >= block_size_2:
                    continue
                block_ids_list.append(torch.cat(
                    [block_output_ids[:, :anchor_pos],
                     state.output_ids[:, start + anchor_pos: start + block_size_2]],
                    dim=1,
                ))
                valid_pls.append(pl)

            if len(block_ids_list) == 0:
                state.past_key_values_dta.crop(start)
            else:
                block_ids = torch.cat(block_ids_list, dim=0)
                dta_noise_embedding = target.model.embed_tokens(block_ids)
                min_pl = min(valid_pls)

                second_draft_logits = target.lm_head(dta(
                    target_hidden=state.target_hidden,
                    noise_embedding=dta_noise_embedding,
                    position_ids=state.position_ids[:, state.past_key_values_dta.get_seq_length(): start + block_size_2],
                    past_key_values=state.past_key_values_dta,
                    use_cache=True,
                    is_causal=False,
                    second_draft=True,
                    prefill=draft_prefill,
                    cascade_graph_runner=self.dta_cascade_runner,
                )[:, -block_size_2 + 1 + min_pl:, :])
                state.past_key_values_dta.crop(start)

                for j, pl in enumerate(valid_pls):
                    offset_j = pl - min_pl
                    second_block_ids, _ = sample(
                        second_draft_logits[j, offset_j:, :].unsqueeze(0)
                    )
                    block_ids[j, 1 + pl:] = second_block_ids

                if block_size_2 == block_size:
                    block_output_ids = torch.cat([block_ids, block_output_ids], dim=0)
                else:
                    block_output_ids = block_ids

            if draft_prefill:
                draft_prefill = False
                decode_start = _cuda_time()

            bsz = block_output_ids.shape[0]
            use_cascade = bsz > 1
            if use_cascade:
                output_logits, output_hidden_states, new_kv_list = cascade_verify_forward(
                    target, block_output_ids, block_position_ids,
                    state.past_key_values_target,
                    output_hidden_states=True,
                    cascade_graph_runner=self.verify_cascade_runner,
                )
                output = SimpleNamespace(logits=output_logits, hidden_states=output_hidden_states)
            else:
                output = target(
                    block_output_ids,
                    position_ids=block_position_ids,
                    past_key_values=state.past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True if block_size_2 > 1 else False,
                )

            output_logits = output.logits
            valid_blocks = torch.ones(output.logits.shape[0], device=device)
            acceptance_length = 0
            accept_ids: List[torch.Tensor] = []
            while True:
                valid_block = (valid_blocks > 0).nonzero(as_tuple=True)[0][0]
                logit = output_logits[valid_block, acceptance_length, :]
                sampled_token, _ = sample(logit.unsqueeze(0).unsqueeze(0), temperature)
                acceptance_length += 1
                accept_ids.append(sampled_token)
                if acceptance_length == block_size_2:
                    break
                next_tokens = block_output_ids[:, acceptance_length]
                valid_mask = (next_tokens == sampled_token)
                valid_blocks = torch.where(valid_mask, valid_blocks, 0)
                if valid_blocks.sum() == 0.:
                    break

            state.output_ids[:, start + 1:start + acceptance_length + 1] = torch.cat(accept_ids, dim=0)

            acceptance_lengths.append(acceptance_length)
            start += acceptance_length

            if use_cascade:
                for layer_idx, layer in enumerate(state.past_key_values_target.layers):
                    k_win = new_kv_list[layer_idx][0][valid_block:valid_block + 1, :, :acceptance_length, :]
                    v_win = new_kv_list[layer_idx][1][valid_block:valid_block + 1, :, :acceptance_length, :]
                    layer.keys = torch.cat([layer.keys, k_win], dim=2)
                    layer.values = torch.cat([layer.values, v_win], dim=2)
            else:
                state.past_key_values_target.crop(start)
                for layer in state.past_key_values_target.layers:
                    layer.keys = layer.keys[valid_block:valid_block + 1, ...]
                    layer.values = layer.values[valid_block:valid_block + 1, ...]

            state.target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)[valid_block, :acceptance_length, :].unsqueeze(0)

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
            output_ids=input_ids,
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
        block_size_2 = self.block_size_2
        num_input_tokens = input_ids.shape[1]

        output_ids = torch.full(
            (1, max_length + block_size_2),
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
            past_key_values_dta=DynamicCache(),
            acceptance_lengths=[[]],
        )
