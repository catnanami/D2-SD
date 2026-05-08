"""Target-model verification forward passes.

The cascade variant uses flashinfer to attend over a shared KV prefix plus
per-branch local KV, avoiding cache expansion when verifying multiple draft
branches for a single sequence. The semantics here match the original
benchmark_d3.py implementation verbatim; batching across sequences (S > 1)
will come from callers flattening [S, B] into the bsz dimension while keeping
the shared-prefix assumption valid per sequence.
"""
from types import SimpleNamespace
from typing import Optional

import torch
import flashinfer
from transformers import AutoModelForCausalLM, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from model import CascadeGraphRunner


def _apply_rotary_pos_emb_qwen3(q, k, cos, sin):
    """Qwen3 RoPE: cos/sin are [bsz, seq_len, head_dim] with the second half a
    copy of the first; combined via the rotate_half trick."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _apply_rotary_pos_emb_gpt_oss(q, k, cos, sin):
    """GPT-OSS RoPE: cos/sin are [bsz, seq_len, head_dim/2] (HALF the head dim);
    applied by splitting q/k into first/second halves and recombining. Shape
    differs from Qwen3 so we cannot share a single helper."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def _apply(x):
        first, second = torch.chunk(x, 2, dim=-1)
        return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)

    return _apply(q), _apply(k)


# Back-compat alias.
_apply_rotary_pos_emb = _apply_rotary_pos_emb_qwen3


def cascade_verify_forward(
    target_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: DynamicCache,
    output_hidden_states: bool = True,
    cascade_graph_runner: Optional[CascadeGraphRunner] = None,
):
    """
    Manual layer-by-layer forward through the target model using flashinfer
    cascade attention. Avoids expanding the shared prefix KV cache to batch size.
    Supports both Qwen3 and GPT-OSS targets (dispatches on the first layer's
    attention class name).

    Args:
        target_model: HuggingFace CausalLM (Qwen3 or GPT-OSS)
        input_ids: [bsz, block_size] candidate token ids
        position_ids: [1, block_size] positions (same for all candidates)
        past_key_values: DynamicCache with [1, H_kv, past_len, D] per layer (untouched)
        output_hidden_states: whether to collect per-layer hidden states

    Returns:
        (logits, hidden_states_list, new_kv_list)
        - logits: [bsz, block_size, vocab_size]
        - hidden_states_list: list of [bsz, block_size, hidden_size] (if requested)
        - new_kv_list: list of (k_new, v_new) per layer, each [bsz, H_kv, block_size, D]
    """
    model = target_model.model
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    # Architecture dispatch by attention class name; lets us avoid importing the
    # GPT-OSS module at file load (it is only present on recent transformers).
    attn_cls_name = type(model.layers[0].self_attn).__name__
    is_gpt_oss = attn_cls_name == "GptOssAttention"
    rope_apply = _apply_rotary_pos_emb_gpt_oss if is_gpt_oss else _apply_rotary_pos_emb_qwen3

    hidden_states = model.embed_tokens(input_ids)

    # cos/sin shape differs between families: Qwen3 returns [bsz, seq_len, head_dim]
    # (duplicated halves), GPT-OSS returns [bsz, seq_len, head_dim/2].
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    cos, sin = position_embeddings

    all_hidden_states = [hidden_states] if output_hidden_states else None
    new_kv_list = []

    for layer_idx, decoder_layer in enumerate(model.layers):
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        attn = decoder_layer.self_attn

        q = attn.q_proj(hidden_states)
        k_new = attn.k_proj(hidden_states)
        v_new = attn.v_proj(hidden_states)

        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim

        q = q.view(bsz, seq_len, num_heads, head_dim)
        k_new = k_new.view(bsz, seq_len, num_kv_heads, head_dim)
        v_new = v_new.view(bsz, seq_len, num_kv_heads, head_dim)

        # Qwen3 has QK-RMSNorm before RoPE; GPT-OSS doesn't.
        if not is_gpt_oss:
            q = attn.q_norm(q)
            k_new = attn.k_norm(k_new)

        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)

        q, k_new = rope_apply(q, k_new, cos, sin)

        new_kv_list.append((k_new, v_new))

        k_shared = past_key_values[layer_idx][0][0]
        v_shared = past_key_values[layer_idx][1][0]

        # Sliding window: GPT-OSS sets attn.sliding_window only on layers with
        # config.layer_types[idx] == "sliding_attention"; Qwen3 leaves it None.
        sliding_window = getattr(attn, 'sliding_window', None)
        if sliding_window is not None and k_shared.shape[1] > sliding_window:
            k_shared = k_shared[:, -sliding_window:, :]
            v_shared = v_shared[:, -sliding_window:, :]

        scaling = head_dim ** -0.5

        k_shared_fi = k_shared.transpose(0, 1).contiguous()
        v_shared_fi = v_shared.transpose(0, 1).contiguous()

        q_fi = q.transpose(1, 2)

        q_all = q_fi.reshape(bsz * seq_len, num_heads, head_dim)
        out_shared, lse_shared = flashinfer.single_prefill_with_kv_cache(
            q_all, k_shared_fi, v_shared_fi,
            causal=False, return_lse=True, sm_scale=scaling,
        )
        out_shared = out_shared.view(bsz, seq_len, num_heads, head_dim)
        lse_shared = lse_shared.view(bsz, seq_len, num_heads)

        k_new_fi = k_new.transpose(1, 2).contiguous()
        v_new_fi = v_new.transpose(1, 2).contiguous()

        if cascade_graph_runner is not None:
            cascade_graph_runner.run(q_fi.contiguous(), k_new_fi, v_new_fi, out_shared, lse_shared)
        else:
            for i in range(bsz):
                out_local, lse_local = flashinfer.single_prefill_with_kv_cache(
                    q_fi[i].contiguous(), k_new_fi[i].contiguous(), v_new_fi[i].contiguous(),
                    causal=True, return_lse=True, sm_scale=scaling,
                )
                flashinfer.merge_state_in_place(
                    out_shared[i], lse_shared[i], out_local, lse_local,
                )

        # GPT-OSS attention sinks: HF concatenates a learnable per-head logit
        # `attn.sinks` to the attention logits before softmax and discards it
        # afterwards, putting an extra `exp(sink_h)` term in the softmax
        # denominator with no value contribution. Equivalently the final output
        # is `O_merged * sigmoid(lse_merged - sink_h)`. After the cascade merge
        # above, `lse_shared` holds the merged log-sum-exp.
        if is_gpt_oss:
            sinks = attn.sinks.to(lse_shared.dtype).view(1, 1, -1)  # [1, 1, num_heads]
            rescale = torch.sigmoid(lse_shared - sinks).to(out_shared.dtype)
            out_shared = out_shared * rescale.unsqueeze(-1)

        attn_output = out_shared.reshape(bsz, seq_len, -1)
        attn_output = attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        # GPT-OSS's MoE MLP returns (hidden_states, router_scores); Qwen3's
        # dense MLP returns a tensor.
        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_out = decoder_layer.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        hidden_states = residual + mlp_out

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

    hidden_states = model.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states.append(hidden_states)

    logits = target_model.lm_head(hidden_states)

    return logits, all_hidden_states, new_kv_list


def plain_verify_forward(
    target_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: DynamicCache,
    output_hidden_states: bool = True,
):
    """Single-branch verification via the standard HF forward. Kept here so the
    generator can call one function regardless of branch count."""
    output = target_model(
        input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=output_hidden_states,
    )
    return SimpleNamespace(logits=output.logits, hidden_states=output.hidden_states)
