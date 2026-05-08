"""Unified benchmark driver for DFlash and D3 (DFlash + DTA) speculative decoding.

Usage:
    torchrun --nproc_per_node=N --master_port=29600 benchmark.py \
        --mode {dflash,d3} \
        --model-name-or-path <target> \
        --draft-name-or-path <dflash> \
        [--dta-name-or-path <dta>]  # required when --mode d3
        --dataset gsm8k --max-samples 16 --max-new-tokens 256

Phase 1: --batch-size is accepted but only S=1 is implemented; values > 1 will
fall back to sequential per-sample execution for now and will be batched
properly in the next refactor phase.
"""
import argparse
import random
from itertools import chain

from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import CascadeGraphRunner, DFlashDraftModel, load_and_process_dataset
from generation import D3Generator, DFlashGenerator
import distributed as dist


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
        return False


def _build_cascade_runners(mode, block_size_2, dta_model, target):
    if mode != "d3":
        return None, None

    dta_config = dta_model.config
    dta_head_dim = getattr(dta_config, "head_dim", dta_config.hidden_size // dta_config.num_attention_heads)
    dta_runner = CascadeGraphRunner(
        block_size=block_size_2,
        num_heads=dta_config.num_attention_heads,
        num_kv_heads=dta_config.num_key_value_heads,
        head_dim=dta_head_dim,
        dtype=next(dta_model.parameters()).dtype,
        device=dta_model.device,
        sm_scale=dta_head_dim ** -0.5,
        causal=False,
    )
    target_config = target.config
    target_head_dim = getattr(target_config, "head_dim", target_config.hidden_size // target_config.num_attention_heads)
    verify_runner = CascadeGraphRunner(
        block_size=block_size_2,
        num_heads=target_config.num_attention_heads,
        num_kv_heads=target_config.num_key_value_heads,
        head_dim=target_head_dim,
        dtype=next(target.parameters()).dtype,
        device=target.device,
        sm_scale=target_head_dim ** -0.5,
        causal=True,
    )
    for warmup_bsz in range(2, 6):
        dta_runner._capture(warmup_bsz)
        verify_runner._capture(warmup_bsz)
    return dta_runner, verify_runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dflash", "d3"], required=True,
                        help="dflash = DFlash-only (single draft); d3 = DFlash + DTA (dual draft)")
    parser.add_argument("--model-name-or-path", type=str, default="../qwen3-8b")
    parser.add_argument("--draft-name-or-path", type=str, default="../qwen3-8b-dflash")
    parser.add_argument("--dta-name-or-path", type=str, default=None,
                        help="Required when --mode d3")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Block size for the first (DFlash) draft.")
    parser.add_argument("--block-size-2", type=int, default=None,
                        help="Block size for the second (DTA) draft and verify step (d3 mode only). "
                             "Must be >= --block-size. Defaults to --block-size.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Phase 1 only supports 1; values >1 run sequentially for now.")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.mode == "d3" and args.dta_name_or_path is None:
        parser.error("--dta-name-or-path is required when --mode d3")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    installed_flash_attn = _has_flash_attn()
    attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    dta_model = None
    if args.mode == "d3":
        dta_model = DFlashDraftModel.from_pretrained(
            args.dta_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    block_size_2 = args.block_size_2 if args.block_size_2 is not None else block_size
    if args.mode == "d3" and block_size_2 < block_size:
        parser.error(f"--block-size-2 ({block_size_2}) must be >= --block-size ({block_size})")

    dta_runner, verify_runner = _build_cascade_runners(args.mode, block_size_2, dta_model, target)

    if args.mode == "dflash":
        generator = DFlashGenerator(target=target, draft=draft_model, block_size=block_size)
    else:
        generator = D3Generator(
            target=target,
            draft=draft_model,
            dta=dta_model,
            block_size=block_size,
            block_size_2=block_size_2,
            dta_cascade_runner=dta_runner,
            verify_cascade_runner=verify_runner,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    if args.batch_size != 1 and dist.is_main():
        logger.warning(f"--batch-size={args.batch_size} requested but Phase 1 runs S=1 sequentially.")

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = generator.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )

            baseline = generator.baseline_generate(
                input_ids=input_ids,
                num_tokens=response.num_output_tokens,
                temperature=args.temperature,
            )
            response.baseline_time_per_token = baseline.time_per_output_token

            generated_ids = response.output_ids[0, response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    tb = np.mean([r.time_per_output_token for r in responses])
    label = "DFlash" if args.mode == "dflash" else "D3"
    print(f"Avg {label} time per output token: {tb * 1000:.2f} ms")

    tb_baseline = np.mean([r.baseline_time_per_token for r in responses])
    print(f"Avg baseline time per output token: {tb_baseline * 1000:.2f} ms")
    print(f"Speedup: {tb_baseline / tb:.2f}x")

    tau = np.mean([np.mean(r.acceptance_lengths) for r in responses if r.acceptance_lengths])
    print(f"Average Acceptance length: {tau:.2f}")

    histogram_span = block_size_2 + 1 if args.mode == "d3" else block_size + 1
    acceptance_lengths = list(chain(*[r.acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(histogram_span)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")


if __name__ == "__main__":
    main()
