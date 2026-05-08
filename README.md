# D²SD: Dual-Diffuse Speculative Decoding

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/2026_D2SD_Arxiv.pdf)

> A dual-draft speculative-decoding framework that pairs a **DFlash** block-parallel draft with a **DTA (Dual Token Anchor)** re-sampling draft, then verifies merged candidate branches against the target model in a single cascade-attention forward pass.

---

## Highlights

- **Dual-draft pipeline.** A first DFlash draft proposes a block of candidate tokens conditioned on target hidden features. A second DTA draft re-samples the most uncertain positions to produce additional candidate branches.
- **Cascade verification.** The target model verifies all branches in one forward pass, sharing the prefix KV via [FlashInfer](https://github.com/flashinfer-ai/flashinfer) cascade attention; CUDA-graph capture removes per-layer kernel-launch overhead.
- **Plug-and-play with HuggingFace models.** Tested on Qwen3 and GPT-OSS targets; no surgery on the target weights.
- **Lossless decoding.** Greedy and temperature sampling are both supported and remain mathematically equivalent to standard autoregressive decoding from the target.
- **Reproducible benchmarks.** One script reproduces results across math, coding, and chat datasets (GSM8K, MATH-500, AIME-24/25, HumanEval, MBPP, LiveCodeBench, SWE-bench, MT-Bench, Alpaca).

## Repository layout

```
D2SD/
├── benchmark.py              # Unified driver for DFlash and D²SD (`--mode {dflash,d3}`)
├── distributed.py            # Thin torch.distributed helpers used by the driver
├── model/
│   ├── dflash.py             # DFlash draft model (Qwen3-based)
│   ├── cascade_graph.py      # CUDA-graph runner for cascade local-attn + merge
│   └── utils.py              # Sampling, layer-id selection, dataset loaders
├── generation/
│   ├── dflash_generator.py   # Single-draft (DFlash-only) generator
│   ├── d3_generator.py       # Dual-draft (DFlash + DTA) generator
│   ├── verification.py       # Cascade target verification (Qwen3 / GPT-OSS)
│   └── state.py              # Per-sequence generation state container
├── examples/
│   ├── run_benchmark.sh      # DFlash baseline sweep
│   └── run_benchmark_dd.sh   # D²SD sweep
├── paper/
│   └── 2026_D2SD_Arxiv.pdf
├── requirements.txt
└── LICENSE
```

## Installation

D²SD targets Linux + CUDA. We recommend Python 3.10 or 3.11.

```bash
# 1. Create a clean environment
conda create -n d2sd python=3.10 -y && conda activate d2sd

# 2. Install PyTorch (pick the wheel matching your CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Install the rest of the dependencies
pip install -r requirements.txt

# 4. (Optional, recommended) FlashAttention for faster target forward
pip install flash-attn --no-build-isolation
```

`flashinfer-python` is required (it powers cascade attention); `flash-attn` is auto-detected at runtime and the code falls back to `torch.sdpa` if it is not installed.

### Hardware

We have validated the benchmark on 8× NVIDIA H100/A100 GPUs. A single GPU is enough to run small batches; the example scripts default to 8 GPUs via `torchrun --nproc_per_node=8` and partition the dataset across ranks.

## Models

D²SD requires three checkpoints:

| Role | What it is | Example |
|------|------------|---------|
| **Target** | The HuggingFace causal-LM you want to accelerate. | `Qwen/Qwen3-8B` |
| **DFlash draft** | A small DFlash-trained model (block-parallel draft conditioned on target hidden states). | `qwen3-8b-dflash` |
| **DTA draft** *(D²SD only)* | A second draft trained to re-sample uncertain positions of the first draft. | `qwen3-8b-dta` |

You can train your own DFlash and DTA drafts following the procedure in the paper. We will release pre-trained checkpoints alongside the camera-ready release; in the meantime point `--draft-name-or-path` and `--dta-name-or-path` at your local checkpoints.

## Quick start

```bash
# Single-draft baseline (DFlash only) on GSM8K with 32 samples on 1 GPU
torchrun --nproc_per_node=1 --master_port=29600 benchmark.py \
    --mode dflash \
    --model-name-or-path /path/to/qwen3-8b \
    --draft-name-or-path /path/to/qwen3-8b-dflash \
    --dataset gsm8k --max-samples 32 --max-new-tokens 1024

# Dual-draft D²SD on GSM8K
torchrun --nproc_per_node=1 --master_port=29600 benchmark.py \
    --mode d3 \
    --model-name-or-path /path/to/qwen3-8b \
    --draft-name-or-path /path/to/qwen3-8b-dflash \
    --dta-name-or-path  /path/to/qwen3-8b-dta \
    --block-size 16 --block-size-2 32 \
    --dataset gsm8k --max-samples 32 --max-new-tokens 1024
```

The driver prints, for each run:

- average per-token latency (D²SD vs. plain target),
- end-to-end speedup,
- average accepted block length and a histogram,
- a per-stage breakdown (draft1 / draft2 / verify / other) in `%`, `ms/tok`, and `ms/iter`.

## Reproducing the paper

The two scripts under `examples/` reproduce the headline numbers. Both accept the same set of environment variables, so you can override paths, GPUs, block sizes, and the dataset list without editing the file.

```bash
# DFlash sweep across all datasets
GPUS=0,1,2,3,4,5,6,7 \
TARGET_MODEL=/path/to/qwen3-8b \
DRAFT_MODEL=/path/to/qwen3-8b-dflash \
bash examples/run_benchmark.sh

# D²SD sweep
GPUS=0,1,2,3,4,5,6,7 \
TARGET_MODEL=/path/to/qwen3-8b \
DRAFT_MODEL=/path/to/qwen3-8b-dflash \
DTA_MODEL=/path/to/qwen3-8b-dta \
BLOCK_SIZE=16 BLOCK_SIZE_2=32 \
bash examples/run_benchmark_dd.sh

# Pick your own datasets / per-task sample counts
TASKS="gsm8k:64,humaneval:32" bash examples/run_benchmark.sh
```

Logs land in `logs/<dataset>.log` (DFlash) and `logs/<dataset>_d2sd.log` (D²SD).

### Supported datasets

`--dataset` accepts: `gsm8k`, `math500`, `aime24`, `aime25`, `humaneval`, `mbpp`, `lbpp`, `livecodebench`, `swe-bench`, `alpaca`, `mt-bench`. Each is loaded from HuggingFace Hub on first use; ensure you have network access (or pre-cache them with `HF_DATASETS_CACHE`).

## Command-line reference

```
benchmark.py [-h] --mode {dflash,d3}
             --model-name-or-path TARGET
             --draft-name-or-path DRAFT
             [--dta-name-or-path DTA]            # required when --mode d3
             [--block-size BLOCK_SIZE]           # DFlash draft block (default: 16)
             [--block-size-2 BLOCK_SIZE_2]       # DTA / verify block (>= block-size)
             [--batch-size BATCH_SIZE]           # currently S=1; >1 runs sequentially
             [--dataset DATASET]
             [--max-samples MAX_SAMPLES]
             [--max-new-tokens MAX_NEW_TOKENS]
             [--temperature TEMPERATURE]
```

## Method (in brief)

1. **First draft (DFlash).** A lightweight Qwen3-flavoured model takes the target's most recent hidden states (selected layers, fused via a small linear) and predicts a block of `block_size` tokens in parallel.
2. **Branch selection.** Per-position confidences from the first draft are used to pick the top-k positions where the prediction is most likely to be wrong.
3. **Second draft (DTA).** A second draft re-samples each selected position and the suffix of the block, producing several candidate branches that share a common prefix.
4. **Cascade verification.** The target model forwards all branches in one pass, attending over a *shared* prefix KV plus per-branch local KV via FlashInfer cascade attention; a CUDA graph fuses the local-attn and LSE-merge kernels per layer. The longest matching branch is accepted.

See `paper/2026_D2SD_Arxiv.pdf` for the full description, ablations, and analysis.

## Citation

If D²SD is useful in your work, please cite us:

```bibtex
@article{d2sd2026,
  title  = {{D}$^2${SD}: Dual-Diffuse Speculative Decoding for Large Language Models},
  author = {The D2SD Authors},
  year   = {2026},
  note   = {Preprint, see paper/2026\_D2SD\_Arxiv.pdf}
}
```

## License

D²SD is released under the [Apache License 2.0](LICENSE).

## Acknowledgements

D²SD builds on top of the open-source ecosystem and would not be possible without it: [PyTorch](https://pytorch.org/), [HuggingFace Transformers](https://github.com/huggingface/transformers), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [SGLang](https://github.com/sgl-project/sglang), and the dataset hosts on HuggingFace Hub. We thank their authors and maintainers.
