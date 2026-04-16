"""
download_models.py
==================
One-command downloader for all models and datasets required to reproduce
the full benchmark. Run this ONCE before running main.py.

Requirements:
  - ~30 GB free disk space (LLaVA-7B ~14GB, BLIP-2 ~15GB, datasets ~1GB)
  - Internet connection
  - HuggingFace account (free) for gated models if needed

Usage:
    python download_models.py                    # download everything
    python download_models.py --models-only      # skip datasets
    python download_models.py --datasets-only    # skip models
    python download_models.py --blip2-only       # BLIP-2 only
"""

import argparse
import os
import sys

import config

# ── Optional: set HF token for gated models ───────────────────────────────────
# HF_TOKEN = os.environ.get("HF_TOKEN", None)   # uncomment if needed
HF_TOKEN = None


def download_llava():
    print("\n[1/5] Downloading LLaVA-1.5-7B processor + model (~14 GB) ...")
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        cache_dir=config.DATA_DIR,
        token=HF_TOKEN,
    )
    LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        cache_dir=config.DATA_DIR,
        token=HF_TOKEN,
    )
    print("  LLaVA-1.5-7B done ✓")


def download_blip2():
    print("\n[2/5] Downloading BLIP-2 OPT-2.7B processor + model (~15 GB) ...")
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir=config.DATA_DIR,
        token=HF_TOKEN,
    )
    Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        cache_dir=config.DATA_DIR,
        token=HF_TOKEN,
    )
    print("  BLIP-2 OPT-2.7B done ✓")


def download_mllmu():
    print("\n[3/5] Downloading MLLMU-Bench (~200 MB) ...")
    from datasets import load_dataset
    load_dataset(
        "franciscoliu/MLLMU-Bench",
        cache_dir=config.DATA_DIR,
        token=HF_TOKEN,
    )
    print("  MLLMU-Bench done ✓")


def download_unlok():
    print("\n[4/5] Downloading UnLOK-VQA (~150 MB) ...")
    from datasets import load_dataset
    try:
        load_dataset(
            "vpatil24/unlok-vqa",
            cache_dir=config.DATA_DIR,
            token=HF_TOKEN,
        )
        print("  UnLOK-VQA done ✓")
    except Exception as e:
        print(f"  WARNING: UnLOK-VQA download failed: {e}")
        print("  Try manually: https://huggingface.co/datasets/vpatil24/unlok-vqa")


def download_mmubench():
    print("\n[5/5] Downloading MMUBench (~300 MB) ...")
    from datasets import load_dataset
    try:
        load_dataset(
            "lmms-lab/MMBench_EN",
            cache_dir=config.DATA_DIR,
            token=HF_TOKEN,
        )
        print("  MMUBench done ✓")
    except Exception as e:
        print(f"  WARNING: MMUBench download failed: {e}")
        print("  Try manually: https://huggingface.co/datasets/lmms-lab/MMBench_EN")


def main():
    parser = argparse.ArgumentParser(
        description="Download all models and datasets for the benchmark."
    )
    parser.add_argument("--models-only",   action="store_true",
                        help="Download only LLaVA and BLIP-2, skip datasets")
    parser.add_argument("--datasets-only", action="store_true",
                        help="Download only datasets, skip models")
    parser.add_argument("--blip2-only",    action="store_true",
                        help="Download only BLIP-2 (e.g. if LLaVA already cached)")
    parser.add_argument("--no-blip2",      action="store_true",
                        help="Skip BLIP-2 download (saves ~15 GB)")
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)

    print(f"Download directory: {config.DATA_DIR}")
    print(f"Estimated total size: ~30 GB (LLaVA ~14 GB + BLIP-2 ~15 GB + datasets ~1 GB)")

    if args.blip2_only:
        download_blip2()
        return

    if not args.datasets_only:
        download_llava()
        if not args.no_blip2:
            download_blip2()

    if not args.models_only:
        download_mllmu()
        download_unlok()
        download_mmubench()

    print("\n" + "="*55)
    print("All downloads complete ✓")
    print("Next step: python main.py --stage all")
    print("="*55)


if __name__ == "__main__":
    main()