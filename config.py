# Force offline mode — no HTTP pings if models/data already cached
import os as _os
_os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
_os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

"""
config.py — Central configuration for all experiments.
Edit ONLY this file to change datasets, methods, seeds, paths.
"""

import os

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEEDS = [42, 123, 5508]

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
# data_cache/ is inside the project folder — model and datasets live here.
DATA_DIR = os.path.join(BASE_DIR, "data_cache")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR     = os.path.join(BASE_DIR, "outputs", "figures")

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME      = "llava-hf/llava-1.5-7b-hf"   # HuggingFace model ID
MAX_NEW_TOKENS  = 50
MAX_INPUT_LEN   = 768    # 576 image tokens + 192 text. NEVER go below 640 (LLaVA image=576).

# ─── Memory optimisation ────────────────────────────────────────────────────────
GRAD_CHECKPOINT = True   # Trades compute for memory — essential for 7B+LoRA
MIXED_PRECISION = True   # Use torch autocast fp16 during forward pass

# ─── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
LORA_TARGET     = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ─── Training ──────────────────────────────────────────────────────────────────
FINETUNE_LR         = 2e-5
FINETUNE_EPOCHS     = 1       # 1 epoch sufficient for LoRA on VQA benchmark
MAX_TRAIN_SAMPLES   = 300     # cap per dataset — enough for ranking analysis, avoids 9hr runs
FINETUNE_BATCH      = 2       # RTX 4090 24GB: keep at 2 for LLaVA-7B+LoRA
UNLEARN_LR          = 1e-5
UNLEARN_EPOCHS      = 2       # 2 epochs for unlearning
UNLEARN_BATCH       = 2
WARMUP_RATIO        = 0.03

# ─── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = ["mllmu_bench", "unlok_vqa", "mmubench"]
FORGET_RATIO = 0.1          # 10% of training set becomes forget set

# ─── Unlearning Methods ────────────────────────────────────────────────────────
METHODS = ["gradient_ascent", "random_labels", "finetune_retain", "salun"]

# ─── SalUn specific ────────────────────────────────────────────────────────────
SALUN_THRESHOLD = 0.5       # saliency mask threshold

# ─── MIA specific ──────────────────────────────────────────────────────────────
MIA_N_SHADOWS   = 4
MIA_BATCH       = 32

# ─── Unimodal Baseline (CIFAR-10) ──────────────────────────────────────────────
CIFAR_FORGET_CLASS  = 0     # forget class "airplane"
CIFAR_EPOCHS        = 30
CIFAR_LR            = 0.01
CIFAR_BATCH         = 128

# ─── UQS ───────────────────────────────────────────────────────────────────────
METRIC_NAMES = ["FA", "RA", "MIA", "AD", "JS"]
# Weights derived empirically from retrained-model correlation (Week 4)
# Set to equal weights initially; updated after correlation analysis
UQS_WEIGHTS  = [0.2, 0.2, 0.2, 0.2, 0.2]

# ─── Device ────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] Using device: {DEVICE}")


# Multi-model support
ACTIVE_MODEL     = "llava"
BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
BLIP2_LORA_TARGET = ["q_proj", "v_proj", "k_proj", "out_proj"]