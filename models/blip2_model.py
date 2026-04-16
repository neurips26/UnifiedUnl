"""
models/blip2_model.py — BLIP-2 wrapper with LoRA fine-tuning.

Drop-in replacement for llava_model.py. Exposes the exact same
function signatures so main.py / metrics.py need zero changes when
--model blip2 is used. All routing happens in model_factory.py.

BLIP-2 architecture:
  ViT-g/14  →  Q-Former (32 learned query tokens)  →  OPT-2.7B / FlanT5-XL
  We use: Salesforce/blip2-opt-2.7b  (6.7B total, ~same VRAM as LLaVA-7B)

Prompt format:
  LLaVA: "USER: <image>\n{question}\nASSISTANT:"
  BLIP-2: "Question: {question} Answer:"
  (No <image> token needed — processor handles vision separately)

Key differences vs LLaVA:
  1. Processor: Blip2Processor instead of AutoProcessor
  2. Model class: Blip2ForConditionalGeneration
  3. Forward pass: pixel_values handled separately (no interleaved tokens)
  4. Layer access: model.language_model.model.decoder.layers (OPT)
  5. Output: model.generate() returns same format
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Tuple
import copy
import time
from tqdm import tqdm

from utils import get_logger, save_checkpoint, load_checkpoint
import config

logger = get_logger(__name__)

# ── BLIP-2 model ID ───────────────────────────────────────────────────────────
BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"

# LoRA target modules for BLIP-2 OPT language model
BLIP2_LORA_TARGET = [
    "q_proj", "v_proj", "k_proj", "out_proj",  # OPT attention
]


# ── Load model + processor ────────────────────────────────────────────────────
def load_blip2(quantize: bool = False) -> Tuple:
    """Load BLIP-2 from local HF cache. No internet needed."""
    logger.info(f"Loading {BLIP2_MODEL_NAME} from local HF cache...")
    _cache = config.DATA_DIR
    logger.info(f"Cache dir: {_cache}")

    processor = Blip2Processor.from_pretrained(
        BLIP2_MODEL_NAME, cache_dir=_cache,
        local_files_only=True,
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": str(config.DEVICE)},
        cache_dir=_cache,
        local_files_only=True,
    )

    logger.info("BLIP-2 model loaded.")

    if config.GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    return model, processor


# ── LoRA ──────────────────────────────────────────────────────────────────────
def apply_lora_blip2(model) -> nn.Module:
    """Apply LoRA to BLIP-2's OPT language model decoder."""
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=BLIP2_LORA_TARGET,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Prompt formatting ─────────────────────────────────────────────────────────
def _make_prompt(question: str) -> str:
    """Format a question in BLIP-2's expected prompt style."""
    return f"Question: {question} Answer:"


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch_blip2(model, dataloader, optimizer, scheduler=None,
                           device=config.DEVICE, loss_sign=1, clip_grad=1.0,
                           epoch_label=""):
    """Single training epoch for BLIP-2. Same interface as LLaVA version."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=epoch_label, unit="batch",
                dynamic_ncols=True, leave=True)

    for step, batch in enumerate(pbar, 1):
        # BLIP-2 batch format from make_dataloader_blip2
        pixel_values = batch["pixel_values"].to(device)
        input_ids    = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels       = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            out = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = loss_sign * out.loss
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += out.loss.item()
        running_avg = total_loss / step
        pbar.set_postfix({"loss": f"{out.loss.item():.4f}",
                          "avg":  f"{running_avg:.4f}"})

    return total_loss / max(len(dataloader), 1)


def full_train_blip2(model, train_loader, n_epochs=config.FINETUNE_EPOCHS,
                      lr=config.FINETUNE_LR, save_path=None, loss_sign=1,
                      label=""):
    """Full training loop for BLIP-2. Same interface as LLaVA version."""
    n_batches = len(train_loader)
    print(f"\n  Starting training: {n_epochs} epoch(s) x {n_batches} batches")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        t_ep = time.time()
        tag  = f"{label} Epoch {epoch}/{n_epochs}" if label else f"Epoch {epoch}/{n_epochs}"
        loss = train_one_epoch_blip2(model, train_loader, optimizer, scheduler,
                                      loss_sign=loss_sign, epoch_label=tag)
        elapsed = time.time() - t_ep
        logger.info(f"  Epoch {epoch}/{n_epochs} done | loss={loss:.4f} | "
                    f"time={elapsed/60:.1f} min")

    total = time.time() - t_start
    print(f"  Training complete in {total/60:.1f} min")

    if save_path:
        save_checkpoint(model, save_path, metadata={"epochs": n_epochs, "lr": lr})

    return model


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_answers_blip2(model, processor, samples, batch_size=8,
                             device=config.DEVICE, desc="Generating"):
    """
    Generate text answers for a list of VQA samples using BLIP-2.
    Same return format as LLaVA version: List[str]
    """
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(samples), batch_size), desc=desc,
                  unit="batch", leave=False):
        batch = samples[i: i + batch_size]
        prompts = [_make_prompt(s["question"]) for s in batch]
        images  = [s["image"] for s in batch]

        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
        )

        # Decode only the newly generated tokens
        for j, ids in enumerate(out_ids):
            # BLIP-2 returns only the generated tokens (no prompt)
            pred = processor.decode(ids, skip_special_tokens=True).strip()
            predictions.append(pred)

    return predictions


# ── Activation extraction ─────────────────────────────────────────────────────
@torch.no_grad()
def extract_activations_blip2(model, processor, samples, layer_idx=-2,
                                device=config.DEVICE):
    """
    Extract penultimate-layer activations from BLIP-2's OPT decoder.
    Same return format as LLaVA version: Tensor[N, hidden_dim]
    """
    model.eval()
    all_acts, captured = [], {}

    def hook_fn(m, inp, out):
        if isinstance(out, torch.Tensor):
            act = out
        elif isinstance(out, tuple):
            act = out[0]
        elif hasattr(out, "last_hidden_state"):
            act = out.last_hidden_state
        else:
            act = out[0]
        captured["act"] = act.detach().cpu()

    # BLIP-2 OPT: model.language_model.model.decoder.layers
    def _get_layers(m):
        base = m.base_model.model if hasattr(m, "base_model") else m
        if hasattr(base, "language_model"):
            lm = base.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                return lm.model.decoder.layers   # OPT
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers           # Flan-T5 decoder
        raise AttributeError("Cannot find decoder layers in BLIP-2 model")

    layers = _get_layers(model)
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    handle = layers[layer_idx].register_forward_hook(hook_fn)

    for s in tqdm(samples, desc="[AD/JS BLIP2] Extracting",
                  unit="sample", leave=False):
        prompt = _make_prompt(s["question"])
        inputs = processor(
            images=s["image"],
            text=prompt,
            return_tensors="pt",
        ).to(device)
        _ = model(**inputs)
        all_acts.append(captured["act"].mean(dim=1).squeeze(0))

    handle.remove()
    return torch.stack(all_acts, dim=0)


# ── Output probability extraction ─────────────────────────────────────────────
@torch.no_grad()
def extract_output_probs_blip2(model, processor, samples,
                                device=config.DEVICE):
    """
    Extract top-1000 output probabilities from BLIP-2.
    Same return format as LLaVA version: Tensor[N, 1000]
    """
    model.eval()
    all_probs, TOP_K = [], 1000

    for s in tqdm(samples, desc="[JS BLIP2] Extracting",
                  unit="sample", leave=False):
        prompt = _make_prompt(s["question"])
        inputs = processor(
            images=s["image"],
            text=prompt,
            return_tensors="pt",
        ).to(device)

        out   = model(**inputs)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1).squeeze(0)
        top_p, _ = torch.topk(probs, min(TOP_K, probs.shape[-1]))
        all_probs.append(top_p.cpu())

    return torch.stack(all_probs, dim=0)
