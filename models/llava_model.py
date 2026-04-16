"""
models/llava_model.py -- LLaVA-1.5 wrapper with LoRA fine-tuning.

IMPORTANT: TRANSFORMERS_OFFLINE=1 must be set before importing this module.
This is done at the top of main.py. The model loads from the default HF
cache at C:/Users/<user>/.cache/huggingface/hub/ -- no internet needed.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Tuple
import copy
import time
from tqdm import tqdm

from utils import get_logger, save_checkpoint, load_checkpoint  # load_checkpoint re-exported for main.py
import config

logger = get_logger(__name__)


# ── Load model + processor ────────────────────────────────────────────────────
def load_llava(quantize: bool = False) -> Tuple:
    """Load LLaVA-1.5-7B from default HF cache. No internet needed."""
    logger.info(f"Loading {config.MODEL_NAME} from local HF cache...")

    # Model lives in data_cache/ inside the project folder.
    # Pass cache_dir explicitly so HF finds it regardless of default cache location.
    _cache = config.DATA_DIR
    logger.info(f"Cache dir: {_cache}")

    processor = AutoProcessor.from_pretrained(
        config.MODEL_NAME, use_fast=True, cache_dir=_cache,
    )

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            config.MODEL_NAME, quantization_config=bnb_config,
            device_map="auto", cache_dir=_cache,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            config.MODEL_NAME, torch_dtype=torch.float16,
            device_map={"": str(config.DEVICE)},  # direct to GPU, avoids meta tensors
            cache_dir=_cache,
        )

    logger.info("Model loaded.")

    # Enable gradient checkpointing — cuts VRAM by ~40% at cost of ~20% slower
    if config.GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    return model, processor


# ── LoRA ──────────────────────────────────────────────────────────────────────
def apply_lora(model) -> nn.Module:
    lora_config = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET, lora_dropout=config.LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, scheduler=None,
                    device=config.DEVICE, loss_sign=1, clip_grad=1.0,
                    epoch_label=""):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=epoch_label, unit="batch",
               dynamic_ncols=True, leave=True)
    for step, batch in enumerate(pbar, 1):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        pix   = batch["pixel_values"].to(device)
        lbls  = batch["labels"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            out = model(input_ids=ids, attention_mask=mask,
                        pixel_values=pix, labels=lbls)
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


def full_train(model, train_loader, n_epochs=config.FINETUNE_EPOCHS,
               lr=config.FINETUNE_LR, save_path=None, loss_sign=1,
               label=""):
    n_batches = len(train_loader)
    print(f"\n  Starting training: {n_epochs} epoch(s) x {n_batches} batches")
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        t_ep = time.time()
        tag  = f"{label} Epoch {epoch}/{n_epochs}" if label else f"Epoch {epoch}/{n_epochs}"
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               loss_sign=loss_sign, epoch_label=tag)
        elapsed = time.time() - t_ep
        logger.info(f"  Epoch {epoch}/{n_epochs} done | loss={loss:.4f} | "
                    f"time={elapsed/60:.1f} min")
    total = time.time() - t_start
    print(f"  Training complete in {total/60:.1f} min")
    if save_path:
        print(f"  Saving checkpoint -> {save_path}")
        save_checkpoint(model, save_path, metadata={"epochs": n_epochs, "lr": lr})
        print(f"  Checkpoint saved.")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_answers(model, processor, samples, batch_size=8, device=config.DEVICE, desc="Generating"):
    from tqdm import tqdm
    model.eval()
    predictions = []
    batches = range(0, len(samples), batch_size)
    for i in tqdm(batches, desc=desc, unit="batch", leave=False):
        b = samples[i: i + batch_size]
        prompts = [f"USER: <image>\n{s['question']}\nASSISTANT:" for s in b]
        inputs  = processor(text=prompts, images=[s["image"] for s in b],
                            return_tensors="pt", padding=True).to(device)
        out_ids = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                                 do_sample=False)
        for ids in out_ids:
            new_ids = ids[inputs["input_ids"].shape[1]:]
            predictions.append(
                processor.decode(new_ids, skip_special_tokens=True).strip()
            )
    return predictions


# ── Activation extraction ─────────────────────────────────────────────────────
@torch.no_grad()
def extract_activations(model, processor, samples, layer_idx=-2, device=config.DEVICE):
    model.eval()
    all_acts, captured = [], {}

    def hook_fn(m, inp, out):
        # out can be: tensor, tuple, or LlavaModelOutputWithPast
        if isinstance(out, torch.Tensor):
            act = out
        elif isinstance(out, tuple):
            act = out[0]
        elif hasattr(out, "last_hidden_state"):
            act = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states:
            act = out.hidden_states[-1]
        else:
            act = out[0]  # fallback
        captured["act"] = act.detach().cpu()

    # Resolve correct layer list for LLaVA (PEFT-wrapped)
    def _get_layers(m):
        # Unwrap PEFT if needed
        base = m.base_model.model if hasattr(m, "base_model") else m
        # LlavaForConditionalGeneration: .language_model.model.layers
        if hasattr(base, "language_model"):
            return base.language_model.model.layers
        # Fallback for plain LlamaModel
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers
        return list(base.children())
    layers = _get_layers(model)
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    handle = layers[layer_idx].register_forward_hook(hook_fn)

    from tqdm import tqdm
    for s in tqdm(samples, desc="[AD/JS] Extracting", unit="sample", leave=False):
        prompt = f"USER: <image>\n{s['question']}\nASSISTANT:"
        inputs = processor(text=prompt, images=s["image"],
                           return_tensors="pt").to(device)
        _ = model(**inputs)
        all_acts.append(captured["act"].mean(dim=1).squeeze(0))

    handle.remove()
    return torch.stack(all_acts, dim=0)


# ── Output probability extraction ─────────────────────────────────────────────
@torch.no_grad()
def extract_output_probs(model, processor, samples, device=config.DEVICE):
    model.eval()
    all_probs, TOP_K = [], 1000
    from tqdm import tqdm
    for s in tqdm(samples, desc="[AD/JS] Extracting", unit="sample", leave=False):
        prompt = f"USER: <image>\n{s['question']}\nASSISTANT:"
        inputs = processor(text=prompt, images=s["image"],
                           return_tensors="pt").to(device)
        out    = model(**inputs)
        probs  = torch.softmax(out.logits[:, -1, :].float(), dim=-1).squeeze(0)
        top_p, _ = torch.topk(probs, TOP_K)
        all_probs.append(top_p.cpu())
    return torch.stack(all_probs, dim=0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)