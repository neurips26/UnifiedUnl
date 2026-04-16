"""
unlearning/methods.py — All 4 unlearning methods.

Methods:
  1. Gradient Ascent (GA)       — flip loss sign on forget set
  2. Random Labels (RL)         — corrupt forget set labels, fine-tune
  3. Fine-tune on Retain (FT)   — train only on retain, rely on catastrophic forgetting
  4. SalUn                      — saliency-based weight masking + unlearning

Each method takes:
    model       — vanilla fine-tuned model (already trained on full dataset)
    forget_set  — list of VQA samples to forget
    retain_set  — list of VQA samples to retain
    processor   — LLaVA processor
    seed        — for reproducibility

Returns: unlearned model
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict
import copy
from tqdm import tqdm

from utils import get_dataloader as make_dataloader  # model-aware routing (LLaVA or BLIP-2)
from models.model_factory import full_train  # model-aware routing

def _get_train_one_epoch():
    """Return correct train_one_epoch for active model."""
    import config as _cfg
    if getattr(_cfg, 'ACTIVE_MODEL', 'llava').lower() == 'blip2':
        from models.blip2_model import train_one_epoch_blip2 as _fn
    else:
        from models.llava_model import train_one_epoch as _fn
    return _fn
from utils import get_logger, set_seed
import config

logger = get_logger(__name__)


# ─── Helper: deep copy model ─────────────────────────────────────────────────
def _copy(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


# ─── 1. Gradient Ascent ───────────────────────────────────────────────────────
def gradient_ascent(
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
) -> nn.Module:
    """
    Flip the cross-entropy loss sign on the forget set.
    Maximises loss on forget → degrades performance on forgotten data.
    Optionally adds retain regularisation to prevent total model collapse.
    """
    set_seed(seed)
    model = _copy(model)
    logger.info("[GA] Starting Gradient Ascent unlearning...")

    forget_loader = make_dataloader(
        forget_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
    )
    retain_loader = make_dataloader(
        retain_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.UNLEARN_LR,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.UNLEARN_EPOCHS)
    retain_iter = iter(retain_loader)

    model.train()
    device = config.DEVICE

    for epoch in range(1, config.UNLEARN_EPOCHS + 1):
        total_loss = 0.0

        pbar = tqdm(forget_loader, desc=f"[GA] Epoch {epoch}/{config.UNLEARN_EPOCHS}", unit="batch")
        for batch in pbar:
            optimizer.zero_grad()

            # ── Forget loss (ascent = negative cross-entropy) ──
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            labels         = batch["labels"].to(device)

            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels)
            forget_loss = -out.loss   # flip sign

            # ── Retain regularisation (optional but stabilises training) ──
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            r_out = model(
                input_ids      = r_batch["input_ids"].to(device),
                attention_mask = r_batch["attention_mask"].to(device),
                pixel_values   = r_batch["pixel_values"].to(device),
                labels         = r_batch["labels"].to(device),
            )
            retain_loss = r_out.loss

            loss = forget_loss + 0.5 * retain_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out.loss.item()
            pbar.set_postfix(forget_loss=f"{out.loss.item():.4f}")

        scheduler.step()
        logger.info(f"[GA] Epoch {epoch}/{config.UNLEARN_EPOCHS} — "
                    f"forget_loss: {total_loss/len(forget_loader):.4f}")

    logger.info("[GA] Unlearning complete.")
    return model


# ─── 2. Random Labels ─────────────────────────────────────────────────────────
def random_labels(
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
) -> nn.Module:
    """
    Replace forget set answers with random wrong answers, then fine-tune.
    Teaches model to associate forgotten questions with wrong answers.
    """
    set_seed(seed)
    model = _copy(model)
    logger.info("[RL] Starting Random Labels unlearning...")

    # Collect all possible answers from retain set
    all_answers = list({s["answer"] for s in retain_set})
    if len(all_answers) < 2:
        all_answers = ["unknown", "none", "other", "N/A"]

    # Build corrupted forget loader
    corrupt_loader = make_dataloader(
        forget_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
        corrupt_labels=True,
        corrupt_answers=all_answers,
    )

    # Fine-tune on corrupted forget set
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.UNLEARN_LR,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.UNLEARN_EPOCHS)

    model.train()
    for epoch in range(1, config.UNLEARN_EPOCHS + 1):
        loss = _get_train_one_epoch()(
            model, corrupt_loader, optimizer, scheduler,
            loss_sign=1,   # normal descent on wrong labels
        )
        logger.info(f"[RL] Epoch {epoch}/{config.UNLEARN_EPOCHS} — loss: {loss:.4f}")

    logger.info("[RL] Unlearning complete.")
    return model


# ─── 3. Fine-tune on Retain ──────────────────────────────────────────────────
def finetune_retain(
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
) -> nn.Module:
    """
    Continue training only on the retain set.
    Catastrophic forgetting handles the forget set passively.
    """
    set_seed(seed)
    model = _copy(model)
    logger.info("[FT] Starting Fine-tune on Retain unlearning...")

    retain_loader = make_dataloader(
        retain_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
    )

    model = full_train(
        model, retain_loader,
        n_epochs=config.UNLEARN_EPOCHS,
        lr=config.UNLEARN_LR,
        loss_sign=1,
    )

    logger.info("[FT] Unlearning complete.")
    return model


# ─── 4. SalUn ─────────────────────────────────────────────────────────────────
def salun(
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
    threshold: float = config.SALUN_THRESHOLD,
) -> nn.Module:
    """
    Saliency Unlearning (SalUn, Fan et al. ICLR 2024).
    Steps:
      1. Compute gradient saliency map on forget set
      2. Create binary mask: top threshold% most salient weights = 1
      3. Apply random perturbation ONLY to masked weights
      4. Fine-tune unmasked (retained) weights on retain set
    """
    set_seed(seed)
    model = _copy(model)
    logger.info("[SalUn] Starting Saliency Unlearning...")

    device = config.DEVICE

    # ── Step 1: Compute saliency map ──────────────────────────────────────────
    logger.info("[SalUn] Computing saliency map on forget set...")
    forget_loader = make_dataloader(
        forget_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=False,
    )

    model.eval()
    grad_accum = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_accum[name] = torch.zeros_like(param.data)

    for batch in tqdm(forget_loader, desc="[SalUn] Saliency map", unit="batch"):
        model.zero_grad()
        out = model(
            input_ids      = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
            pixel_values   = batch["pixel_values"].to(device),
            labels         = batch["labels"].to(device),
        )
        out.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_accum[name] += param.grad.data.abs()

    # Average across batches
    n = len(forget_loader)
    for name in grad_accum:
        grad_accum[name] /= max(n, 1)

    # ── Step 2: Build binary saliency mask ───────────────────────────────────
    logger.info(f"[SalUn] Building mask with threshold={threshold}...")
    all_grads = torch.cat([g.flatten() for g in grad_accum.values()])
    cutoff    = torch.quantile(all_grads, 1.0 - threshold)

    mask = {}
    for name, grad in grad_accum.items():
        mask[name] = (grad >= cutoff).float()

    # ── Step 3: Perturb masked (forget-salient) weights ──────────────────────
    logger.info("[SalUn] Perturbing salient weights...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                noise = torch.randn_like(param.data) * 0.01
                param.data += mask[name] * noise

    # ── Step 4: Fine-tune on retain set (unmasked weights) ───────────────────
    logger.info("[SalUn] Fine-tuning on retain set...")

    # Freeze salient weights, only update non-salient
    for name, param in model.named_parameters():
        if name in mask:
            param.register_hook(
                lambda grad, m=mask[name]: grad * (1.0 - m.to(grad.device))
            )

    # Cap retain set — SalUn only needs a small subset to restore utility
    import random as _random
    _random.seed(seed)
    salun_retain = retain_set if len(retain_set) <= 50 else _random.sample(retain_set, 50)
    logger.info(f"[SalUn] Using {len(salun_retain)} retain samples (capped at 50)")

    retain_loader = make_dataloader(
        salun_retain, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.UNLEARN_LR,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.UNLEARN_EPOCHS)

    model.train()
    salun_epochs = 1  # SalUn only needs 1 epoch — weight perturbation does the heavy lifting
    for epoch in range(1, salun_epochs + 1):
        loss = _get_train_one_epoch()(model, retain_loader, optimizer, scheduler)
        logger.info(f"[SalUn] Epoch {epoch}/{salun_epochs} — loss: {loss:.4f}")

    # Remove hooks safely
    try:
        for param in model.parameters():
            if hasattr(param, '_backward_hooks') and param._backward_hooks:
                param._backward_hooks.clear()
    except Exception:
        pass  # hooks already cleared or never set

    logger.info("[SalUn] Unlearning complete.")
    return model


# ─── Master dispatcher ────────────────────────────────────────────────────────
def run_unlearning(
    method_name: str,
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
) -> nn.Module:
    """
    Call the correct unlearning method by name.
    method_name: one of config.METHODS
    """
    dispatch = {
        "gradient_ascent":  gradient_ascent,
        "random_labels":    random_labels,
        "finetune_retain":  finetune_retain,
        "salun":            salun,
    }
    assert method_name in dispatch, \
        f"Unknown method: {method_name}. Choose from {list(dispatch)}"

    logger.info(f"=== Running unlearning method: {method_name} (seed={seed}) ===")
    return dispatch[method_name](model, forget_set, retain_set, processor, seed=seed)
