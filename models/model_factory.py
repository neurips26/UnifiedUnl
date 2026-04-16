"""
models/model_factory.py — Single import point for all model operations.

All of main.py, metrics.py, methods.py call through here.
Switching between LLaVA and BLIP-2 happens by setting config.ACTIVE_MODEL.

Usage:
    from models.model_factory import (
        load_model, apply_lora, full_train,
        generate_answers, extract_activations, extract_output_probs,
    )

Everything routes to the correct model wrapper automatically.
"""

import config
from utils import get_logger

logger = get_logger(__name__)


def _is_blip2():
    return getattr(config, "ACTIVE_MODEL", "llava").lower() == "blip2"


# ── Load ──────────────────────────────────────────────────────────────────────
def load_model(quantize: bool = False):
    """
    Load model + processor. Returns (model, processor).
    Automatically chooses LLaVA or BLIP-2 based on config.ACTIVE_MODEL.
    """
    if _is_blip2():
        from models.blip2_model import load_blip2
        return load_blip2(quantize=quantize)
    else:
        from models.llava_model import load_llava
        return load_llava(quantize=quantize)


# ── LoRA ──────────────────────────────────────────────────────────────────────
def apply_lora(model):
    """Apply LoRA adapters to the model."""
    if _is_blip2():
        from models.blip2_model import apply_lora_blip2
        return apply_lora_blip2(model)
    else:
        from models.llava_model import apply_lora as apply_lora_llava
        return apply_lora_llava(model)


# ── Training ──────────────────────────────────────────────────────────────────
def full_train(model, train_loader, n_epochs=None, lr=None,
               save_path=None, loss_sign=1, label=""):
    """Full training loop. Routes to correct model trainer."""
    n_epochs = n_epochs or config.FINETUNE_EPOCHS
    lr       = lr       or config.FINETUNE_LR

    if _is_blip2():
        from models.blip2_model import full_train_blip2
        return full_train_blip2(
            model, train_loader, n_epochs=n_epochs, lr=lr,
            save_path=save_path, loss_sign=loss_sign, label=label,
        )
    else:
        from models.llava_model import full_train as full_train_llava
        return full_train_llava(
            model, train_loader, n_epochs=n_epochs, lr=lr,
            save_path=save_path, loss_sign=loss_sign, label=label,
        )


# ── Inference ─────────────────────────────────────────────────────────────────
def generate_answers(model, processor, samples, batch_size=8,
                     device=None, desc="Generating"):
    """Generate text answers. Returns List[str]."""
    device = device or config.DEVICE
    if _is_blip2():
        from models.blip2_model import generate_answers_blip2
        return generate_answers_blip2(
            model, processor, samples, batch_size=batch_size,
            device=device, desc=desc,
        )
    else:
        from models.llava_model import generate_answers as gen_llava
        return gen_llava(
            model, processor, samples, batch_size=batch_size,
            device=device, desc=desc,
        )


def extract_activations(model, processor, samples, layer_idx=-2, device=None):
    """Extract penultimate-layer activations. Returns Tensor[N, hidden]."""
    device = device or config.DEVICE
    if _is_blip2():
        from models.blip2_model import extract_activations_blip2
        return extract_activations_blip2(
            model, processor, samples, layer_idx=layer_idx, device=device,
        )
    else:
        from models.llava_model import extract_activations as ea_llava
        return ea_llava(model, processor, samples, layer_idx=layer_idx, device=device)


def extract_output_probs(model, processor, samples, device=None):
    """Extract top-K output probabilities. Returns Tensor[N, K]."""
    device = device or config.DEVICE
    if _is_blip2():
        from models.blip2_model import extract_output_probs_blip2
        return extract_output_probs_blip2(model, processor, samples, device=device)
    else:
        from models.llava_model import extract_output_probs as eop_llava
        return eop_llava(model, processor, samples, device=device)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint(model, path):
    """Load a saved LoRA checkpoint into model."""
    from utils import load_checkpoint as _load
    return _load(model, path)


def checkpoint_prefix():
    """Return model-specific prefix for checkpoint filenames."""
    model_name = getattr(config, "ACTIVE_MODEL", "llava")
    return model_name  # e.g. "llava" or "blip2"


def get_model_name():
    """Return the active model's HF model ID string."""
    if _is_blip2():
        return "Salesforce/blip2-opt-2.7b"
    return config.MODEL_NAME
