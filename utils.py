"""
utils.py — Shared utility functions.
"""

import os
import random
import json
import logging
import numpy as np
import torch

# ─── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)

logger = get_logger(__name__)


# ─── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


# ─── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(model, path: str, metadata: dict = None):
    """Save model state dict + optional metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)
    logger.info(f"Saved checkpoint → {path}")


def load_checkpoint(model, path: str):
    """Load state dict into model."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=False, assign=True)
    meta = payload.get("metadata", {})
    logger.info(f"Loaded checkpoint ← {path}")
    return model, meta


# ─── JSON helpers ──────────────────────────────────────────────────────────────
def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info(f"Saved JSON → {path}")


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# ─── Accuracy ──────────────────────────────────────────────────────────────────
def compute_accuracy(predictions: list, labels: list) -> float:
    """Simple exact-match accuracy."""
    assert len(predictions) == len(labels), "Length mismatch"
    correct = sum(p.strip().lower() == l.strip().lower()
                  for p, l in zip(predictions, labels))
    return correct / len(labels) if labels else 0.0


# ─── Softmax / probs ───────────────────────────────────────────────────────────
def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


# ─── Flatten nested dict ───────────────────────────────────────────────────────
def flatten_dict(d: dict, parent_key="", sep="_") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def get_dataloader(samples, processor, batch_size=2, shuffle=True,
                   corrupt_labels=False, corrupt_answers=None):
    import config as _cfg
    active = getattr(_cfg, "ACTIVE_MODEL", "llava").lower()
    if active == "blip2":
        from data.blip2_dataset import make_dataloader_blip2
        return make_dataloader_blip2(
            samples, processor,
            batch_size=batch_size, shuffle=shuffle,
            corrupt_labels=corrupt_labels, corrupt_answers=corrupt_answers,
        )
    else:
        from data.dataset import make_dataloader
        return make_dataloader(
            samples, processor,
            batch_size=batch_size, shuffle=shuffle,
            corrupt_labels=corrupt_labels, corrupt_answers=corrupt_answers,
        )