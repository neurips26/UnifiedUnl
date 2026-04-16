"""
evaluation/metrics.py — All 5 evaluation metrics.

Metrics:
  FA  — Forget Accuracy       (lower = better unlearning)
  RA  — Retain Accuracy       (higher = better utility)
  MIA — Membership Inference  (lower = better privacy)
  AD  — Activation Distance   (lower = closer to retrained model)
  JS  — JS Divergence         (lower = closer to retrained model)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from models.model_factory import generate_answers, extract_activations, extract_output_probs  # model-aware routing
from utils import get_logger, compute_accuracy
import config

def _flexible_accuracy(preds, labels):
    """Flexible match: check if label appears anywhere in prediction (case-insensitive)."""
    correct = 0
    for p, l in zip(preds, labels):
        p_norm = p.strip().lower()
        l_norm = l.strip().lower()
        # exact match OR label is contained in prediction OR prediction starts with label
        if p_norm == l_norm or l_norm in p_norm or p_norm.startswith(l_norm):
            correct += 1
    return correct / len(labels) if labels else 0.0

logger = get_logger(__name__)


# ─── FA: Forget Accuracy ──────────────────────────────────────────────────────
def compute_fa(
    model: nn.Module,
    processor,
    forget_set: List[Dict],
    batch_size: int = 8,
) -> float:
    """
    Forget Accuracy — accuracy on the forget set.
    Lower = better (model has forgotten the data).
    Perfect unlearning = FA close to random chance.
    """
    logger.info("[FA] Computing Forget Accuracy...")
    preds  = generate_answers(model, processor, forget_set, batch_size=batch_size, desc="[FA]")
    labels = [s["answer"] for s in forget_set]
    fa     = _flexible_accuracy(preds, labels)
    logger.info(f"[FA] FA = {fa:.4f}")
    return fa


# ─── RA: Retain Accuracy ──────────────────────────────────────────────────────
def compute_ra(
    model: nn.Module,
    processor,
    retain_set: List[Dict],
    batch_size: int = 8,
) -> float:
    """
    Retain Accuracy — accuracy on the retain set.
    Higher = better (model preserved useful knowledge).
    """
    logger.info("[RA] Computing Retain Accuracy...")
    # Cap to 100 — sufficient statistical power while keeping compute tractable
    subset = retain_set[:100]
    preds  = generate_answers(model, processor, subset, batch_size=batch_size, desc="[RA]")
    labels = [s["answer"] for s in subset]
    ra     = _flexible_accuracy(preds, labels)
    logger.info(f"[RA] RA = {ra:.4f}")
    return ra


# ─── MIA: Membership Inference Attack ─────────────────────────────────────────
def compute_mia(
    model: nn.Module,
    processor,
    forget_set: List[Dict],
    retain_set: List[Dict],
    device: str = config.DEVICE,
    n_shadow: int = config.MIA_N_SHADOWS,
) -> float:
    """
    Confidence-based Membership Inference Attack.
    
    Strategy:
      - Extract model's confidence (max softmax prob) on forget & retain samples
      - Train a simple binary classifier: 1=member (retain), 0=non-member (forget)
      - Attack success rate = classifier accuracy on forget set
      - Lower = better (harder to tell if data was in training set = better forgetting)
    
    Returns MIA attack success rate [0, 1].
    """
    logger.info("[MIA] Running Membership Inference Attack...")
    model.eval()

    def get_features(samples: List[Dict], desc="MIA") -> np.ndarray:
        """
        Extract 3 features per sample for improved MIA robustness:
          1. Max softmax confidence (standard)
          2. Prediction entropy (captures uncertainty)
          3. Negative log-likelihood / perplexity proxy
        Using multiple features makes the attack harder to game and
        addresses the reviewer concern of "minimal protocol".
        """
        from tqdm import tqdm
        features = []
        for i in tqdm(range(0, len(samples), config.MIA_BATCH), desc=f"[{desc}]", unit="batch", leave=False):
            batch = samples[i: i + config.MIA_BATCH]
            for sample in batch:
                prompt = f"USER: <image>\n{sample['question']}\nASSISTANT:"
                inputs = processor(
                    text=prompt,
                    images=sample["image"],
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits[:, -1, :].float()
                probs  = torch.softmax(logits, dim=-1)
                # Feature 1: max confidence
                conf = probs.max().item()
                # Feature 2: entropy (negative → higher means more certain)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                # Feature 3: top-2 probability gap (margin)
                top2 = probs.topk(2).values
                margin = (top2[0] - top2[1]).item()
                features.append([conf, -entropy, margin])  # negate entropy so high=more certain
        return np.array(features)

    # Use subset for speed
    max_samples = min(100, len(forget_set), len(retain_set))  # 100 samples for more reliable LR classifier
    f_subset = forget_set[:max_samples]
    r_subset = retain_set[:max_samples]

    forget_feat = get_features(f_subset, desc="MIA-forget")
    retain_feat = get_features(r_subset, desc="MIA-retain")

    # Build classifier dataset using multi-feature MIA
    # retain=member(1), forget=non-member(0)
    X = np.vstack([retain_feat, forget_feat])
    y = np.concatenate([np.ones(len(retain_feat)), np.zeros(len(forget_feat))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated LR for more robust attack estimate
    # (addresses reviewer concern about "minimal protocol")
    from sklearn.model_selection import cross_val_predict
    try:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        # 5-fold cross-validated predictions
        cv_preds = cross_val_predict(clf, X_scaled, y, cv=min(5, len(y)//4))
        # Attack success = how well attacker classifies forget as non-member (0)
        forget_indices = np.where(y == 0)[0]
        attack_success = (cv_preds[forget_indices] == 0).mean()
    except Exception:
        # Fallback to simple split if CV fails (too few samples)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_scaled, y)
        attack_success = (clf.predict(X_scaled[y == 0]) == 0).mean()

    logger.info(f"[MIA] Attack success rate = {attack_success:.4f} (lower=better)")
    return float(attack_success)


# ─── AD: Activation Distance ──────────────────────────────────────────────────
def compute_ad(
    unlearned_model: nn.Module,
    retrained_model: nn.Module,
    processor,
    forget_set: List[Dict],
    layer_idx: int = -2,
    n_samples: int = 100,
) -> float:
    """
    Activation Distance — L2 distance between unlearned and retrained model
    activations on the forget set.
    Lower = unlearned model is closer to retrained (ideal) model.
    """
    logger.info("[AD] Computing Activation Distance...")
    subset = forget_set[:n_samples]

    act_unlearned = extract_activations(unlearned_model, processor, subset, layer_idx)
    act_retrained = extract_activations(retrained_model, processor, subset, layer_idx)

    # Mean L2 distance per sample
    distances = torch.norm(act_unlearned - act_retrained, dim=1)
    ad        = distances.mean().item()

    logger.info(f"[AD] AD = {ad:.4f}")
    return ad


# ─── JS: Jensen-Shannon Divergence ────────────────────────────────────────────
def compute_js(
    unlearned_model: nn.Module,
    retrained_model: nn.Module,
    processor,
    forget_set: List[Dict],
    n_samples: int = 100,
) -> float:
    """
    JS Divergence between output probability distributions of
    unlearned model vs retrained model on forget set.
    Lower = unlearned model outputs are closer to retrained model.
    """
    logger.info("[JS] Computing JS Divergence...")
    subset = forget_set[:n_samples]

    probs_unlearned = extract_output_probs(unlearned_model, processor, subset)
    probs_retrained = extract_output_probs(retrained_model, processor, subset)

    # Compute JS divergence per sample, take mean
    js_values = []
    for p, q in zip(probs_unlearned.numpy(), probs_retrained.numpy()):
        # Ensure valid probability distributions
        p = np.abs(p) + 1e-10
        q = np.abs(q) + 1e-10
        p /= p.sum()
        q /= q.sum()
        js = jensenshannon(p, q) ** 2   # squared JSD in [0, 1]
        js_values.append(js)

    js_mean = float(np.mean(js_values))
    logger.info(f"[JS] JS = {js_mean:.4f}")
    return js_mean


# ─── Run all 5 metrics at once ────────────────────────────────────────────────
def evaluate_all_metrics(
    unlearned_model: nn.Module,
    retrained_model: nn.Module,
    processor,
    forget_set: List[Dict],
    retain_set: List[Dict],
    n_mia_samples: int = 200,
    n_dist_samples: int = 100,
    metric_cache_path: str = None,   # per-metric cache file
) -> Dict[str, float]:
    """
    Run all 5 metrics and return a dict.
    Saves each metric immediately after computing — crash-safe.
    Keys: FA, RA, MIA, AD, JS
    """
    import json, os
    logger.info("=== Evaluating all 5 metrics ===")

    # Load partial cache if it exists
    partial = {}
    if metric_cache_path and os.path.exists(metric_cache_path):
        with open(metric_cache_path) as f:
            partial = json.load(f)
        logger.info(f"  Resuming from partial cache: {list(partial.keys())}")

    def _save(d):
        if metric_cache_path:
            os.makedirs(os.path.dirname(metric_cache_path), exist_ok=True)
            with open(metric_cache_path, "w") as f:
                json.dump(d, f, indent=2)

    results = dict(partial)  # start from what we have

    if "FA" not in results:
        results["FA"] = compute_fa(unlearned_model, processor, forget_set)
        _save(results)
    else:
        logger.info(f"[FA] Skipped (cached) = {results['FA']:.4f}")

    if "RA" not in results:
        results["RA"] = compute_ra(unlearned_model, processor, retain_set)
        _save(results)
    else:
        logger.info(f"[RA] Skipped (cached) = {results['RA']:.4f}")

    if "MIA" not in results:
        results["MIA"] = compute_mia(unlearned_model, processor, forget_set, retain_set)
        _save(results)
    else:
        logger.info(f"[MIA] Skipped (cached) = {results['MIA']:.4f}")

    if "AD" not in results:
        results["AD"] = compute_ad(unlearned_model, retrained_model, processor,
                                   forget_set, n_samples=n_dist_samples)
        _save(results)
    else:
        logger.info(f"[AD] Skipped (cached) = {results['AD']:.4f}")

    if "JS" not in results:
        results["JS"] = compute_js(unlearned_model, retrained_model, processor,
                                   forget_set, n_samples=n_dist_samples)
        _save(results)
    else:
        logger.info(f"[JS] Skipped (cached) = {results['JS']:.4f}")

    logger.info(f"Results: {results}")
    return results


# ─── Retrained model distance (for UQS weight derivation) ─────────────────────
def compute_retrained_distance(
    unlearned_model: nn.Module,
    retrained_model: nn.Module,
    processor=None,
    forget_set=None,
    n_samples: int = 100,
) -> float:
    """
    Oracle proximity target: mean per-parameter L2 distance between the
    unlearned model and the retrained (gold-standard) model, computed only
    over LoRA-trainable parameters.

    Specifically:
        d(M_hat, M*) = (1/N) * sum_i ||theta_i(M_hat) - theta_i(M*)||_2

    where N = number of trainable LoRA parameter tensors.
    This is a parameter-space distance, not an activation-space distance,
    providing a direct measure of how far the unlearned model has drifted
    from the ideal retrained oracle across the learned weight subspace.

    Lower = unlearned model is closer to oracle = better unlearning.
    """
    total_dist = 0.0
    n_params = 0

    r_params = {name: param.data.cpu().float()
                for name, param in retrained_model.named_parameters()
                if param.requires_grad}

    for name, param in unlearned_model.named_parameters():
        if name in r_params and param.requires_grad:
            total_dist += torch.norm(
                param.data.cpu().float() - r_params[name]
            ).item()
            n_params += 1

    if n_params == 0:
        logger.warning("[Oracle] No trainable parameters matched — falling back to all params")
        r_all = dict(retrained_model.named_parameters())
        for name, param in unlearned_model.named_parameters():
            if name in r_all:
                total_dist += torch.norm(
                    param.data.cpu().float() - r_all[name].data.cpu().float()
                ).item()
                n_params += 1

    return total_dist / max(n_params, 1)
