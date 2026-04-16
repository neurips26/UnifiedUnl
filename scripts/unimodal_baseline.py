"""
scripts/unimodal_baseline.py — CIFAR-10 unimodal baseline pipeline.

Trains ResNet-18 on CIFAR-10, applies all 4 unlearning methods,
computes all 5 metrics, returns results in same format as multimodal.

Used for Finding 2: showing multimodal has lower metric agreement.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as tvm
import numpy as np
from typing import Dict, List
import copy
import os

from utils import get_logger, set_seed, save_json
import config

logger = get_logger(__name__)


# ─── ResNet-18 model ─────────────────────────────────────────────────────────
def get_resnet18(n_classes: int = 10) -> nn.Module:
    model = tvm.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(config.DEVICE)


# ─── Training ────────────────────────────────────────────────────────────────
def train_cifar(model, loader, n_epochs=config.CIFAR_EPOCHS,
                lr=config.CIFAR_LR, loss_sign=1) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    for epoch in range(1, n_epochs + 1):
        total = 0.0
        for X, y in loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            out  = model(X)
            loss = loss_sign * criterion(out, y)
            loss.backward()
            optimizer.step()
            total += criterion(out, y).item()
        scheduler.step()
        if epoch % 5 == 0:
            logger.info(f"[CIFAR] Epoch {epoch}/{n_epochs} — loss: {total/len(loader):.4f}")
    return model


# ─── Accuracy evaluation ─────────────────────────────────────────────────────
@torch.no_grad()
def eval_accuracy(model, loader) -> float:
    model.eval()
    correct = total = 0
    for X, y in loader:
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)
        preds = model(X).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / max(total, 1)


# ─── Confidence for MIA ──────────────────────────────────────────────────────
@torch.no_grad()
def get_confidences(model, loader) -> np.ndarray:
    model.eval()
    confs = []
    for X, _ in loader:
        X = X.to(config.DEVICE)
        probs = torch.softmax(model(X).float(), dim=-1)
        confs.extend(probs.max(1).values.cpu().numpy())
    return np.array(confs)


# ─── Activation distance ─────────────────────────────────────────────────────
@torch.no_grad()
def activation_distance(m1, m2, loader) -> float:
    m1.eval(); m2.eval()
    dists = []
    for X, _ in loader:
        X = X.to(config.DEVICE)
        # Extract penultimate layer features
        with torch.no_grad():
            a1 = m1.avgpool(m1.layer4(m1.layer3(m1.layer2(m1.layer1(
                m1.maxpool(m1.relu(m1.bn1(m1.conv1(X)))))))))
            a2 = m2.avgpool(m2.layer4(m2.layer3(m2.layer2(m2.layer1(
                m2.maxpool(m2.relu(m2.bn1(m2.conv1(X)))))))))
        a1 = a1.flatten(1).cpu()
        a2 = a2.flatten(1).cpu()
        dists.append(torch.norm(a1 - a2, dim=1).mean().item())
    return float(np.mean(dists))


# ─── Full unimodal pipeline ───────────────────────────────────────────────────
def run_unimodal_pipeline() -> List[Dict]:
    """
    Run complete unimodal baseline:
    - Train vanilla + retrained ResNet-18
    - Apply all 4 unlearning methods
    - Compute all 5 metrics
    Returns list of result dicts (same format as multimodal results).
    """
    from data.loader import load_cifar10
    from sklearn.linear_model import LogisticRegression

    all_results = []

    for seed in config.SEEDS:
        set_seed(seed)
        logger.info(f"[CIFAR] Seed {seed}")

        cifar = load_cifar10()

        forget_loader  = DataLoader(cifar["forget"],  batch_size=config.CIFAR_BATCH, shuffle=True)
        retain_loader  = DataLoader(cifar["retain"],  batch_size=config.CIFAR_BATCH, shuffle=True)
        test_loader    = DataLoader(cifar["test"],    batch_size=config.CIFAR_BATCH)
        forget_loader_eval = DataLoader(cifar["forget"], batch_size=config.CIFAR_BATCH)

        # ── Vanilla model ──
        vanilla = get_resnet18()
        train_cifar(vanilla, DataLoader(cifar["train"], batch_size=config.CIFAR_BATCH, shuffle=True))

        # ── Retrained model (no forget class) ──
        retrained = get_resnet18()
        train_cifar(retrained, retain_loader)

        # ── Unlearning methods ──
        unlearning_fns = {
            "gradient_ascent": lambda m: train_cifar(copy.deepcopy(m), forget_loader,
                                                      n_epochs=5, loss_sign=-1),
            "random_labels":   lambda m: _cifar_random_labels(copy.deepcopy(m),
                                                               cifar["forget"], seed),
            "finetune_retain": lambda m: train_cifar(copy.deepcopy(m), retain_loader,
                                                      n_epochs=5),
            "salun":           lambda m: _cifar_salun(copy.deepcopy(m),
                                                       cifar["forget"], cifar["retain"], seed),
        }

        for method_name, fn in unlearning_fns.items():
            logger.info(f"[CIFAR] Method: {method_name}")
            unlearned = fn(vanilla)

            fa  = 1.0 - eval_accuracy(unlearned, forget_loader_eval)
            ra  = eval_accuracy(unlearned, retain_loader)
            ad  = activation_distance(unlearned, retrained, forget_loader_eval)

            # MIA
            f_conf = get_confidences(unlearned, forget_loader_eval)
            r_conf = get_confidences(unlearned, retain_loader)
            X = np.concatenate([r_conf.reshape(-1,1), f_conf.reshape(-1,1)])
            y = np.concatenate([np.ones(len(r_conf)), np.zeros(len(f_conf))])
            clf = LogisticRegression(max_iter=500).fit(X, y)
            mia = clf.predict(f_conf.reshape(-1,1)).mean()

            # JS Divergence
            from scipy.spatial.distance import jensenshannon
            with torch.no_grad():
                all_js = []
                for X_b, _ in forget_loader_eval:
                    X_b = X_b.to(config.DEVICE)
                    p = torch.softmax(unlearned(X_b).float(), -1).cpu().numpy()
                    q = torch.softmax(retrained(X_b).float(), -1).cpu().numpy()
                    for pi, qi in zip(p, q):
                        all_js.append(jensenshannon(pi, qi)**2)
            js = float(np.mean(all_js))

            # FA = proportion of forget set predicted correctly (lower=better)
            fa_acc = eval_accuracy(unlearned, forget_loader_eval)

            all_results.append({
                "dataset": "cifar10",
                "method":  method_name,
                "seed":    seed,
                "FA":  fa_acc,
                "RA":  ra,
                "MIA": mia,
                "AD":  ad,
                "JS":  js,
            })
            logger.info(f"  FA={fa_acc:.4f} RA={ra:.4f} MIA={mia:.4f} "
                        f"AD={ad:.4f} JS={js:.4f}")

    save_json(all_results,
              os.path.join(config.RESULTS_DIR, "unimodal_results.json"))
    logger.info(f"Saved unimodal results ({len(all_results)} entries)")
    return all_results


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _cifar_random_labels(model, forget_ds, seed):
    set_seed(seed)
    import random
    class CorruptDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            X, y = self.ds[i]
            wrong = random.choice([c for c in range(10) if c != y])
            return X, wrong
    loader = DataLoader(CorruptDataset(forget_ds),
                        batch_size=config.CIFAR_BATCH, shuffle=True)
    return train_cifar(model, loader, n_epochs=5)


def _cifar_salun(model, forget_ds, retain_ds, seed):
    set_seed(seed)
    device = config.DEVICE
    forget_loader = DataLoader(forget_ds, batch_size=config.CIFAR_BATCH)
    retain_loader = DataLoader(retain_ds, batch_size=config.CIFAR_BATCH, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # Saliency map
    grad_accum = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    for X, y in forget_loader:
        X, y = X.to(device), y.to(device)
        model.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_accum[n] += p.grad.abs()

    all_g = torch.cat([g.flatten() for g in grad_accum.values()])
    cutoff = torch.quantile(all_g, 1.0 - config.SALUN_THRESHOLD)
    mask   = {n: (g >= cutoff).float() for n, g in grad_accum.items()}

    # Perturb salient weights
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in mask:
                p.data += mask[n] * torch.randn_like(p) * 0.01

    # Fine-tune retain
    for n, p in model.named_parameters():
        if n in mask:
            p.register_hook(lambda g, m=mask[n]: g * (1 - m.to(g.device)))

    model = train_cifar(model, retain_loader, n_epochs=5)
    for p in model.parameters():
        p._backward_hooks.clear()
    return model
