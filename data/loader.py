"""
data/loader.py — Load MLLMU-Bench, UnLOK-VQA, MMUBench, and CIFAR-10.
Each dataset returns a dict:
    {
        "train":   [{"image": PIL, "question": str, "answer": str}, ...],
        "forget":  [...],
        "retain":  [...],
        "test":    [...],
    }
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np

from utils import get_logger, set_seed
import config

logger = get_logger(__name__)


# ─── Generic split helper ──────────────────────────────────────────────────────
def make_forget_retain_split(samples: List[dict], forget_ratio: float, seed: int = 42):
    """Randomly split samples into forget / retain sets."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    n_forget = max(1, int(len(shuffled) * forget_ratio))
    forget  = shuffled[:n_forget]
    retain  = shuffled[n_forget:]
    return forget, retain


# ─── MLLMU-Bench ──────────────────────────────────────────────────────────────
def load_mllmu_bench(split: str = "all") -> dict:
    """
    Load MLLMU-Bench from the HF arrow cache by finding .arrow files directly.
    Bypasses lock files entirely (Windows 260-char path limit workaround).
    """
    import os, glob
    from datasets import Dataset

    def _find_arrow_files(config_name: str):
        """Walk the cache and return all .arrow files for a given config."""
        base = os.path.join(config.DATA_DIR, "MLLMMU___mllmu-bench", config_name)
        if not os.path.isdir(base):
            return []
        arrows = glob.glob(os.path.join(base, "**", "*.arrow"), recursive=True)
        # Filter out index files, keep data files
        arrows = [a for a in arrows if not a.endswith("_indices.arrow")]
        return sorted(arrows)

    def parse(example):
        img = example.get("image", None)
        if img is None or not isinstance(img, Image.Image):
            img = Image.new("RGB", (336, 336), color=(128, 128, 128))
        else:
            img = img.convert("RGB")
        question = str(example.get("question") or example.get("Question") or "")
        answer   = str(example.get("answer")   or example.get("Answer")   or "")
        return {"image": img, "question": question,
                "answer": answer, "source": "mllmu_bench"}

    def load_config(config_name: str):
        arrows = _find_arrow_files(config_name)
        if not arrows:
            raise FileNotFoundError(f"No .arrow files found for {config_name}")
        logger.info(f"  {config_name}: found {len(arrows)} arrow file(s)")
        # Load each arrow file as a Dataset and concatenate
        from datasets import concatenate_datasets
        parts = [Dataset.from_file(a) for a in arrows]
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    try:
        logger.info("Loading MLLMU-Bench from arrow cache...")

        forget_ds = load_config("forget_10")
        retain_ds = load_config("retain_90")
        test_ds   = load_config("Test_Set")

        forget_samples = [parse(e) for e in forget_ds]
        retain_samples = [parse(e) for e in retain_ds]
        test_samples   = [parse(e) for e in test_ds]

        logger.info(f"MLLMU-Bench loaded: forget={len(forget_samples)}, "
                    f"retain={len(retain_samples)}, test={len(test_samples)}")
        return {
            "train":  forget_samples + retain_samples,
            "forget": forget_samples,
            "retain": retain_samples,
            "test":   test_samples,
        }

    except Exception as e:
        logger.warning(f"MLLMU-Bench load failed ({e}). Using synthetic fallback.")
        return _synthetic_vqa_dataset("mllmu_bench")


def load_unlok_vqa() -> Dict[str, List[dict]]:
    """
    Load UnLOK-VQA from arrow cache or download if not present.
    Correct HF path: vaidehi99/UnLOK-VQA
    Paper: Patil et al., arXiv 2505.01456
    """
    import glob
    from datasets import Dataset, concatenate_datasets

    HF_PATH    = "vaidehi99/UnLOK-VQA"
    # HF cache folder name: "/" -> "---" and special chars encoded
    CACHE_NAME = "vaidehi99___un_lok-vqa"   # HF encodes hyphen differently

    def _find_arrows():
        base = os.path.join(config.DATA_DIR, CACHE_NAME)
        if not os.path.isdir(base):
            return []
        arrows = glob.glob(os.path.join(base, "**", "*.arrow"), recursive=True)
        return [a for a in sorted(arrows) if not a.endswith("_indices.arrow")]

    def parse(example):
        img = example.get("image", None)
        if img is None or not isinstance(img, Image.Image):
            img = Image.new("RGB", (336, 336), color=(100, 149, 237))
        else:
            img = img.convert("RGB")
        question = str(example.get("question") or example.get("Question") or "")
        answer   = str(example.get("answer")   or example.get("Answer")   or "")
        return {"image": img, "question": question,
                "answer": answer, "source": "unlok_vqa"}

    try:
        logger.info("Loading UnLOK-VQA from arrow cache...")
        arrows = _find_arrows()

        if arrows:
            logger.info(f"  Found {len(arrows)} arrow file(s) in cache")
            parts = [Dataset.from_file(a) for a in arrows]
            ds_all = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            all_samples = [parse(e) for e in ds_all]
        else:
            logger.info(f"  Not cached — downloading {HF_PATH}...")
            from datasets import load_dataset
            import os as _os
            # Temporarily re-enable network for this download
            _os.environ["HF_DATASETS_OFFLINE"] = "0"
            ds = load_dataset(HF_PATH, cache_dir=config.DATA_DIR,
                              token=_os.environ.get("HF_TOKEN"))
            _os.environ["HF_DATASETS_OFFLINE"] = "1"
            split_key = "train" if "train" in ds else list(ds.keys())[0]
            all_samples = [parse(e) for e in ds[split_key]]

        cap = getattr(config, "MAX_TRAIN_SAMPLES", len(all_samples))
        all_samples = all_samples[:cap]
        forget, retain = make_forget_retain_split(all_samples, config.FORGET_RATIO)
        test_samples   = all_samples[:min(200, len(all_samples))]

        logger.info(f"UnLOK-VQA loaded: total={len(all_samples)}, "
                    f"forget={len(forget)}, retain={len(retain)}")
        return {"train": all_samples, "forget": forget,
                "retain": retain, "test": test_samples}

    except Exception as e:
        logger.warning(f"UnLOK-VQA load failed ({e}). Using synthetic fallback.")
        return _synthetic_vqa_dataset("unlok_vqa")


# ─── MMUBench ─────────────────────────────────────────────────────────────────
def load_mmubench() -> Dict[str, List[dict]]:
    """
    Load MMUBench from arrow cache or download if not present.
    HF path: linhx/MMUBench  (NeurIPS 2024, Li et al.)
    1000 images across 20 concepts (50 images/concept).
    """
    import glob
    from datasets import Dataset, concatenate_datasets

    HF_PATH    = "lmms-lab/MMBench_EN"
    CACHE_NAME = "lmms-lab___mm_bench_en"   # actual folder in data_cache

    def _find_arrows():
        base = os.path.join(config.DATA_DIR, CACHE_NAME)
        if not os.path.isdir(base):
            return []
        all_arrows = glob.glob(os.path.join(base, "**", "*.arrow"), recursive=True)
        return [a for a in sorted(all_arrows) if not a.endswith("_indices.arrow")]

    def parse(example):
        img = example.get("image", None)
        if img is None or not isinstance(img, Image.Image):
            img = Image.new("RGB", (336, 336), color=(144, 238, 144))
        else:
            img = img.convert("RGB")
        # MMUBench field names vary — try multiple
        question = str(example.get("question") or example.get("Question")
                       or example.get("caption") or "What is in this image?")
        answer   = str(example.get("answer")   or example.get("Answer")
                       or example.get("label")  or "")
        return {"image": img, "question": question,
                "answer": answer, "source": "mmubench"}

    try:
        logger.info("Loading MMUBench from arrow cache...")
        arrows = _find_arrows()

        if arrows:
            logger.info(f"  Found {len(arrows)} arrow file(s) in cache")
            parts = [Dataset.from_file(a) for a in arrows]
            ds_all = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            all_samples = [parse(e) for e in ds_all]
        else:
            logger.info(f"  Not cached — downloading {HF_PATH}...")
            from datasets import load_dataset
            import os as _os
            _os.environ["HF_DATASETS_OFFLINE"] = "0"
            ds = load_dataset(HF_PATH, cache_dir=config.DATA_DIR,
                              token=_os.environ.get("HF_TOKEN"))
            _os.environ["HF_DATASETS_OFFLINE"] = "1"
            split_key = "train" if "train" in ds else list(ds.keys())[0]
            all_samples = [parse(e) for e in ds[split_key]]

        cap = getattr(config, "MAX_TRAIN_SAMPLES", len(all_samples))
        all_samples = all_samples[:cap]
        forget, retain = make_forget_retain_split(all_samples, config.FORGET_RATIO)
        test_samples   = all_samples[:min(200, len(all_samples))]

        logger.info(f"MMUBench loaded: total={len(all_samples)}, "
                    f"forget={len(forget)}, retain={len(retain)}")
        return {"train": all_samples, "forget": forget,
                "retain": retain, "test": test_samples}

    except Exception as e:
        logger.warning(f"MMUBench load failed ({e}). Using synthetic fallback.")
        return _synthetic_vqa_dataset("mmubench")


# ─── CIFAR-10 (Unimodal baseline) ──────────────────────────────────────────────
def load_cifar10():
    """Load CIFAR-10 with forget class = config.CIFAR_FORGET_CLASS."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=transform)

    forget_idx = [i for i, (_, y) in enumerate(train_ds)
                  if y == config.CIFAR_FORGET_CLASS]
    retain_idx = [i for i, (_, y) in enumerate(train_ds)
                  if y != config.CIFAR_FORGET_CLASS]

    from torch.utils.data import Subset
    forget_set = Subset(train_ds, forget_idx)
    retain_set = Subset(train_ds, retain_idx)

    logger.info(f"CIFAR-10: forget={len(forget_set)}, retain={len(retain_set)}, "
                f"test={len(test_ds)}")
    return {
        "train":  train_ds,
        "forget": forget_set,
        "retain": retain_set,
        "test":   test_ds,
    }


# ─── Master loader ────────────────────────────────────────────────────────────
def load_dataset_by_name(name: str) -> Dict[str, List[dict]]:
    loaders = {
        "mllmu_bench": load_mllmu_bench,
        "unlok_vqa":   load_unlok_vqa,
        "mmubench":    load_mmubench,
    }
    assert name in loaders, f"Unknown dataset: {name}. Choose from {list(loaders)}"
    return loaders[name]()


# ─── Synthetic fallback (when datasets not yet released / no internet) ─────────
def _synthetic_vqa_dataset(name: str, n_train=1000, n_test=200) -> Dict[str, List[dict]]:
    """
    Generate a small synthetic VQA dataset for testing your pipeline
    before real datasets are available. Replace with real data for paper.
    """
    logger.warning(f"Using SYNTHETIC data for {name} — replace with real dataset!")

    TEMPLATES = [
        ("What color is the object?", ["red", "blue", "green", "yellow"]),
        ("How many objects are there?", ["1", "2", "3", "4"]),
        ("What is the shape?", ["circle", "square", "triangle", "rectangle"]),
        ("Is the object large or small?", ["large", "small"]),
    ]

    def make_sample(idx):
        q_tmpl, answers = TEMPLATES[idx % len(TEMPLATES)]
        color = (np.random.randint(50,200), np.random.randint(50,200), np.random.randint(50,200))
        img = Image.new("RGB", (224, 224), color=color)
        return {
            "image":    img,
            "question": q_tmpl,
            "answer":   answers[idx % len(answers)],
            "source":   name,
        }

    train = [make_sample(i) for i in range(n_train)]
    test  = [make_sample(i) for i in range(n_train, n_train + n_test)]
    forget, retain = make_forget_retain_split(train, config.FORGET_RATIO)

    return {"train": train, "forget": forget, "retain": retain, "test": test}