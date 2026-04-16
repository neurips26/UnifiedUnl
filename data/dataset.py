"""
data/dataset.py — PyTorch Dataset wrapper for VQA samples.
Handles image preprocessing + tokenization for LLaVA-1.5.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
from utils import get_logger
import config

logger = get_logger(__name__)


class VQADataset(Dataset):
    """
    Wraps a list of VQA samples for LLaVA-1.5.
    Each sample: {"image": PIL.Image, "question": str, "answer": str}
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        processor=None,
        max_length: int = config.MAX_INPUT_LEN,
        corrupt_labels: bool = False,     # for Random Labels method
        corrupt_answers: List[str] = None,
    ):
        self.samples         = samples
        self.processor       = processor
        self.max_length      = max_length
        self.corrupt_labels  = corrupt_labels
        self.corrupt_answers = corrupt_answers or []

        if corrupt_labels and not corrupt_answers:
            logger.warning("corrupt_labels=True but no corrupt_answers provided.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        image    = sample["image"]
        question = sample["question"]
        answer   = sample["answer"]

        # Corrupt answer for Random Labels unlearning method
        if self.corrupt_labels and self.corrupt_answers:
            wrong = [a for a in self.corrupt_answers if a != answer]
            answer = np.random.choice(wrong) if wrong else "unknown"

        # Format prompt in LLaVA conversation style
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"

        if self.processor is not None:
            # IMPORTANT: Do NOT pass padding or truncation to LLaVA processor.
            # The image occupies 576 tokens; padding="max_length" causes a token
            # count mismatch in _check_special_mm_tokens. Padding is done in
            # the collate_fn (llava_collate_fn) after encoding.
            encoding = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            return {
                "input_ids":      input_ids,
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "pixel_values":   encoding["pixel_values"].squeeze(0),
                "labels":         input_ids.clone(),
                "answer":         answer,
                "question":       question,
            }
        else:
            # Return raw for inference without processor
            return {
                "image":    image,
                "question": question,
                "answer":   answer,
                "prompt":   prompt,
            }


class VQAInferenceDataset(Dataset):
    """Lightweight dataset for inference only (no labels needed)."""

    def __init__(self, samples: List[Dict], processor=None):
        self.samples   = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        question = sample["question"]
        image    = sample["image"]
        # Prompt without answer for generation
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        if self.processor:
            encoding = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
            return {
                "input_ids":      encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "pixel_values":   encoding["pixel_values"].squeeze(0),
                "answer":         sample["answer"],
                "question":       question,
            }
        return sample



def llava_collate_fn(batch):
    """
    Custom collate for LLaVA batches.
    Pads input_ids, attention_mask, labels to the longest sequence in the batch.
    pixel_values are already the same size (vision encoder output).
    Handles both training dicts (with labels) and inference dicts (without).
    """
    import torch

    # Stack pixel_values (always same shape: [3, H, W])
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    # Pad sequences to max length in this batch
    input_ids_list      = [b["input_ids"]      for b in batch]
    attention_mask_list = [b["attention_mask"]  for b in batch]

    # Hard cap at MAX_INPUT_LEN — attention is O(n^2), long seqs = OOM + slow.
    # IMPORTANT: LLaVA image tokens = 576. Never truncate below 640
    # or the image placeholder count won't match image features -> crash.
    import config as _cfg
    LLAVA_MIN_LEN = 640   # 576 image tokens + 64 text minimum
    cap = max(_cfg.MAX_INPUT_LEN, LLAVA_MIN_LEN)
    max_len = min(
        max(ids.shape[0] for ids in input_ids_list),
        cap
    )

    padded_input_ids  = []
    padded_attn_mask  = []
    for ids, mask in zip(input_ids_list, attention_mask_list):
        # Truncate if longer than max_len, then pad if shorter
        ids  = ids[:max_len]
        mask = mask[:max_len]
        pad_len = max_len - ids.shape[0]
        padded_input_ids.append(
            torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
        )
        padded_attn_mask.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
        )

    result = {
        "input_ids":      torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attn_mask),
        "pixel_values":   pixel_values,
    }

    # Labels (training only)
    if "labels" in batch[0]:
        padded_labels = []
        for b in batch:
            lbl     = b["labels"][:max_len]
            pad_len = max_len - lbl.shape[0]
            # -100 tells CrossEntropyLoss to ignore padding positions
            padded_labels.append(
                torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
            )
        result["labels"] = torch.stack(padded_labels)

    # Pass through string fields
    if "answer" in batch[0]:
        result["answer"]   = [b["answer"]   for b in batch]
        result["question"] = [b["question"] for b in batch]

    return result


def make_dataloader(
    samples: List[Dict],
    processor=None,
    batch_size: int = 4,
    shuffle: bool = True,
    corrupt_labels: bool = False,
    corrupt_answers: List[str] = None,
    for_inference: bool = False,
) -> DataLoader:
    """Factory function — returns a ready DataLoader."""
    if for_inference:
        ds = VQAInferenceDataset(samples, processor)
    else:
        ds = VQADataset(
            samples,
            processor=processor,
            corrupt_labels=corrupt_labels,
            corrupt_answers=corrupt_answers,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,       # keep 0 to avoid multiprocessing issues with PIL
        pin_memory=torch.cuda.is_available(),
        collate_fn=llava_collate_fn,
    )