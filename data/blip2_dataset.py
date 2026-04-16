"""
data/blip2_dataset.py — PyTorch Dataset wrapper for BLIP-2.

BLIP-2 processes images and text separately (no interleaved image tokens).
The processor returns pixel_values directly; no <image> token in text.

Differences from VQADataset (LLaVA):
  - Prompt format: "Question: {q} Answer: {a}" instead of "USER: <image>\n..."
  - No image token in text input
  - pixel_values shape: [B, 3, 224, 224] (ViT-g/14 input resolution)
  - Labels: only text tokens, not image tokens
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional

from utils import get_logger
import config

logger = get_logger(__name__)

# BLIP-2 ViT-g/14 image resolution
BLIP2_IMG_SIZE = 224


class BLIP2VQADataset(Dataset):
    """
    Dataset wrapper for BLIP-2 VQA fine-tuning.
    Each sample: {"image": PIL.Image, "question": str, "answer": str}
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        processor=None,
        corrupt_labels: bool = False,
        corrupt_answers: List[str] = None,
    ):
        self.samples         = samples
        self.processor       = processor
        self.corrupt_labels  = corrupt_labels
        self.corrupt_answers = corrupt_answers or []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        image    = sample["image"].convert("RGB")
        question = sample["question"]
        answer   = sample["answer"]

        # Corrupt answer for Random Labels unlearning method
        if self.corrupt_labels and self.corrupt_answers:
            wrong  = [a for a in self.corrupt_answers if a != answer]
            answer = np.random.choice(wrong) if wrong else "unknown"

        # BLIP-2 prompt format (no <image> token — processor handles image)
        prompt = f"Question: {question} Answer: {answer}"

        if self.processor is not None:
            encoding = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=128,        # BLIP-2 text is shorter than LLaVA
                truncation=True,
            )
            input_ids      = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            pixel_values   = encoding["pixel_values"].squeeze(0)

            # Labels: mask out the question part, only learn the answer
            # Find where "Answer:" ends in the token sequence
            labels = input_ids.clone()
            # Mask everything with -100 (ignore) for the question prefix
            # Simple heuristic: mask first 60% of tokens as question
            q_tokens = self.processor.tokenizer(
                f"Question: {question} Answer:",
                return_tensors="pt", truncation=True,
            )["input_ids"].squeeze(0)
            n_q = min(len(q_tokens), len(labels))
            labels[:n_q] = -100

            return {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "pixel_values":   pixel_values,
                "labels":         labels,
            }
        else:
            return {"image": image, "question": question, "answer": answer}


def blip2_collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of BLIP-2 dataset items into a padded batch."""
    if "pixel_values" not in batch[0]:
        # Raw samples (no processor) — return as list
        return batch

    input_ids      = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    pixel_values   = torch.stack([b["pixel_values"] for b in batch])
    labels         = torch.stack([b["labels"] for b in batch])

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "pixel_values":   pixel_values,
        "labels":         labels,
    }


def make_dataloader_blip2(
    samples: List[Dict],
    processor,
    batch_size: int = config.FINETUNE_BATCH,
    shuffle: bool = True,
    corrupt_labels: bool = False,
    corrupt_answers: List[str] = None,
) -> DataLoader:
    """Create a DataLoader for BLIP-2 training/evaluation."""
    dataset = BLIP2VQADataset(
        samples,
        processor=processor,
        corrupt_labels=corrupt_labels,
        corrupt_answers=corrupt_answers,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=blip2_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
