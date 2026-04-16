"""
test_pipeline.py — Smoke test. Runs full pipeline with synthetic data + tiny models.
No GPU needed. Verifies every component works before running on real data.

Run: python test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

print("=" * 60)
print("SMOKE TEST — Full Pipeline Verification")
print("=" * 60)


# ─── Patch config for fast testing ───────────────────────────────────────────
import config
config.FINETUNE_EPOCHS  = 1
config.UNLEARN_EPOCHS   = 1
config.FINETUNE_BATCH   = 2
config.UNLEARN_BATCH    = 2
config.CIFAR_EPOCHS     = 2
config.CIFAR_BATCH      = 64
config.DATASETS         = ["mllmu_bench"]   # only 1 dataset
config.SEEDS            = [42]              # only 1 seed
config.DEVICE           = "cpu"


# ─── 1. Synthetic data ────────────────────────────────────────────────────────
print("\n[1] Loading synthetic VQA data...")
from data.loader import _synthetic_vqa_dataset, make_forget_retain_split
data = _synthetic_vqa_dataset("test", n_train=50, n_test=20)
print(f"  forget={len(data['forget'])}, retain={len(data['retain'])}, test={len(data['test'])}")
assert len(data["forget"]) > 0
assert len(data["retain"]) > 0
print("  ✓ Data loading OK")


# ─── 2. Tiny mock model (avoids downloading LLaVA) ───────────────────────────
print("\n[2] Building tiny mock model...")

class TinyVQAModel(nn.Module):
    """Tiny model that mimics LLaVA interface for testing."""
    def __init__(self, vocab=100, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.vision = nn.Linear(3 * 224 * 224, hidden)
        self.fusion = nn.Linear(hidden * 2, hidden)
        self.lm_head = nn.Linear(hidden, vocab)
        self.vocab = vocab

    def forward(self, input_ids=None, attention_mask=None,
                pixel_values=None, labels=None, **kwargs):
        B = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 10

        txt = self.embed(input_ids).mean(1) if input_ids is not None \
              else torch.zeros(B, 64)

        if pixel_values is not None:
            flat = pixel_values.reshape(B, -1)
            if flat.shape[1] != 3*224*224:
                vis = torch.zeros(B, 64)
            else:
                vis = self.vision(flat.float())
        else:
            vis = torch.zeros(B, 64)

        fused  = self.fusion(torch.cat([txt, vis], dim=1))           # (B, hidden)
        logits = self.lm_head(fused).unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq, vocab)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, self.vocab),
                labels.reshape(-1).clamp(0, self.vocab - 1)
            )

        class Out:
            pass
        out        = Out()
        out.loss   = loss if loss is not None else torch.tensor(0.0)
        out.logits = logits
        return out

    def generate(self, input_ids=None, attention_mask=None,
                 pixel_values=None, max_new_tokens=50, **kwargs):
        """Stub generate() — appends random token IDs, mimics LLaVA output shape."""
        B    = input_ids.shape[0] if input_ids is not None else 1
        base = input_ids if input_ids is not None else torch.zeros(B, 1, dtype=torch.long)
        new  = torch.randint(0, self.vocab, (B, max_new_tokens))
        return torch.cat([base, new], dim=1)

model = TinyVQAModel()
print("  ✓ Tiny model built")


# ─── 3. Mock processor ────────────────────────────────────────────────────────
print("\n[3] Building mock processor...")

class MockProcessor:
    """Converts PIL + text to tensors without real tokeniser."""
    def __call__(self, text=None, images=None, return_tensors="pt",
                 padding=None, truncation=None, max_length=None):
        import torchvision.transforms as T
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        if isinstance(images, list):
            imgs = torch.stack([transform(img.convert("RGB")) for img in images])
        elif images is not None:
            imgs = transform(images.convert("RGB")).unsqueeze(0)
        else:
            imgs = torch.zeros(1, 3, 224, 224)

        B = imgs.shape[0]
        L = min(max_length or 32, 32)
        input_ids = torch.randint(0, 100, (B, L))
        attn_mask = torch.ones(B, L, dtype=torch.long)

        class Enc(dict):
            def to(self, device): return self
        return Enc({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "pixel_values": imgs,
        })

    def decode(self, ids, skip_special_tokens=True):
        answers = ["yes", "no", "red", "blue", "1", "2"]
        return answers[ids[0].item() % len(answers)] if len(ids) > 0 else "unknown"

processor = MockProcessor()
print("  ✓ Mock processor built")


# ─── 4. DataLoader ────────────────────────────────────────────────────────────
print("\n[4] Testing DataLoader...")
from data.dataset import make_dataloader
loader = make_dataloader(data["forget"], processor, batch_size=4, shuffle=False)
batch  = next(iter(loader))
assert "input_ids" in batch
assert "pixel_values" in batch
print(f"  Batch keys: {list(batch.keys())}")
print("  ✓ DataLoader OK")


# ─── 5. Training ──────────────────────────────────────────────────────────────
print("\n[5] Testing training loop...")
from models.llava_model import train_one_epoch
import copy
from torch.optim import AdamW

m_copy = copy.deepcopy(model)
opt    = AdamW([p for p in m_copy.parameters() if p.requires_grad], lr=1e-3)
loss   = train_one_epoch(m_copy, loader, opt, device="cpu")
print(f"  Training loss: {loss:.4f}")
assert not np.isnan(loss)
print("  ✓ Training OK")


# ─── 6. Unlearning methods ────────────────────────────────────────────────────
print("\n[6] Testing unlearning methods...")
from unlearning.methods import run_unlearning

for method in config.METHODS:
    print(f"  Testing {method}...")
    unlearned = run_unlearning(
        method_name = method,
        model       = copy.deepcopy(model),
        forget_set  = data["forget"],
        retain_set  = data["retain"],
        processor   = processor,
        seed        = 42,
    )
    assert unlearned is not None
    print(f"    ✓ {method} OK")


# ─── 7. Metrics ───────────────────────────────────────────────────────────────
print("\n[7] Testing evaluation metrics (on tiny subsets)...")
from evaluation.metrics import compute_fa, compute_ra, compute_mia

subset_forget = data["forget"][:10]
subset_retain = data["retain"][:10]

fa  = compute_fa(model, processor, subset_forget)
ra  = compute_ra(model, processor, subset_retain)
mia = compute_mia(model, processor, subset_forget, subset_retain)

print(f"  FA={fa:.4f}, RA={ra:.4f}, MIA={mia:.4f}")
assert 0 <= fa  <= 1
assert 0 <= ra  <= 1
assert 0 <= mia <= 1
print("  ✓ FA, RA, MIA OK")

# AD and JS need two models
unlearned_m = copy.deepcopy(model)
from evaluation.metrics import compute_ad, compute_js

# Patch extraction to avoid real LLaVA layers
def mock_extract_activations(m, p, samples, layer_idx=-2):
    return torch.randn(len(samples), 64)

def mock_extract_probs(m, p, samples):
    return torch.softmax(torch.randn(len(samples), 100), dim=-1)

import evaluation.metrics as metrics_module
metrics_module.extract_activations = mock_extract_activations
metrics_module.extract_output_probs = mock_extract_probs

ad = compute_ad(unlearned_m, model, processor, subset_forget, n_samples=5)
js = compute_js(unlearned_m, model, processor, subset_forget, n_samples=5)
print(f"  AD={ad:.4f}, JS={js:.4f}")
assert ad >= 0
assert 0 <= js <= 1
print("  ✓ AD, JS OK")


# ─── 8. UQS ───────────────────────────────────────────────────────────────────
print("\n[8] Testing UQS...")
from evaluation.uqs import compute_uqs, build_ranking_table, uqs_ablation

fake_results = {
    "gradient_ascent": {"FA": 0.1, "RA": 0.4, "MIA": 0.6, "AD": 2.0, "JS": 0.3},
    "random_labels":   {"FA": 0.3, "RA": 0.7, "MIA": 0.4, "AD": 1.5, "JS": 0.2},
    "finetune_retain": {"FA": 0.8, "RA": 0.9, "MIA": 0.2, "AD": 0.5, "JS": 0.1},
    "salun":           {"FA": 0.2, "RA": 0.6, "MIA": 0.5, "AD": 1.0, "JS": 0.25},
}

for method, scores in fake_results.items():
    uqs = compute_uqs(scores)
    print(f"  UQS({method}) = {uqs:.4f}")
    assert 0 <= uqs <= 1

table    = build_ranking_table(fake_results)
ablation = uqs_ablation(fake_results, n_trials=20)
print(f"  Ablation τ = {ablation['mean_tau']:.4f} ± {ablation['std_tau']:.4f}")
print("  ✓ UQS OK")


# ─── 9. Analysis ──────────────────────────────────────────────────────────────
print("\n[9] Testing findings analysis...")
from analysis.findings import (
    compute_kendall_tau_matrix,
    build_contradiction_table,
    compare_modality_agreement,
)

# Build fake all_results format
flat_results = []
for method, scores in fake_results.items():
    for dataset in ["mllmu_bench", "unlok_vqa"]:
        for seed in [42, 123]:
            flat_results.append({
                "method": method, "dataset": dataset, "seed": seed,
                **scores
            })

tau_data = compute_kendall_tau_matrix(flat_results)
table2   = build_contradiction_table(flat_results)
assert len(table2) == 4
print(f"  Contradiction table: {len(table2)} methods")

finding2 = compare_modality_agreement(flat_results, flat_results[:8])
print(f"  Modality delta τ = {finding2['delta']:.4f}")
print("  ✓ Findings OK")


# ─── 10. Visualisation ────────────────────────────────────────────────────────
print("\n[10] Testing figure generation...")
from analysis.visualise import (
    plot_tau_heatmap,
    plot_contradiction_table,
    plot_modality_comparison,
)
import tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    config.FIGURES_DIR = tmpdir

    p1 = plot_tau_heatmap(tau_data,   save_path=os.path.join(tmpdir, "f1.pdf"))
    p2 = plot_contradiction_table(table2, save_path=os.path.join(tmpdir, "f2.pdf"))
    p3 = plot_modality_comparison(finding2, save_path=os.path.join(tmpdir, "f3.pdf"))

    assert os.path.exists(p1)
    assert os.path.exists(p2)
    assert os.path.exists(p3)
    print(f"  Figures saved: {[os.path.basename(p) for p in [p1,p2,p3]]}")
print("  ✓ Figures OK")


# ─── CIFAR-10 (unimodal) ──────────────────────────────────────────────────────
print("\n[11] Testing CIFAR-10 unimodal baseline (quick)...")
try:
    import torchvision
    print("  torchvision available ✓")
    # Full CIFAR run takes a few minutes; just verify imports
    from scripts.unimodal_baseline import get_resnet18, eval_accuracy
    m = get_resnet18()
    print(f"  ResNet-18 parameters: {sum(p.numel() for p in m.parameters()):,}")
    print("  ✓ Unimodal baseline imports OK")
except ImportError:
    print("  torchvision not installed — run: pip install torchvision")


# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED ✓")
print("=" * 60)
print("""
Next steps:
  1. Confirm GPU: python -c "import torch; print(torch.cuda.is_available())"
  2. Download LLaVA: python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')"
  3. Train vanilla models: python main.py --stage train
  4. Run unlearning:       python main.py --stage unlearn
  5. Evaluate metrics:     python main.py --stage evaluate
  6. Run unimodal:         python main.py --stage unimodal
  7. Generate figures:     python main.py --stage analyse
  
  Or run everything at once: python main.py --stage all
  
  Single run example:
    python main.py --stage unlearn --dataset mllmu_bench --method gradient_ascent --seed 42
""")