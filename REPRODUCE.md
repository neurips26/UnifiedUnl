# Reproduction Guide

This document explains how to reproduce every result in the paper
**"Metric Unreliability in Multimodal Machine Unlearning"** (NeurIPS 2026 E&D Track).

---

## Quick-start for reviewers (no GPU needed — ~5 minutes)

```bash
git clone https://github.com/neurips26/UnifiedUnl
cd UnifiedUnl
pip install -r requirements.txt
python benchmark/run_benchmark.py --quick
```

This runs the complete analysis pipeline on **pre-loaded synthetic data**,
reproduces all 5 figures and 6 tables, and opens the interactive leaderboard —
all on CPU in about 5 minutes.

Outputs produced:
```
outputs/leaderboard.html          ← open in browser
outputs/figures/fig1_tau_heatmap.pdf
outputs/figures/fig2_contradiction.pdf
outputs/figures/fig3_modality.pdf
outputs/figures/fig4_uqs_weights.pdf
outputs/figures/fig5_uqs_stability.pdf
outputs/tables/table1_main_results.tex
outputs/tables/all_tables.tex
```

---

## Reproduce from pre-computed results (no GPU, real numbers — ~2 minutes)

All 36 evaluation result JSONs are included in the repo under `outputs/`.
To regenerate figures and tables from these real results without re-running
the full experiment:

```bash
python benchmark/run_benchmark.py --results-only
```

This reads from `outputs/multimodal_results.json` and `outputs/unimodal_results.json`
and reproduces every figure, table, and statistical test in the paper.
The numbers will exactly match those in the paper.

---

## Pre-computed results on HuggingFace

All result files are also hosted at:

```
https://huggingface.co/datasets/TODO/multimodal-unlearning-eval
```

Files available:
| File | Description |
|------|-------------|
| `multimodal_results.json` | 36 runs: 4 methods × 3 datasets × 3 seeds, 5 metrics each |
| `unimodal_results.json` | 12 runs: CIFAR-10 unimodal baseline |
| `uqs_weights.json` | Derived UQS weights (Spearman ρ with oracle) |
| `ranking_table.json` | Per-metric and UQS rankings |
| `blip2_minimal_summary.json` | BLIP-2 cross-architecture validation |
| `kr_pilot_results.json` | Knowledge Recoverability pilot (23/24 leaked) |
| `analysis_results.json` | Full statistical analysis (τ matrix, modality gap) |

---

## Representative checkpoints

Due to storage constraints (~583 GB total), model checkpoints are not included.

All evaluation results are provided in `outputs/` and can be reproduced using the provided pipeline.


| Checkpoint | Description |
|------------|-------------|
| `llava_retrained_mmubench_seed42.pt` | Oracle M* — retrained on retain set only |
| `llava_unlearned_mmubench_gradient_ascent_seed42.pt` | Best-forgetting model (GA, FA=0.867) |
| `llava_unlearned_mmubench_finetune_retain_seed42.pt` | Best-utility model (FT-Retain, RA=0.84) |

To evaluate these checkpoints:
```bash
python main.py --stage evaluate --dataset mmubench --method gradient_ascent --seed 42
python main.py --stage evaluate --dataset mmubench --method finetune_retain --seed 42
python benchmark/run_benchmark.py --results-only
```

---

## Full reproduction (GPU required — ~2 days)

To reproduce all 36 models from scratch:

### Step 1: Download all models and datasets (~30 GB)
```bash
python download_models.py
```

This downloads:
- LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) — ~14 GB
- BLIP-2 OPT-2.7B (`Salesforce/blip2-opt-2.7b`) — ~15 GB
- MLLMU-Bench (`franciscoliu/MLLMU-Bench`)
- UnLOK-VQA (`vpatil24/unlok-vqa`)
- MMUBench (`lmms-lab/MMBench_EN`)

To skip BLIP-2 (saves 15 GB, skips cross-architecture validation):
```bash
python download_models.py --no-blip2
```

### Step 2: Run the full pipeline
```bash
python main.py --stage all
```

Stages run in order:
1. `train` — fine-tune vanilla LLaVA on each dataset (~6 hours)
2. `unlearn` — run 4 methods × 3 datasets × 3 seeds = 36 models (~36 hours)
3. `evaluate` — compute FA/RA/MIA/AD/JS for all 36 models (~8 hours)
4. `analyse` — statistical analysis, figures, tables (~5 minutes)

To run a single stage:
```bash
python main.py --stage train
python main.py --stage unlearn
python main.py --stage evaluate
python main.py --stage analyse
```

To run a single method/dataset/seed:
```bash
python main.py --stage unlearn --dataset mmubench --method gradient_ascent --seed 42
```

### Step 3: BLIP-2 cross-architecture validation (~12 hours)
```bash
python run_blip2_minimal.py
```

Produces `outputs/blip2_minimal_summary.json`.

### Step 4: Knowledge Recoverability pilot (~30 minutes)
```bash
python kr_pilot.py
```

Requires Anthropic API key for probe generation (falls back to templates if unavailable):
```bash
export ANTHROPIC_API_KEY=your_key_here
python kr_pilot.py
```

Produces:
- `outputs/kr_pilot_results.json`
- `outputs/kr_pilot_table.tex`
- `outputs/kr_pilot_summary.txt`

---

## Hardware requirements

| Task | Minimum | Used in paper |
|------|---------|---------------|
| Quick mode (synthetic) | Any CPU | — |
| Results-only mode | Any CPU | — |
| Full pipeline | GPU ≥ 16 GB VRAM | RTX 4090 (24 GB) |
| BLIP-2 validation | GPU ≥ 20 GB VRAM | RTX 4090 (24 GB) |

Estimated GPU-hours for full reproduction: ~200 GPU-hours on RTX 4090.

---

## Verifying a single finding

**Finding 1 (metric contradiction, τ_FA,AD = −0.26):**
```bash
python benchmark/run_benchmark.py --results-only
# Check outputs/analysis_results.json → "tau_matrix" → "FA_AD"
```

**Finding 2 (multimodal gap, Δτ = 0.072):**
```bash
# Check outputs/analysis_results.json → "modality_gap"
```

**Finding 3 (RA reliability, ρ = 0.484, p = 0.003):**
```bash
# Check outputs/uqs_weights.json → "RA" → "spearman_rho" and "p_value"
```

**KR pilot (23/24 FA=0 samples leaked):**
```bash
python kr_pilot.py
# Check outputs/kr_pilot_results.json
```

**BLIP-2 replication (τ_RA,AD = −1.0):**
```bash
python run_blip2_minimal.py
# Check outputs/blip2_minimal_summary.json → "blip2_tau" → "RA_AD"
```

---

## Environment

Tested on:
- Python 3.10, 3.11, 3.12
- PyTorch 2.2.0 (CUDA 12.1)
- transformers 4.40.0
- Ubuntu 22.04, Windows 11

CI runs on Python 3.10 (CPU only) via GitHub Actions — see `.github/workflows/ci.yml`.

---

## Questions

If you encounter issues reproducing results, please open a GitHub issue at
`https://github.com/neurips26/UnifiedUnl/issues`.
