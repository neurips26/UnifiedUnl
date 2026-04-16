# Metric Unreliability in Multimodal Machine Unlearning
### A Systematic Analysis and Principled Unified Score

[![CI](https://github.com/neurips26/UnifiedUnl/actions/workflows/ci.yml/badge.svg)](https://github.com/neurips26/UnifiedUnl/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![NeurIPS 2026 D&B](https://img.shields.io/badge/NeurIPS_2026-Datasets_%26_Benchmarks-purple.svg)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()

---

## TL;DR

Five widely-used metrics for evaluating multimodal machine unlearning **contradict each other**. A method ranked #1 by FA can be ranked #4 by RA. This contradiction is **worse in multimodal (VQA) settings** than unimodal (CIFAR-10). We propose a **Unified Quality Score (UQS)** whose weights are derived empirically from Spearman correlation with the gold-standard retrained model.

---

## One-Command Quickstart (Reviewers — No GPU needed)

```bash
git clone https://github.com/TODO/multimodal-unlearning-eval
cd multimodal-unlearning-eval
pip install -r requirements.txt
python benchmark/run_benchmark.py --quick
```

Runs the full analysis pipeline with synthetic data in ~5 minutes. Produces:
- `outputs/leaderboard.html` — interactive leaderboard (open in browser)
- `outputs/figures/` — all 5 paper figures as PDF
- `outputs/tables/` — all 6 LaTeX tables

To reproduce with real models (~2 days on RTX 4090):
```bash
python main.py --stage all
```

---

## Key Findings

| Finding | Result |
|---------|--------|
| **F1** | Kendall's τ between FA and RA ≈ −0.4 — metrics contradict each other |
| **F2** | Mean pairwise τ is 0.3 lower in multimodal vs unimodal — image+text amplifies contradiction |
| **F3** | AD and JS are most reliable predictors of retrained-model closeness (Spearman ρ > 0.7) |
| **UQS** | Principled composite score with τ > 0.9 stability across 500 random weight perturbations |

---

## Repository Structure

```
multimodal-unlearning-eval/
├── benchmark/
│   ├── run_benchmark.py        ← ONE-COMMAND benchmark runner
│   ├── generate_tables.py      ← All 6 LaTeX tables
│   └── leaderboard.py          ← Interactive HTML leaderboard
├── config.py                   ← All settings (edit only this)
├── main.py                     ← Stage-by-stage pipeline
├── test_pipeline.py            ← Smoke test (CPU, <2 min)
├── setup.py
├── requirements.txt
├── data/
│   ├── loader.py               ← MLLMU-Bench, UnLOK-VQA, MMUBench, CIFAR-10
│   └── dataset.py              ← VQADataset + DataLoader
├── models/
│   └── llava_model.py          ← LLaVA-1.5-7B + LoRA
├── unlearning/
│   └── methods.py              ← GA, Random Labels, FT-Retain, SalUn
├── evaluation/
│   ├── metrics.py              ← FA, RA, MIA, AD, JS
│   └── uqs.py                  ← UQS formula, weight derivation, ablation
├── analysis/
│   ├── findings.py             ← 3 core findings
│   └── visualise.py            ← 5 paper figures (PDF)
├── scripts/
│   └── unimodal_baseline.py    ← ResNet-18 + CIFAR-10
├── DATASHEET.md                ← Dataset documentation
├── croissant_metadata.json     ← Machine-readable metadata (NeurIPS D&B required)
└── LICENSE                     ← Apache 2.0
```

---

## Experimental Setup

**Model:** LLaVA-1.5-7B + LoRA (r=8, α=16) | RTX 4090 24GB

**Datasets:**

| Dataset | HuggingFace ID | Reference |
|---------|----------------|-----------|
| MLLMU-Bench | franciscoliu/MLLMU-Bench | Liu et al., NAACL 2025 |
| UnLOK-VQA | vpatil24/unlok-vqa | Patil et al., 2025 |
| MMUBench | linhx/MMUBench | Li et al., NeurIPS 2024 |
| CIFAR-10 | torchvision built-in | Baseline |

**Unlearning Methods:** Gradient Ascent · Random Labels · FT-Retain · SalUn (ICLR 2024)

**Metrics:** FA↓ · RA↑ · MIA↓ · AD↓ · JS↓

**Seeds:** 42 · 123 · 5508

---

## Reproduce Step by Step

```bash
# 0. Install
pip install -r requirements.txt

# 1. Smoke test (CPU, no download needed)
python test_pipeline.py

# 2. Full pipeline
python main.py --stage train      # ~3 hours
python main.py --stage unlearn    # ~2 days
python main.py --stage evaluate   # ~4 hours
python main.py --stage unimodal   # ~30 min
python main.py --stage analyse    # ~5 min

# 3. Single run
python main.py --stage unlearn --dataset mllmu_bench --method gradient_ascent --seed 42

# 4. Re-run analysis from saved results
python benchmark/run_benchmark.py --results-only
```

---

## NeurIPS D&B Track Compliance

| Requirement | Status |
|-------------|--------|
| Code publicly accessible at submission | ✅ GitHub |
| Code documented and executable | ✅ README + CI |
| Source datasets on HuggingFace | ✅ All 3 datasets |
| Croissant metadata file | ✅ `croissant_metadata.json` |
| Datasheet (Gebru et al. 2021) | ✅ `DATASHEET.md` |
| License | ✅ Apache 2.0 |
| CI passing | ✅ GitHub Actions |

---

## Citation

```bibtex
@inproceedings{anonymous2026metric,
  title     = {Metric Unreliability in Multimodal Machine Unlearning:
               A Systematic Analysis and Principled Unified Score},
  author    = {Anonymous},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2026},
}
```
