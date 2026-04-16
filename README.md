# Metric Unreliability in Multimodal Machine Unlearning

### A Systematic Analysis and Principled Unified Score

[![Benchmark CI](https://github.com/neurips26/UnifiedUnl/actions/workflows/ci.yml/badge.svg?branch=main\&event=push)](https://github.com/neurips26/UnifiedUnl/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/neurips26/UnifiedUnl/blob/main/LICENSE)
[![NeurIPS 2026 D\&B](https://img.shields.io/badge/NeurIPS_2026-Datasets_%26_Benchmarks-purple.svg)](https://openreview.net/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

---

## TL;DR

Standard metrics for multimodal machine unlearning are mutually contradictory. We show that output-based metrics (FA, RA, MIA) and representation-based metrics (AD, JS) capture fundamentally different objectives, leading to inconsistent rankings. We introduce a Unified Quality Score (UQS) that aggregates metrics using empirically derived reliability weights.

---

## One-Command Quickstart (Reviewers — No GPU needed)

```bash
git clone https://github.com/neurips26/UnifiedUnl
cd UnifiedUnl
pip install -r requirements.txt
python benchmark/run_benchmark.py --quick
```

Runs a lightweight version of the pipeline with synthetic data (~5 minutes). Produces:

* `outputs/leaderboard.html` — interactive leaderboard
* `outputs/figures/` — representative figures (quick mode subset)
* `outputs/tables/` — generated LaTeX tables

For full reproduction (~2 days on a single RTX 4090 GPU):

```bash
python main.py --stage all
```

---

## Key Findings

| Finding | Result                                                                         |
| ------- | ------------------------------------------------------------------------------ |
| **F1**  | Metrics form two conflicting clusters: {FA, RA, MIA} vs {AD, JS}               |
| **F2**  | Strong contradiction between clusters (e.g., τ(FA, AD) ≈ −0.26)                |
| **F3**  | Multimodal settings amplify disagreement (mean τ: 0.086 vs 0.158 in unimodal)  |
| **F4**  | RA is the most reliable metric (ρ ≈ 0.484), FA is negatively correlated        |
| **UQS** | Reliability-weighted score with stable rankings (τ ≈ 0.65 under perturbations) |

---

## Repository Structure

```
UnifiedUnl/
├── benchmark/
│   ├── run_benchmark.py
│   ├── generate_tables.py
│   └── leaderboard.py
├── config.py
├── main.py
├── test_pipeline.py
├── requirements.txt
├── data/
├── models/
├── unlearning/
├── evaluation/
├── analysis/
├── scripts/
├── DATASHEET.md
├── croissant_metadata.json
└── LICENSE
```

---

## Experimental Setup

**Model:** LLaVA-1.5-7B with LoRA (r=8, α=16)

**Datasets:**

| Dataset     | HuggingFace ID           |
| ----------- | ------------------------ |
| MLLMU-Bench | franciscoliu/MLLMU-Bench |
| UnLOK-VQA   | vpatil24/unlok-vqa       |
| MMUBench    | linhx/MMUBench           |
| CIFAR-10    | torchvision              |

**Unlearning Methods:**
Gradient Ascent · Random Labels · FT-Retain · SalUn

**Metrics:**
FA ↓ · RA ↑ · MIA ↓ · AD ↓ · JS ↓

---

## Reproduce Step by Step

```bash
# Install dependencies
pip install -r requirements.txt

# Smoke test (CPU)
python test_pipeline.py

# Full pipeline
python main.py --stage train
python main.py --stage unlearn
python main.py --stage evaluate
python main.py --stage unimodal
python main.py --stage analyse
```

---

## NeurIPS D&B Track Compliance

| Requirement              | Status |
| ------------------------ | ------ |
| Code publicly accessible | Yes    |
| Executable pipeline      | Yes    |
| Datasets available       | Yes    |
| Metadata (Croissant)     | Yes    |
| Datasheet                | Yes    |
| License                  | Yes    |
| CI passing               | Yes    |

---

## Citation

```bibtex
@inproceedings{anonymous2026metric,
  title={Metric Unreliability in Multimodal Machine Unlearning: A Systematic Analysis and Principled Unified Score},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2026}
}
```
