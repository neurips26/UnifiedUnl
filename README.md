# Metric Unreliability in Multimodal Machine Unlearning
### A Systematic Analysis and Principled Unified Score

[![Benchmark CI](https://github.com/neurips26/UnifiedUnl/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/neurips26/UnifiedUnl/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![NeurIPS 2026 E&D](https://img.shields.io/badge/NeurIPS_2026-Evaluations_%26_Datasets-purple.svg)](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FEvaluations_and_Datasets_Track)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

---

## TL;DR

Five standard metrics for evaluating multimodal machine unlearning **contradict each other** in a systematic way.

This repository provides a fully reproducible evaluation pipeline and benchmark. Output-based metrics (FA, RA, MIA) and oracle-alignment metrics (AD, JS) measure different objectives, producing inconsistent method rankings. We identify **Knowledge Recoverability (KR)** as the aspect missed by all five metrics, demonstrate it in a pilot study (23/24 samples, 96% leakage), and introduce **UQS** — a data-driven composite score with reliability-weighted aggregation.

---

## Core Problem

<p align="center">
  <img src="outputs/figures/fig1_tau_heatmap_page-0001.jpg" width="500">
</p>

Metrics form **two opposing metric clusters**:

- **Output-behaviour cluster:** FA, RA, MIA — measure model outputs
- **Oracle-alignment cluster:** AD, JS — measure distance from retrained model M*

This produces systematic rank reversals. The same method ranked **#1 by MIA** is ranked **#4 by AD**.

Key statistic: **τ(FA, AD) = −0.26** — the two most commonly reported metrics show statistical disagreement.

---

## The Contradiction in Practice

<p align="center">
  <img src="outputs/figures/fig2_contradiction_page-0001.jpg" width="500">
</p>

**No method is consistently ranked best across all metrics.** Gradient Ascent ranks #1 by MIA but #4 by AD. Random Labels ranks #1 by four metrics but #3 by MIA. Different metric choices lead to different conclusions about method performance.

| Method | FA↓ | RA↑ | MIA↓ | AD↓ | JS↓ | UQS↑ |
|--------|-----|-----|------|------|-----|------|
| Gradient Ascent | 0.622 | 0.620 | **0.282** | 106.17 | 0.065 | **0.637** |
| Random Labels | **0.611** | **0.627** | 0.338 | **41.64** | **0.039** | 0.628 |
| FT-Retain | 0.622 | 0.580 | 0.351 | 87.43 | 0.157 | 0.590 |
| SalUn | 0.622 | 0.613 | 0.322 | 50.99 | 0.041 | 0.622 |

*Bold = best per column. UQS provides a principled resolution.*

---

## Multimodal Settings Amplify the Problem

<p align="center">
  <img src="outputs/figures/fig3_modality_page-0001.jpg" width="500">
</p>

Mean pairwise metric agreement (Kendall's τ) is lower in multimodal VQA than unimodal classification (Δτ = 0.072):

| Setting | Mean pairwise τ |
|---------|----------------|
| Multimodal VQA | **0.086** |
| Unimodal CIFAR-10 | 0.158 |
| Gap (Δ) | −0.072 |

The image–text dual pathway creates additional dimensions for inter-metric divergence that may amplify inter-metric divergence in single-pathway models.

---

## The Missing Aspect: Knowledge Recoverability (KR)

<p align="center">
  <img src="outputs/figures/flow_page-0001.jpg" width="700">
</p>

**FA = 0 does not mean the knowledge is erased.** After unlearning, a model with FA=0 can still reveal forgotten information via:

- **Rephrasing:** "Complete: [entity]'s birthplace is ___" → correct answer
- **Multi-hop:** "Who was born in the same city as [entity]?" → correct (requires knowing birthplace)
- **Negation probe:** "Is [wrong answer] [entity]'s birthplace?" → model says *no*, revealing it knows the right answer

**KR Pilot Results (2 methods × 3 seeds, MMUBench):**

| Method | Seeds | FA=0 samples | KR Leaks | Mean KR |
|--------|-------|-------------|----------|---------|
| Gradient Ascent | 3 | 6 | 5/6 (83%) | 0.44 |
| SalUn | 3 | 18 | 18/18 (100%) | 0.74 |
| **Combined** | 6 | 24 | **23/24 (96%)** | 0.65 |

Negation probes achieved **100% recovery within the pilot setup** (2 methods × 3 seeds, MMUBench). None of FA, RA, MIA, AD, or JS measures KR.

---

## Metric Reliability

<p align="center">
  <img src="outputs/figures/fig4_uqs_weights_page-0001.jpg" width="500">
</p>

Spearman ρ with retrained oracle distance (n = 36 models):

| Metric | Spearman ρ | p-value | Interpretation |
|--------|-----------|---------|----------------|
| **RA** | **+0.484** | **0.003\*\*** | Most reliable |
| MIA | +0.224 | 0.189 | Moderate, not significant |
| FA | −0.418 | 0.011\* | **Negatively reliable** |
| AD | −0.215 | 0.207 | Weak standalone |
| JS | −0.051 | 0.766 | Weakest standalone |

FA is negatively reliable because methods achieving FA≈0 often do so via **output collapse** — suppressing responses without erasing the underlying knowledge, while drifting far from M*.

---

## Unified Quality Score (UQS)

<p align="center">
  <img src="outputs/figures/fig5_uqs_stability_page-0001.jpg" width="500">
</p>

UQS aggregates all five metrics weighted by empirical reliability:

```
UQS(M̂) = w₁·(1−FA) + w₂·RA + w₃·(1−MIA) + w₄·exp(−AD/100) + w₅·(1−JS)
```

Weights `wᵢ = max(ρᵢ, ε) / Σ max(ρⱼ, ε)` are derived via 5-fold cross-validated Spearman ρ with oracle M* — not hand-tuned. The reported values (w_RA=0.656, w_MIA=0.304, others=0.014) reflect the LLaVA-MMUBench setting; weights should be re-derived for different models or datasets.

**Why not Borda count or voting?** Both assign equal weight to RA (p=0.003) and JS (p=0.766). Giving them equal vote ignores the data. Weighted average by empirical reliability is a data-driven aggregation based on observed metric performance — the minimum-assumption choice consistent with the data.

**Stability:** τ = 0.647 ± 0.262 across 100 Dirichlet-sampled random weight perturbations — rankings are robust, not brittle.

**Recalibration:** UQS is not a fixed score. Privacy-first deployments can use w_FA=0.5, w_MIA=0.5, others=0. Utility-first can use w_RA=1.0.

---

## Cross-Architecture Validation (BLIP-2)

Replicated on BLIP-2 OPT-2.7B (ViT-g/14 + Q-Former + OPT-2.7B), MMUBench, seed 42:

- GA: FA=0.667, RA=0.600, AD=1.87
- FT-Retain: FA=0.667, RA=0.630, AD=44.84
- **τ(RA, AD) = −1.0** — full rank reversal, same direction as LLaVA (−0.11)

Preliminary evidence suggests the output-surface vs oracle-alignment contradiction **generalises across architectures**.

---

## Summary of Findings

| Finding | Result | Where |
|---------|--------|-------|
| **F1** | τ(FA,AD) = −0.26: two anti-correlated metric clusters | §5.1, Table 3 |
| **F2** | Multimodal VQA has 45% lower metric agreement (Δτ = 0.072) | §5.2, Table 5 |
| **F3** | RA most reliable (ρ=0.484, p=0.003); FA negatively correlated (p=0.011) | §5.3, Table 6 |
| **KR** | 23/24 FA=0 samples leaked in pilot study (96%); negation probes 100% within pilot setup | §4.2, Table 2 |
| **UQS** | Stable composite score τ=0.647±0.262 across 100 perturbations | §6, Table 8 |

---

## Quickstart

### No GPU — verify analysis in ~5 minutes

```bash
git clone https://github.com/neurips26/UnifiedUnl
cd UnifiedUnl
pip install -r requirements.txt
python benchmark/run_benchmark.py --quick
```

Produces representative figures and tables (synthetic data, not exact paper reproduction) and an interactive leaderboard.

### From pre-computed results (real numbers, no GPU)

```bash
python benchmark/run_benchmark.py --results-only
```

Reads from `outputs/multimodal_results.json` (included in repo). Reproduces exact paper numbers.

### Full reproduction (~2 days, RTX 4090)

```bash
python download_models.py   # ~30 GB: LLaVA-7B + BLIP-2 + 3 datasets
python main.py --stage all  # train → unlearn → evaluate → analyse
```

See [REPRODUCE.md](REPRODUCE.md) for step-by-step instructions, hardware requirements, and how to verify each individual finding.

---

## Repository Structure

```
UnifiedUnl/
├── main.py                    # Pipeline entry point (--stage train/unlearn/evaluate/analyse)
├── config.py                  # All hyperparameters
├── download_models.py         # Download LLaVA, BLIP-2, and all datasets
├── run_blip2_minimal.py       # BLIP-2 cross-architecture validation
├── kr_pilot.py                # Knowledge Recoverability pilot experiment
│
├── models/                    # LLaVA, BLIP-2, model_factory router
├── data/                      # Dataset loaders (MLLMU-Bench, UnLOK-VQA, MMUBench)
├── unlearning/                # GA, RL, FT-Retain, SalUn methods
├── evaluation/                # FA, RA, MIA, AD, JS metrics + UQS
├── analysis/                  # Kendall τ, Spearman ρ, modality gap
├── benchmark/                 # One-command benchmark runner + leaderboard
├── scripts/                   # CIFAR-10 unimodal baseline
│
├── outputs/
│   ├── multimodal_results.json      # All 36 evaluation results
│   ├── unimodal_results.json        # CIFAR-10 baseline results
│   ├── uqs_weights.json             # Derived UQS weights
│   ├── blip2_minimal_summary.json   # BLIP-2 validation
│   ├── kr_pilot_results.json        # KR pilot (23/24 leaked)
│   ├── analysis_results.json        # Full τ matrix, reliability scores
│   ├── figures/                     # All 5 paper figures (PDF)
│   └── tables/                      # All 6 LaTeX tables
│
├── REPRODUCE.md               # Detailed reproduction guide
├── DATASHEET.md               # Dataset documentation (Gebru et al.)
├── CITATION.cff               # Machine-readable citation
└── croissant_metadata.json    # NeurIPS Croissant metadata with RAI fields
```

---

## Pre-computed Results

All evaluation result files are included in `outputs/` directly in this repository.
Three representative checkpoints (~15 GB) and a HuggingFace dataset page will be linked here upon deanonymization after review.

---

## Citation

```bibtex
@inproceedings{anonymous2026metric,
  title     = {Metric Unreliability in Multimodal Machine Unlearning:
               A Systematic Analysis and Principled Unified Score},
  booktitle = {NeurIPS Evaluations and Datasets Track},
  year      = {2026}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
