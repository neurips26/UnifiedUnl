"""
benchmark/run_benchmark.py
==========================
ONE-COMMAND BENCHMARK RUNNER for NeurIPS D&B reviewers.

This script runs the full benchmark and produces:
  - All metric scores (outputs/multimodal_results.json)
  - UQS weights and rankings (outputs/ranking_table.json)
  - All 5 paper figures (outputs/figures/)
  - LaTeX tables (outputs/tables/)
  - Leaderboard HTML (outputs/leaderboard.html)

Usage:
    # Full benchmark (requires GPU, ~2 days)
    python benchmark/run_benchmark.py

    # Quick smoke test with synthetic data (CPU, ~5 minutes)
    python benchmark/run_benchmark.py --quick

    # Single method/dataset
    python benchmark/run_benchmark.py --dataset mllmu_bench --method gradient_ascent --seed 42

    # From pre-computed results (skip experiments, just analysis)
    python benchmark/run_benchmark.py --results-only
"""

import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import get_logger, save_json, load_json, set_seed

logger = get_logger("benchmark")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   Multimodal Machine Unlearning — Metric Consistency Benchmark  ║
║   NeurIPS 2026 Datasets & Benchmarks Track                      ║
╚══════════════════════════════════════════════════════════════════╝
"""


def run_quick_mode():
    """
    Smoke-test mode: synthetic data, tiny model, CPU only.
    Verifies entire pipeline works without GPU.
    Produces real outputs/figures/tables but with synthetic numbers.
    """
    logger.info("Running in QUICK mode (synthetic data, no GPU required)")

    # Patch config for speed
    config.FINETUNE_EPOCHS = 1
    config.UNLEARN_EPOCHS  = 1
    config.CIFAR_EPOCHS    = 2
    config.DATASETS        = ["mllmu_bench"]
    config.SEEDS           = [42]
    config.DEVICE          = "cpu"

    # Generate synthetic results that mimic real experiment structure
    import numpy as np
    np.random.seed(42)

    methods   = config.METHODS
    datasets  = config.DATASETS
    seeds     = config.SEEDS

    # Simulate realistic metric values with known contradictions
    BASE_SCORES = {
        "gradient_ascent": {"FA": 0.08, "RA": 0.41, "MIA": 0.63, "AD": 2.31, "JS": 0.29},
        "random_labels":   {"FA": 0.19, "RA": 0.68, "MIA": 0.48, "AD": 1.74, "JS": 0.22},
        "finetune_retain": {"FA": 0.77, "RA": 0.89, "MIA": 0.21, "AD": 0.43, "JS": 0.09},
        "salun":           {"FA": 0.13, "RA": 0.61, "MIA": 0.54, "AD": 1.12, "JS": 0.19},
    }

    mm_results = []
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                scores = {k: v + np.random.normal(0, 0.02)
                          for k, v in BASE_SCORES[method].items()}
                scores = {k: float(np.clip(v, 0, 1)) for k, v in scores.items()
                          if k != "AD"}
                scores["AD"] = float(max(0.01, BASE_SCORES[method]["AD"]
                                         + np.random.normal(0, 0.1)))
                mm_results.append({
                    "dataset": dataset, "method": method, "seed": seed,
                    **scores,
                    "retrained_distance": float(scores.get("AD", 1.0) * 0.8),
                })

    # Unimodal baseline — higher metric agreement (smaller contradiction)
    uni_results = []
    for method in methods:
        for seed in seeds:
            scores = {k: v * 0.9 + np.random.normal(0, 0.01)
                      for k, v in BASE_SCORES[method].items()}
            scores = {k: float(np.clip(v, 0, 1)) for k, v in scores.items()
                      if k != "AD"}
            scores["AD"] = float(max(0.01, BASE_SCORES[method]["AD"] * 0.7))
            uni_results.append({
                "dataset": "cifar10", "method": method, "seed": seed, **scores,
                "retrained_distance": float(scores.get("AD", 1.0) * 0.6),
            })

    save_json(mm_results,
              os.path.join(config.RESULTS_DIR, "multimodal_results.json"))
    save_json(uni_results,
              os.path.join(config.RESULTS_DIR, "unimodal_results.json"))

    logger.info(f"Generated {len(mm_results)} multimodal + "
                f"{len(uni_results)} unimodal synthetic results")
    return mm_results, uni_results


def run_analysis_and_outputs(mm_results, uni_results):
    """Run all analysis, generate figures, tables, leaderboard."""
    from analysis.findings import run_full_analysis
    from analysis.visualise import generate_all_figures
    from evaluation.uqs import (derive_uqs_weights, uqs_ablation,
                                 build_ranking_table, compute_uqs)
    from benchmark.generate_tables import generate_all_latex_tables
    from benchmark.leaderboard import generate_leaderboard_html
    import numpy as np

    ret_dists = [r.get("retrained_distance", 0.0) for r in mm_results]

    # ── Findings ──────────────────────────────────────────────────────────────
    logger.info("Running analysis pipeline...")
    analysis = run_full_analysis(mm_results, uni_results, ret_dists)

    # ── UQS weights ───────────────────────────────────────────────────────────
    weights = derive_uqs_weights(
        mm_results, ret_dists,
        save_path=os.path.join(config.RESULTS_DIR, "uqs_weights.json"),
    )
    config.UQS_WEIGHTS = [weights[m] for m in config.METRIC_NAMES]

    # ── Ranking table ─────────────────────────────────────────────────────────
    from collections import defaultdict
    method_agg = defaultdict(lambda: defaultdict(list))
    for r in mm_results:
        for m in config.METRIC_NAMES:
            if m in r:
                method_agg[r["method"]][m].append(r[m])

    mean_results = {
        method: {m: float(np.mean(vals)) for m, vals in metrics.items()}
        for method, metrics in method_agg.items()
    }

    ranking_table = build_ranking_table(mean_results, weights)
    save_json(ranking_table,
              os.path.join(config.RESULTS_DIR, "ranking_table.json"))

    # ── UQS ablation ──────────────────────────────────────────────────────────
    ablation = uqs_ablation(mean_results, n_trials=500)
    save_json(ablation,
              os.path.join(config.RESULTS_DIR, "uqs_ablation.json"))

    # ── Figures ───────────────────────────────────────────────────────────────
    logger.info("Generating figures...")
    figure_paths = generate_all_figures(analysis, ablation)

    # ── LaTeX tables ──────────────────────────────────────────────────────────
    logger.info("Generating LaTeX tables...")
    os.makedirs(os.path.join(config.RESULTS_DIR, "tables"), exist_ok=True)
    generate_all_latex_tables(
        mm_results, mean_results, ranking_table, weights, ablation,
        output_dir=os.path.join(config.RESULTS_DIR, "tables"),
    )

    # ── Leaderboard HTML ──────────────────────────────────────────────────────
    logger.info("Generating leaderboard...")
    generate_leaderboard_html(
        mean_results, ranking_table, weights,
        save_path=os.path.join(config.RESULTS_DIR, "leaderboard.html"),
    )

    return analysis, ranking_table, ablation


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="Multimodal Unlearning Metric Consistency Benchmark"
    )
    parser.add_argument("--quick",        action="store_true",
                        help="Smoke test with synthetic data (no GPU needed)")
    parser.add_argument("--results-only", action="store_true",
                        help="Skip experiments, load saved results and re-run analysis")
    parser.add_argument("--dataset",      nargs="+", default=None)
    parser.add_argument("--method",       nargs="+", default=None)
    parser.add_argument("--seed",         nargs="+", type=int, default=None)
    parser.add_argument("--no-unimodal",  action="store_true",
                        help="Skip CIFAR-10 unimodal baseline")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    mm_results  = None
    uni_results = []

    # ── Mode: quick smoke test ─────────────────────────────────────────────
    if args.quick:
        mm_results, uni_results = run_quick_mode()

    # ── Mode: load pre-computed results ───────────────────────────────────
    elif args.results_only:
        mm_path  = os.path.join(config.RESULTS_DIR, "multimodal_results.json")
        uni_path = os.path.join(config.RESULTS_DIR, "unimodal_results.json")
        if not os.path.exists(mm_path):
            logger.error(f"No results found at {mm_path}. Run experiments first.")
            sys.exit(1)
        mm_results  = load_json(mm_path)
        uni_results = load_json(uni_path) if os.path.exists(uni_path) else []
        logger.info(f"Loaded {len(mm_results)} saved results.")

    # ── Mode: full experiments ────────────────────────────────────────────
    else:
        import main as pipeline
        datasets = args.dataset
        methods  = args.method
        seeds    = args.seed

        logger.info("Stage 1: Training vanilla + retrained models...")
        pipeline.stage_train(datasets, seeds)

        logger.info("Stage 2: Running unlearning methods...")
        pipeline.stage_unlearn(datasets, methods, seeds)

        logger.info("Stage 3: Evaluating all metrics...")
        mm_results = pipeline.stage_evaluate(datasets, methods, seeds)

        if not args.no_unimodal:
            logger.info("Stage 4: CIFAR-10 unimodal baseline...")
            pipeline.stage_unimodal()

        mm_path  = os.path.join(config.RESULTS_DIR, "multimodal_results.json")
        uni_path = os.path.join(config.RESULTS_DIR, "unimodal_results.json")
        mm_results  = load_json(mm_path)  if os.path.exists(mm_path)  else mm_results or []
        uni_results = load_json(uni_path) if os.path.exists(uni_path) else []

    # ── Analysis + outputs ────────────────────────────────────────────────
    logger.info("Running analysis and generating all outputs...")
    analysis, ranking_table, ablation = run_analysis_and_outputs(
        mm_results, uni_results
    )

    # ── Print summary ─────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nTime: {elapsed:.1f} minutes")
    print(f"\nOutputs saved to: {config.RESULTS_DIR}/")
    print("  multimodal_results.json  — all metric scores")
    print("  unimodal_results.json    — CIFAR-10 baseline")
    print("  uqs_weights.json         — derived UQS weights")
    print("  ranking_table.json       — method rankings")
    print("  figures/                 — 5 PDF figures for paper")
    print("  tables/                  — LaTeX tables for paper")
    print("  leaderboard.html         — interactive leaderboard")

    print("\n── UQS Leaderboard ──")
    sorted_methods = sorted(
        ranking_table.items(),
        key=lambda x: x[1].get("UQS", 0), reverse=True
    )
    print(f"  {'Method':<22} {'UQS':>6}  {'FA_rank':>7}  {'RA_rank':>7}  {'UQS_rank':>8}")
    print("  " + "-"*55)
    for method, scores in sorted_methods:
        print(f"  {method:<22} {scores.get('UQS',0):>6.4f}  "
              f"{scores.get('FA_rank','-'):>7}  "
              f"{scores.get('RA_rank','-'):>7}  "
              f"{scores.get('UQS_rank','-'):>8}")

    print("\n── Key Finding (Metric Contradiction) ──")
    print("  FA rank vs RA rank for each method:")
    for method, scores in sorted_methods:
        fa_r = scores.get("FA_rank", "?")
        ra_r = scores.get("RA_rank", "?")
        delta = abs((fa_r if isinstance(fa_r, int) else 0) -
                    (ra_r if isinstance(ra_r, int) else 0))
        bar = "★ CONTRADICTION" if delta >= 2 else ""
        print(f"  {method:<22} FA=#{fa_r}  RA=#{ra_r}  Δ={delta}  {bar}")

    print(f"\nTo view leaderboard: open {config.RESULTS_DIR}/leaderboard.html")
    print("="*60)


if __name__ == "__main__":
    main()
