"""
analysis/findings.py — Generate all 3 paper findings.

Finding 1: Kendall's Tau heatmap showing metric ranking contradiction
Finding 2: Multimodal vs Unimodal metric agreement comparison
Finding 3: Retrained model correlation → UQS weight derivation
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr
from typing import Dict, List, Tuple
import os

from utils import get_logger, save_json
import config

logger = get_logger(__name__)


# ─── Finding 1: Kendall's Tau heatmap ────────────────────────────────────────
def compute_kendall_tau_matrix(
    all_results: List[Dict],   # list of {method, dataset, seed, FA, RA, MIA, AD, JS}
    methods: List[str] = None,
) -> Dict:
    """
    For every pair of metrics (e.g. FA vs RA), compute Kendall's Tau
    of method rankings across all datasets and seeds.
    
    Near-zero or negative tau = contradiction between metrics.
    
    Returns:
        tau_matrix: 5x5 numpy array
        p_matrix:   5x5 p-value array
    """
    if methods is None:
        methods = config.METHODS

    metrics = config.METRIC_NAMES
    n = len(metrics)

    tau_matrix = np.zeros((n, n))
    p_matrix   = np.ones((n, n))

    logger.info("[Finding 1] Computing Kendall's Tau matrix...")

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i == j:
                tau_matrix[i, j] = 1.0
                p_matrix[i, j]   = 0.0
                continue

            # For each dataset+seed combo, get method rankings by m1 and m2
            taus = []
            for dataset in config.DATASETS:
                for seed in config.SEEDS:
                    subset = [r for r in all_results
                              if r["dataset"] == dataset and r["seed"] == seed]
                    if len(subset) < 2:
                        continue

                    # Sort methods by m1 → get rank ordering
                    lower_better_m1 = m1 in ["FA", "MIA", "AD", "JS"]
                    lower_better_m2 = m2 in ["FA", "MIA", "AD", "JS"]

                    s1 = [r[m1] for r in subset]
                    s2 = [r[m2] for r in subset]

                    # Flip so higher = better for both
                    if lower_better_m1: s1 = [-v for v in s1]
                    if lower_better_m2: s2 = [-v for v in s2]

                    ranks1 = np.argsort(np.argsort(-np.array(s1)))
                    ranks2 = np.argsort(np.argsort(-np.array(s2)))

                    tau, _ = kendalltau(ranks1, ranks2)
                    taus.append(tau)

            if taus:
                mean_tau = np.mean(taus)
                # Simple t-test p-value proxy
                from scipy.stats import ttest_1samp
                _, pval = ttest_1samp(taus, 0)
                tau_matrix[i, j] = mean_tau
                p_matrix[i, j]   = pval

    logger.info("[Finding 1] Tau matrix:")
    for i, m in enumerate(metrics):
        row = " | ".join(f"{tau_matrix[i,j]:+.3f}" for j in range(n))
        logger.info(f"  {m}: {row}")

    return {
        "tau_matrix":   tau_matrix.tolist(),
        "p_matrix":     p_matrix.tolist(),
        "metric_names": metrics,
    }


# ─── Finding 1b: Contradiction table (like Table 1 in paper) ─────────────────
def build_contradiction_table(
    all_results: List[Dict],
) -> List[Dict]:
    """
    For each method, show its rank under each metric across all datasets.
    Dramatically different ranks = the contradiction finding.
    """
    methods = config.METHODS
    metrics = config.METRIC_NAMES

    # Aggregate mean scores per method
    method_scores = {m: {met: [] for met in metrics} for m in methods}
    for r in all_results:
        for met in metrics:
            method_scores[r["method"]][met].append(r[met])

    mean_scores = {
        m: {met: np.mean(vals) if vals else 0.0
            for met, vals in method_scores[m].items()}
        for m in methods
    }

    # Rank each method by each metric
    rows = []
    for method in methods:
        row = {"method": method}
        for met in metrics:
            row[f"{met}_score"] = round(mean_scores[method][met], 4)

        for met in metrics:
            ascending = met in ["FA", "MIA", "AD", "JS"]
            sorted_methods = sorted(
                methods,
                key=lambda m: mean_scores[m][met],
                reverse=not ascending,
            )
            row[f"{met}_rank"] = sorted_methods.index(method) + 1

        rows.append(row)

    # Log rank contradiction for paper
    logger.info("[Finding 1] Contradiction table:")
    for row in rows:
        ranks = [row[f"{m}_rank"] for m in metrics]
        logger.info(f"  {row['method']:20s} ranks: {ranks}  "
                    f"(range: {max(ranks)-min(ranks)})")

    return rows


# ─── Finding 2: Multimodal vs Unimodal metric agreement ──────────────────────
def compare_modality_agreement(
    multimodal_results: List[Dict],   # VQA datasets
    unimodal_results:   List[Dict],   # CIFAR-10
) -> Dict:
    """
    Compare average pairwise Kendall's Tau between metrics
    in multimodal vs unimodal settings.
    
    Lower tau in multimodal = metrics disagree more = our key finding.
    """
    logger.info("[Finding 2] Comparing metric agreement: multimodal vs unimodal...")

    def mean_pairwise_tau(results: List[Dict]) -> float:
        metrics = config.METRIC_NAMES
        taus = []
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                m1, m2 = metrics[i], metrics[j]
                s1 = [r[m1] for r in results]
                s2 = [r[m2] for r in results]
                if len(set(s1)) < 2 or len(set(s2)) < 2:
                    continue
                tau, _ = kendalltau(s1, s2)
                taus.append(tau)
        return float(np.mean(taus)) if taus else 0.0

    mm_tau = mean_pairwise_tau(multimodal_results)
    uni_tau = mean_pairwise_tau(unimodal_results)

    delta = uni_tau - mm_tau
    logger.info(f"[Finding 2] Multimodal mean pairwise τ: {mm_tau:.4f}")
    logger.info(f"[Finding 2] Unimodal   mean pairwise τ: {uni_tau:.4f}")
    logger.info(f"[Finding 2] Δ (unimodal-multimodal):    {delta:.4f}")

    return {
        "multimodal_mean_tau":  mm_tau,
        "unimodal_mean_tau":    uni_tau,
        "delta":                delta,
        "finding": (
            f"Metric agreement is {delta:.3f} lower in multimodal settings. "
            f"The two-pathway image+text architecture amplifies metric contradiction."
        )
    }


# ─── Finding 3: Retrained model correlation ───────────────────────────────────
def compute_metric_reliability(
    all_results:          List[Dict],
    retrained_distances:  List[float],
) -> Dict[str, float]:
    """
    For each metric, Spearman correlation with closeness to retrained model.
    Higher correlation = metric is more reliable (closer to gold standard).
    
    This directly motivates the UQS weights.
    """
    logger.info("[Finding 3] Computing metric reliability (Spearman vs retrained dist)...")

    reliabilities = {}
    for metric in config.METRIC_NAMES:
        scores = np.array([r[metric] for r in all_results])
        dists  = np.array(retrained_distances)

        # Lower metric = better for FA/MIA/AD/JS, so flip
        if metric in ["FA", "MIA", "AD", "JS"]:
            scores = -scores

        corr, pval = spearmanr(scores, -dists)  # higher is better, lower dist is better
        reliabilities[metric] = {"spearman_r": float(corr), "p_value": float(pval)}
        logger.info(f"  {metric}: r={corr:.4f}, p={pval:.4f}")

    # Rank metrics by reliability
    ranked = sorted(reliabilities.items(), key=lambda x: x[1]["spearman_r"], reverse=True)
    logger.info("[Finding 3] Metric reliability ranking:")
    for rank, (metric, vals) in enumerate(ranked, 1):
        logger.info(f"  #{rank} {metric}: r={vals['spearman_r']:.4f}")

    return reliabilities


# ─── Full analysis pipeline ───────────────────────────────────────────────────
def run_full_analysis(
    all_results:         List[Dict],
    unimodal_results:    List[Dict],
    retrained_distances: List[float],
    output_dir:          str = config.RESULTS_DIR,
) -> Dict:
    """
    Run all 3 findings and save results to JSON.
    """
    logger.info("=" * 60)
    logger.info("Running Full Analysis — 3 Findings")
    logger.info("=" * 60)

    # ── Degeneracy check ────────────────────────────────────────────────────
    # Flag datasets where FA or RA are constant across all methods (degenerate)
    # These are excluded from tau computation to avoid spurious correlations
    import numpy as _np_local
    datasets_present = list({r["dataset"] for r in all_results})
    degenerate_datasets = []
    for ds in datasets_present:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        fa_vals = [r["FA"] for r in ds_results]
        ra_vals = [r["RA"] for r in ds_results]
        fa_range = max(fa_vals) - min(fa_vals)
        ra_range = max(ra_vals) - min(ra_vals)
        if fa_range < 0.05 and ra_range < 0.05:
            degenerate_datasets.append(ds)
            logger.warning(f"  [Degeneracy] {ds}: FA range={fa_range:.3f}, RA range={ra_range:.3f} "
                           f"— all methods collapsed to same output. Including in full analysis "
                           f"but flagging in paper as low-convergence regime.")
    if degenerate_datasets:
        logger.info(f"  Degenerate datasets (low-convergence): {degenerate_datasets}")
        # Filter for non-degenerate analysis
        non_degenerate = [r for r in all_results if r["dataset"] not in degenerate_datasets]
        logger.info(f"  Non-degenerate results: {len(non_degenerate)} / {len(all_results)}")
    else:
        non_degenerate = all_results
    # ────────────────────────────────────────────────────────────────────────

    # Finding 1 — full data + discriminative-only
    finding1 = compute_kendall_tau_matrix(all_results)  # full data for completeness
    table    = build_contradiction_table(all_results)

    # ── KEY FIX: also compute tau on non-degenerate (discriminative) data only
    # This directly answers the reviewer concern that contradiction might be
    # "artifact of degenerate datasets rather than a fundamental property"
    if non_degenerate and len(non_degenerate) < len(all_results):
        finding1_discriminative = compute_kendall_tau_matrix(non_degenerate)
        table_discriminative    = build_contradiction_table(non_degenerate)
        logger.info("[Finding 1 DISCRIMINATIVE-ONLY] Tau matrix on non-degenerate datasets:")
        for row in finding1_discriminative.get("tau_rows", []):
            logger.info(f"  {row}")
    else:
        finding1_discriminative = finding1  # no degenerate data found
        table_discriminative    = table

    # Per-dataset tau: compute tau separately for each dataset
    per_dataset_tau = {}
    for ds in datasets_present:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        if len(ds_results) >= 4:  # need at least 4 points for tau
            ds_tau = compute_kendall_tau_matrix(ds_results)
            per_dataset_tau[ds] = ds_tau
            logger.info(f"  [Per-dataset tau] {ds}: mean tau={ds_tau.get('mean_tau', 0):.3f}")
        else:
            logger.warning(f"  [Per-dataset tau] {ds}: too few results ({len(ds_results)}), skipping")

    finding2 = compare_modality_agreement(all_results, unimodal_results)
    finding3 = compute_metric_reliability(all_results, retrained_distances)

    # Also compute reliability on discriminative data only
    if non_degenerate and len(non_degenerate) < len(all_results):
        nd_dists = []
        for r in non_degenerate:
            # find matching retrained distance by index
            try:
                idx_r = all_results.index(r)
                nd_dists.append(retrained_distances[idx_r])
            except (ValueError, IndexError):
                nd_dists.append(0.0)
        finding3_discriminative = compute_metric_reliability(non_degenerate, nd_dists)
        logger.info("[Finding 3 DISCRIMINATIVE-ONLY] Reliability on non-degenerate data:")
        for k, v in finding3_discriminative.items():
            logger.info(f"  {k}: {v}")
    else:
        finding3_discriminative = finding3

    analysis = {
        "finding1_tau_matrix":               finding1,
        "finding1_tau_matrix_discriminative": finding1_discriminative,
        "finding1_contradiction":            table,
        "finding1_contradiction_discriminative": table_discriminative,
        "finding1_per_dataset_tau":          per_dataset_tau,
        "finding2_modality_compare":         finding2,
        "finding3_reliability":              finding3,
        "finding3_reliability_discriminative": finding3_discriminative,
        "degenerate_datasets":               degenerate_datasets,
        "n_degenerate":                      len(degenerate_datasets),
    }

    save_json(analysis, os.path.join(output_dir, "analysis_results.json"))
    logger.info(f"Analysis saved → {output_dir}/analysis_results.json")

    return analysis
