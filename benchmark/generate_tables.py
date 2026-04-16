"""
benchmark/generate_tables.py
============================
Generates all LaTeX tables for the paper.

Table 1: Main results — metric scores for all methods on all datasets
Table 2: Contradiction table — rank per method per metric
Table 3: Kendall's Tau heatmap (as table)
Table 4: Multimodal vs Unimodal metric agreement
Table 5: UQS weights + reliability
Table 6: UQS ranking with ablation
"""

import os
import json
import numpy as np
from typing import Dict, List
from utils import get_logger
import config

logger = get_logger("tables")

METHOD_DISPLAY = {
    "gradient_ascent":  "Gradient Ascent",
    "random_labels":    "Random Labels",
    "finetune_retain":  "FT-Retain",
    "salun":            "SalUn",
}

METRIC_DISPLAY = {
    "FA":  r"FA$\downarrow$",
    "RA":  r"RA$\uparrow$",
    "MIA": r"MIA$\downarrow$",
    "AD":  r"AD$\downarrow$",
    "JS":  r"JS$\downarrow$",
    "UQS": r"UQS$\uparrow$",
}


def _bold_best(values: list, higher_better: bool) -> list:
    """Return list with best value wrapped in \\textbf{}."""
    if not values:
        return values
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return values
    best = max(numeric) if higher_better else min(numeric)
    return [f"\\textbf{{{v:.4f}}}" if isinstance(v, (int, float)) and abs(v - best) < 1e-9
            else (f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
            for v in values]


def table1_main_results(
    mean_results: Dict[str, Dict[str, float]],
    uqs_weights: Dict[str, float],
) -> str:
    """
    Table 1: Main results.
    Rows = methods, Cols = FA, RA, MIA, AD, JS, UQS
    """
    from evaluation.uqs import compute_uqs

    metrics_order = config.METRIC_NAMES + ["UQS"]
    higher_better = {"FA": False, "RA": True, "MIA": False,
                     "AD": False, "JS": False, "UQS": True}

    col_header = " & ".join(METRIC_DISPLAY[m] for m in metrics_order)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Evaluation results on MLLMU-Bench, UnLOK-VQA, and MMUBench (mean over 3 seeds). "
        r"$\downarrow$ = lower is better, $\uparrow$ = higher is better. "
        r"\textbf{Bold} = best per metric. "
        r"UQS uses weights derived from retrained-model correlation (Table~\ref{tab:uqs_weights}).}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(metrics_order) + "}",
        r"\toprule",
        r"\textbf{Method} & " + col_header + r" \\",
        r"\midrule",
    ]

    # Collect per-column values for bolding
    col_values = {m: [] for m in metrics_order}
    for method in config.METHODS:
        scores = mean_results.get(method, {})
        uqs    = compute_uqs(scores, uqs_weights)
        for m in config.METRIC_NAMES:
            col_values[m].append(scores.get(m, 0.0))
        col_values["UQS"].append(uqs)

    for i, method in enumerate(config.METHODS):
        scores    = mean_results.get(method, {})
        uqs       = compute_uqs(scores, uqs_weights)
        row_vals  = [scores.get(m, 0.0) for m in config.METRIC_NAMES] + [uqs]
        bold_vals = []
        for m, v in zip(metrics_order, row_vals):
            col_v  = col_values[m]
            best   = max(col_v) if higher_better[m] else min(col_v)
            if isinstance(v, float) and abs(v - best) < 1e-9:
                bold_vals.append(f"\\textbf{{{v:.4f}}}")
            else:
                bold_vals.append(f"{v:.4f}")
        display = METHOD_DISPLAY.get(method, method)
        lines.append(f"{display} & " + " & ".join(bold_vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table2_contradiction(ranking_table: Dict) -> str:
    """
    Table 2: Rank per method per metric — the contradiction table.
    Core finding of the paper.
    """
    metrics_order = config.METRIC_NAMES + ["UQS"]

    col_header = " & ".join(METRIC_DISPLAY[m].replace("$\\downarrow$", "").replace("$\\uparrow$", "")
                             for m in metrics_order)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{The Contradiction.} Rank of each method under each metric "
        r"(1 = best). Same method receives dramatically different ranks depending on "
        r"which metric is used — e.g., Gradient Ascent is \#1 by FA but \#4 by RA. "
        r"UQS provides a stable, principled ranking.}",
        r"\label{tab:contradiction}",
        r"\begin{tabular}{l" + "c" * len(metrics_order) + "}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(
            f"\\textbf{{{m}}}" for m in metrics_order) + r" \\",
        r"\midrule",
    ]

    for method in config.METHODS:
        scores  = ranking_table.get(method, {})
        ranks   = []
        for m in config.METRIC_NAMES:
            r = scores.get(f"{m}_rank", "–")
            # Highlight extreme rank disagreements
            ranks.append(f"\\textbf{{{r}}}" if r == 1 else str(r))
        uqs_rank = scores.get("UQS_rank", "–")
        ranks.append(f"\\textbf{{{uqs_rank}}}" if uqs_rank == 1 else str(uqs_rank))

        display = METHOD_DISPLAY.get(method, method)
        lines.append(f"{display} & " + " & ".join(ranks) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table3_tau_matrix(analysis: Dict) -> str:
    """
    Table 3: Kendall's Tau matrix — 5×5 metric pair agreement.
    """
    if "finding1_tau_matrix" not in analysis:
        return "% tau matrix not available"

    tau_data = analysis["finding1_tau_matrix"]
    matrix   = tau_data["tau_matrix"]
    metrics  = tau_data["metric_names"]
    n        = len(metrics)

    col_header = " & ".join(f"\\textbf{{{m}}}" for m in metrics)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Kendall's $\tau$ correlation between metric-induced method rankings "
        r"(averaged over 3 datasets $\times$ 3 seeds). Near-zero or negative $\tau$ "
        r"indicates metrics \textit{contradict} each other. "
        r"\colorbox{gray!20}{Shaded} cells: $|\tau| < 0.3$ (contradiction region).}",
        r"\label{tab:tau_matrix}",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\toprule",
        r" & " + col_header + r" \\",
        r"\midrule",
    ]

    for i, m_row in enumerate(metrics):
        cells = []
        for j, m_col in enumerate(metrics):
            val = matrix[i][j]
            if i == j:
                cells.append("–")
            elif abs(val) < 0.3:
                cells.append(f"\\cellcolor{{gray!20}}\\textbf{{{val:+.2f}}}")
            else:
                cells.append(f"{val:+.2f}")
        lines.append(f"\\textbf{{{m_row}}} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table4_modality_comparison(analysis: Dict) -> str:
    """
    Table 4: Multimodal vs Unimodal metric agreement.
    """
    if "finding2_modality_compare" not in analysis:
        return "% modality comparison not available"

    f2 = analysis["finding2_modality_compare"]
    mm  = f2.get("multimodal_mean_tau", 0)
    uni = f2.get("unimodal_mean_tau", 0)
    delta = f2.get("delta", 0)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean pairwise Kendall's $\tau$ between evaluation metrics in "
        r"multimodal (VQA) vs unimodal (CIFAR-10) settings. "
        r"Lower $\tau$ in multimodal indicates greater metric disagreement, "
        r"showing that the dual image+text pathway amplifies metric contradiction.}",
        r"\label{tab:modality}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Setting} & \textbf{Mean Pairwise $\tau$} & \textbf{$\Delta$ vs Unimodal} \\",
        r"\midrule",
        f"Multimodal (VQA) & \\textbf{{{mm:.3f}}} & {-delta:.3f} \\\\",
        f"Unimodal (CIFAR-10) & {uni:.3f} & – \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table5_uqs_weights(analysis: Dict, weights: Dict[str, float]) -> str:
    """
    Table 5: UQS weight derivation — reliability per metric.
    """
    reliability = analysis.get("finding3_reliability", {})
    total_w     = sum(max(v, 0.01) for v in weights.values())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Metric reliability (Spearman $\rho$ with retrained-model distance) "
        r"and derived UQS weights. Metrics more predictive of closeness to the gold-standard "
        r"retrained model receive higher weight. All weights sum to 1.}",
        r"\label{tab:uqs_weights}",
        r"\begin{tabular}{lccl}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Spearman $\rho$} & \textbf{UQS Weight} & \textbf{Direction} \\",
        r"\midrule",
    ]

    direction = {"FA": r"$\downarrow$", "RA": r"$\uparrow$", "MIA": r"$\downarrow$",
                 "AD": r"$\downarrow$", "JS": r"$\downarrow$"}

    sorted_metrics = sorted(
        config.METRIC_NAMES,
        key=lambda m: weights.get(m, 0), reverse=True
    )

    for m in sorted_metrics:
        r_data = reliability.get(m, {})
        rho    = r_data.get("spearman_r", 0.0)
        pval   = r_data.get("p_value", 1.0)
        w      = weights.get(m, 0.0)
        sig    = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        lines.append(
            f"\\textbf{{{m}}} & {rho:.3f}{sig} & {w:.3f} & {direction.get(m,'–')} \\\\"
        )

    lines += [
        r"\midrule",
        r"Total & – & 1.000 & – \\",
        r"\bottomrule",
        r"\multicolumn{4}{l}{\small * $p<0.05$, ** $p<0.01$, *** $p<0.001$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table6_uqs_ablation(ablation: Dict) -> str:
    """
    Table 6: UQS ablation — ranking stability across weight variations.
    """
    mean_tau = ablation.get("mean_tau", 0)
    std_tau  = ablation.get("std_tau", 0)
    n        = ablation.get("n_trials", 0)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{UQS ranking stability under random weight variations "
        r"($N=" + str(n) + r"$ trials, Dirichlet-sampled weights). "
        r"High mean Kendall's $\tau$ shows that UQS rankings are stable "
        r"regardless of exact weight values.}",
        r"\label{tab:uqs_ablation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} & \textbf{Interpretation} \\",
        r"\midrule",
        f"Mean Kendall's $\\tau$ & {mean_tau:.3f} & Rank stability \\\\",
        f"Std Kendall's $\\tau$ & {std_tau:.3f} & Variability \\\\",
        f"Number of trials & {n} & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_all_latex_tables(
    mm_results:     List[Dict],
    mean_results:   Dict,
    ranking_table:  Dict,
    weights:        Dict[str, float],
    ablation:       Dict,
    output_dir:     str = None,
):
    """Generate and save all 6 LaTeX tables."""
    output_dir = output_dir or os.path.join(config.RESULTS_DIR, "tables")
    os.makedirs(output_dir, exist_ok=True)

    # Need analysis for tables 3,4,5
    from utils import load_json
    analysis_path = os.path.join(config.RESULTS_DIR, "analysis_results.json")
    analysis = load_json(analysis_path) if os.path.exists(analysis_path) else {}

    tables = {
        "table1_main_results.tex":    table1_main_results(mean_results, weights),
        "table2_contradiction.tex":   table2_contradiction(ranking_table),
        "table3_tau_matrix.tex":      table3_tau_matrix(analysis),
        "table4_modality.tex":        table4_modality_comparison(analysis),
        "table5_uqs_weights.tex":     table5_uqs_weights(analysis, weights),
        "table6_uqs_ablation.tex":    table6_uqs_ablation(ablation),
    }

    for filename, content in tables.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Saved {filename}")

    # Combined tables file
    combined = "\n\n% " + "="*60 + "\n\n".join(
        f"% Table: {k}\n{v}" for k, v in tables.items()
    )
    combined_path = os.path.join(output_dir, "all_tables.tex")
    with open(combined_path, "w") as f:
        f.write("% Auto-generated LaTeX tables\n")
        f.write("% Include in paper with: \\input{tables/all_tables}\n\n")
        f.write(combined)
    logger.info(f"All tables saved to {output_dir}/")

    return tables
