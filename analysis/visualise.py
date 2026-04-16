"""
analysis/visualise.py — Generate all paper figures.

Figure 1: Kendall's Tau heatmap (5×5 metric pairs)
Figure 2: Contradiction table bar chart (ranks per method per metric)
Figure 3: Multimodal vs Unimodal metric agreement
Figure 4: UQS weight derivation (Spearman bars)
Figure 5: UQS ranking with ablation stability
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List
import os

from utils import get_logger
import config

logger = get_logger(__name__)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":    150,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
})

METHOD_COLORS = {
    "gradient_ascent":  "#E74C3C",
    "random_labels":    "#3498DB",
    "finetune_retain":  "#2ECC71",
    "salun":            "#9B59B6",
}
METHOD_LABELS = {
    "gradient_ascent":  "Grad. Ascent",
    "random_labels":    "Rand. Labels",
    "finetune_retain":  "FT-Retain",
    "salun":            "SalUn",
}


# ─── Figure 1: Kendall's Tau Heatmap ─────────────────────────────────────────
def plot_tau_heatmap(tau_data: Dict, save_path: str = None):
    """
    Plot 5×5 Kendall's Tau heatmap showing metric pair agreement.
    Near-zero/negative values = contradiction.
    """
    matrix = np.array(tau_data["tau_matrix"])
    labels = tau_data["metric_names"]

    fig, ax = plt.subplots(figsize=(6, 5))

    mask = np.eye(len(labels), dtype=bool)   # mask diagonal
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Kendall's τ", "shrink": 0.8},
    )

    ax.set_title("Metric Ranking Agreement\n(Kendall's τ across 4 methods × 3 datasets)",
                 pad=12)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Metric")

    # Highlight near-zero cells (contradictions)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and abs(matrix[i, j]) < 0.3:
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, fill=False,
                    edgecolor="black", linewidth=2.5,
                ))

    plt.tight_layout()
    path = save_path or os.path.join(config.FIGURES_DIR, "fig1_tau_heatmap.pdf")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved Figure 1 → {path}")
    return path


# ─── Figure 2: Contradiction Table (rank per method per metric) ───────────────
def plot_contradiction_table(table_rows: List[Dict], save_path: str = None):
    """
    Grouped bar chart: for each method, show its rank under each metric.
    Dramatic rank differences = contradiction.
    """
    methods = [r["method"] for r in table_rows]
    metrics = config.METRIC_NAMES
    n_methods = len(methods)
    n_metrics = len(metrics)

    x = np.arange(n_methods)
    width = 0.15
    offsets = np.linspace(-(n_metrics-1)/2, (n_metrics-1)/2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    metric_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]

    for idx, (metric, color, offset) in enumerate(
            zip(metrics, metric_colors, offsets)):
        ranks = [r[f"{metric}_rank"] for r in table_rows]
        bars  = ax.bar(
            x + offset, ranks,
            width, label=metric,
            color=color, alpha=0.85,
            edgecolor="white", linewidth=0.8,
        )
        for bar, rank in zip(bars, ranks):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"#{rank}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=10)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_ylim(0, n_methods + 1.2)
    ax.set_yticks(range(1, n_methods + 1))
    ax.invert_yaxis()
    ax.set_title("Method Rankings by Individual Metric\n"
                 "(Same method, different metric = different rank = contradiction)", pad=12)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(config.FIGURES_DIR, "fig2_contradiction.pdf")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved Figure 2 → {path}")
    return path


# ─── Figure 3: Multimodal vs Unimodal metric agreement ───────────────────────
def plot_modality_comparison(finding2: Dict, save_path: str = None):
    """
    Bar chart: mean pairwise Kendall's τ in multimodal vs unimodal.
    """
    import math
    import numpy as np

    mm_tau = finding2["multimodal_mean_tau"]
    uni_tau = finding2["unimodal_mean_tau"]

    # Replace NaN with 0 for plotting — annotate with note
    mm_val  = 0.0 if (mm_tau is None or (isinstance(mm_tau, float) and math.isnan(mm_tau))) else float(mm_tau)
    uni_val = 0.0 if (uni_tau is None or (isinstance(uni_tau, float) and math.isnan(uni_tau))) else float(uni_tau)

    labels = ["Multimodal\n(VQA)", "Unimodal\n(CIFAR-10)"]
    values = [mm_val, uni_val]
    colors = ["#E74C3C", "#3498DB"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5,
                  edgecolor="white", linewidth=1.5)

    for bar, v, raw in zip(bars, values, [mm_tau, uni_tau]):
        label = f"τ = {v:.3f}" if not (isinstance(raw, float) and math.isnan(raw)) else "τ = N/A"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                label,
                ha="center", va="bottom", fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Pairwise Kendall's τ\n(higher = metrics agree more)")
    ax.set_title("Metric Agreement:\nMultimodal vs Unimodal Settings", pad=12)
    ymin = min(values) - 0.2
    ymax = max(values) + 0.15
    if ymin == ymax:
        ymin, ymax = -0.3, 0.3
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.3)

    # Add delta as plain text instead of annotate arrow (avoids matplotlib crash)
    delta = finding2.get("delta", mm_val - uni_val)
    if not (isinstance(delta, float) and math.isnan(delta)):
        ax.text(0.97, 0.05, f"Δ = {delta:.3f}",
                transform=ax.transAxes, fontsize=9, color="#E74C3C",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaea", edgecolor="#E74C3C"))

    plt.savefig(save_path or os.path.join(config.FIGURES_DIR, "fig3_modality.pdf"),
                bbox_inches="tight")
    plt.close()
    path = save_path or os.path.join(config.FIGURES_DIR, "fig3_modality.pdf")
    logger.info(f"Saved Figure 3 → {path}")
    return path


# ─── Figure 4: UQS Weight Derivation (Spearman bars) ─────────────────────────
def plot_weight_derivation(reliability: Dict, save_path: str = None):
    """
    Horizontal bar chart: Spearman correlation of each metric with retrained model.
    Used to justify UQS weights.
    """
    metrics = config.METRIC_NAMES
    corrs   = [reliability[m]["spearman_r"] for m in metrics]
    total   = sum(max(c, 0.01) for c in corrs)
    weights = [max(c, 0.01) / total for c in corrs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Spearman correlations
    colors = ["#2ECC71" if c > 0.3 else "#E67E22" if c > 0 else "#E74C3C"
              for c in corrs]
    axes[0].barh(metrics, corrs, color=colors, alpha=0.85, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Spearman r with Retrained Model Distance")
    axes[0].set_title("Metric Reliability\n(correlation with gold standard)")
    for i, (c, m) in enumerate(zip(corrs, metrics)):
        # Place label inside bar for negative, outside for positive
        if c >= 0:
            axes[0].text(c + 0.01, i, f"{c:.3f}",
                         va="center", ha="left", fontsize=9)
        else:
            axes[0].text(c + 0.01, i, f"{c:.3f}",
                         va="center", ha="left", fontsize=9, color="white")
    axes[0].grid(axis="x", alpha=0.3)

    # Right: Derived UQS weights
    axes[1].barh(metrics, weights, color="#3498DB", alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("UQS Weight (normalised)")
    axes[1].set_title("Derived UQS Weights\n(proportional to reliability)")
    for i, (w, m) in enumerate(zip(weights, metrics)):
        axes[1].text(w + 0.005, i, f"{w:.3f}",
                     va="center", ha="left", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("From Metric Reliability to UQS Weights", fontsize=13, y=1.02)
    plt.tight_layout()
    path = save_path or os.path.join(config.FIGURES_DIR, "fig4_uqs_weights.pdf")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved Figure 4 → {path}")
    return path


# ─── Figure 5: UQS ranking stability (ablation) ───────────────────────────────
def plot_uqs_stability(ablation: Dict, save_path: str = None):
    """
    Show that UQS rankings are stable across weight variations.
    """
    mean_tau = ablation["mean_tau"]
    std_tau  = ablation["std_tau"]
    n_trials = ablation["n_trials"]

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.bar(["UQS Rankings\nStability"], [mean_tau],
           yerr=[std_tau], color="#3498DB", alpha=0.85,
           width=0.4, capsize=10, error_kw={"linewidth": 2})

    ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Perfect agreement")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="τ=0.5 threshold")

    ax.set_ylabel("Mean Kendall's τ\n(vs equal-weight UQS)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"UQS Rank Stability\nAcross {n_trials} Random Weight Variations")
    ax.text(0, mean_tau + std_tau + 0.03,
            f"τ = {mean_tau:.3f} ± {std_tau:.3f}",
            ha="center", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(config.FIGURES_DIR, "fig5_uqs_stability.pdf")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved Figure 5 → {path}")
    return path


# ─── Generate all figures at once ─────────────────────────────────────────────
def generate_all_figures(analysis: Dict, ablation: Dict):
    """Call all plot functions with saved analysis dict."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    paths = []

    if "finding1_tau_matrix" in analysis:
        paths.append(plot_tau_heatmap(analysis["finding1_tau_matrix"]))

    if "finding1_contradiction" in analysis:
        paths.append(plot_contradiction_table(analysis["finding1_contradiction"]))

    if "finding2_modality_compare" in analysis:
        try:
            paths.append(plot_modality_comparison(analysis["finding2_modality_compare"]))
        except Exception as e:
            logger.warning(f"[Fig3] Failed: {e}")

    if "finding3_reliability" in analysis:
        paths.append(plot_weight_derivation(analysis["finding3_reliability"]))

    if ablation:
        paths.append(plot_uqs_stability(ablation))

    logger.info(f"Generated {len(paths)} figures in {config.FIGURES_DIR}")
    return paths