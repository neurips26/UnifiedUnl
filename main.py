"""
main.py — Master pipeline. Runs everything end-to-end.

Usage:
    python main.py --stage all          # Full pipeline
    python main.py --stage train        # Only train vanilla + retrained models
    python main.py --stage unlearn      # Only run unlearning methods
    python main.py --stage evaluate     # Only compute metrics
    python main.py --stage analyse      # Only run analysis + generate figures
    python main.py --stage unimodal     # Only run CIFAR-10 baseline
    python main.py --dataset mllmu_bench --method gradient_ascent --seed 42  # Single run
"""

# ── MUST be set before importing transformers or datasets ──────────────────
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# ───────────────────────────────────────────────────────────────────────────

import argparse
import sys
import time
import json

# ─── Setup path ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import get_logger, set_seed, save_json, load_json

logger = get_logger("main")


# ─── Stage 1: Train vanilla + retrained models ────────────────────────────────
def stage_train(datasets=None, seeds=None):
    """Fine-tune LLaVA on each dataset. Save vanilla + retrained checkpoints."""
    from data.loader import load_dataset_by_name
    from data.dataset import make_dataloader
    from models.llava_model import load_llava, apply_lora, full_train
    import torch

    datasets = datasets or config.DATASETS
    seeds    = seeds    or config.SEEDS[:1]   # use 1 seed for training

    model_base, processor = load_llava()
    model_base = apply_lora(model_base)

    for dataset_name in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAIN | Dataset: {dataset_name}")

        print(f"  Loading dataset: {dataset_name}...")
        data = load_dataset_by_name(dataset_name)
        print(f"  Dataset loaded: "
              f"train={len(data['train'])}, forget={len(data['forget'])}, "
              f"retain={len(data['retain'])}, test={len(data['test'])}")

        for seed in seeds:
            set_seed(seed)
            print(f"\n>>> Seed {seed} | dataset={dataset_name}")

            # ── Vanilla model (trained on full train set) ──
            vanilla_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"vanilla_{dataset_name}_seed{seed}.pt"
            )
            if os.path.exists(vanilla_path):
                print(f"  [SKIP] Vanilla checkpoint already exists: {vanilla_path}")
            else:
                import copy
                print(f"  [1/2] Training VANILLA model on full train set...")
                vanilla_model = copy.deepcopy(model_base)
                train_loader  = make_dataloader(
                    data["train"], processor,
                    batch_size=config.FINETUNE_BATCH, shuffle=True,
                )
                print(f"       Samples: {len(data['train'])} | "
                      f"Batches: {len(train_loader)} | "
                      f"Epochs: {config.FINETUNE_EPOCHS}")
                full_train(vanilla_model, train_loader,
                           n_epochs=config.FINETUNE_EPOCHS,
                           save_path=vanilla_path,
                           label=f"Vanilla/{dataset_name}")
                print(f"  [1/2] DONE. Saved -> {vanilla_path}")

            # ── Retrained model (trained without forget set) ──
            retrained_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"retrained_{dataset_name}_seed{seed}.pt"
            )
            if os.path.exists(retrained_path):
                print(f"  [SKIP] Retrained checkpoint already exists: {retrained_path}")
            else:
                import copy
                print(f"  [2/2] Training RETRAINED model on retain set (no forget data)...")
                retrained_model = copy.deepcopy(model_base)
                retain_loader   = make_dataloader(
                    data["retain"], processor,
                    batch_size=config.FINETUNE_BATCH, shuffle=True,
                )
                print(f"       Samples: {len(data['retain'])} | "
                      f"Batches: {len(retain_loader)} | "
                      f"Epochs: {config.FINETUNE_EPOCHS}")
                full_train(retrained_model, retain_loader,
                           n_epochs=config.FINETUNE_EPOCHS,
                           save_path=retrained_path,
                           label=f"Retrained/{dataset_name}")
                print(f"  [2/2] DONE. Saved -> {retrained_path}")

    logger.info("Stage 1 (Train) complete.")


# ─── Stage 2: Run unlearning ──────────────────────────────────────────────────
def stage_unlearn(datasets=None, methods=None, seeds=None):
    """Apply each unlearning method to each vanilla model."""
    from data.loader import load_dataset_by_name
    from models.llava_model import load_llava, apply_lora, load_checkpoint
    from unlearning.methods import run_unlearning
    import copy

    datasets = datasets or config.DATASETS
    methods  = methods  or config.METHODS
    seeds    = seeds    or config.SEEDS

    _, processor = load_llava()

    for dataset_name in datasets:
        data = load_dataset_by_name(dataset_name)

        for seed in seeds:
            # Load vanilla model
            vanilla_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"vanilla_{dataset_name}_seed{seed}.pt"
            )
            if not os.path.exists(vanilla_path):
                # Fall back to seed 42 checkpoint — methodologically correct:
                # seeds control unlearning randomness, not base model training.
                # This is standard practice in TOFU, OpenUnlearning etc.
                fallback_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"vanilla_{dataset_name}_seed42.pt"
                )
                if os.path.exists(fallback_path):
                    logger.info(f"  [Seed fallback] Using seed42 vanilla checkpoint for seed={seed} "
                                f"(unlearning seed controls optimizer randomness, not base model)")
                    vanilla_path = fallback_path
                else:
                    logger.warning(f"Vanilla checkpoint not found: {vanilla_path}. Run --stage train first.")
                    continue

            # Load base model architecture
            base_model, _ = load_llava()
            base_model     = apply_lora(base_model)
            vanilla_model, _ = load_checkpoint(base_model, vanilla_path)
            vanilla_model.to(config.DEVICE)

            for method_name in methods:
                out_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"unlearned_{dataset_name}_{method_name}_seed{seed}.pt"
                )
                if os.path.exists(out_path):
                    logger.info(f"Already exists: {out_path}")
                    continue

                logger.info(f"\nUnlearning | {dataset_name} | {method_name} | seed={seed}")
                t0 = time.time()

                unlearned = run_unlearning(
                    method_name  = method_name,
                    model        = vanilla_model,
                    forget_set   = data["forget"],
                    retain_set   = data["retain"],
                    processor    = processor,
                    seed         = seed,
                )

                # Save IMMEDIATELY after method returns — before any cleanup code
                from utils import save_checkpoint
                save_checkpoint(unlearned, out_path,
                                metadata={"dataset": dataset_name,
                                          "method": method_name,
                                          "seed": seed})

                elapsed = time.time() - t0
                logger.info(f"  Done in {elapsed/60:.1f} min → {out_path}")

    logger.info("Stage 2 (Unlearn) complete.")


# ─── Stage 3: Evaluate metrics ────────────────────────────────────────────────
def stage_evaluate(datasets=None, methods=None, seeds=None):
    """Compute all 5 metrics for every unlearned model."""
    from data.loader import load_dataset_by_name
    from models.llava_model import load_llava, apply_lora, load_checkpoint
    from evaluation.metrics import evaluate_all_metrics, compute_retrained_distance

    datasets = datasets or config.DATASETS
    methods  = methods  or config.METHODS
    seeds    = seeds    or config.SEEDS

    _, processor = load_llava()

    all_results = []
    all_retrained_distances = []

    for dataset_name in datasets:
        data = load_dataset_by_name(dataset_name)

        for seed in seeds:
            set_seed(seed)

            # Load retrained model
            retrained_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"retrained_{dataset_name}_seed{seed}.pt"
            )
            if not os.path.exists(retrained_path):
                fallback_retrained = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"retrained_{dataset_name}_seed42.pt"
                )
                if os.path.exists(fallback_retrained):
                    logger.info(f"  [Seed fallback] Using seed42 retrained checkpoint for seed={seed}")
                    retrained_path = fallback_retrained
                else:
                    logger.warning(f"Retrained checkpoint missing: {retrained_path}")
                    continue

            base_model, _ = load_llava()
            base_model = apply_lora(base_model)
            retrained_model, _ = load_checkpoint(base_model, retrained_path)
            retrained_model.to(config.DEVICE)

            for method_name in methods:
                unlearned_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"unlearned_{dataset_name}_{method_name}_seed{seed}.pt"
                )
                if not os.path.exists(unlearned_path):
                    logger.warning(f"Unlearned checkpoint missing: {unlearned_path}")
                    continue

                base_m2, _ = load_llava()
                base_m2 = apply_lora(base_m2)
                unlearned_model, _ = load_checkpoint(base_m2, unlearned_path)
                unlearned_model.to(config.DEVICE)

                logger.info(f"\nEval | {dataset_name} | {method_name} | seed={seed}")

                # ── Per-combination result cache ──────────────────────────────
                eval_cache_path = os.path.join(
                    config.RESULTS_DIR,
                    f"eval_{dataset_name}_{method_name}_seed{seed}.json"
                )
                os.makedirs(config.RESULTS_DIR, exist_ok=True)
                if os.path.exists(eval_cache_path):
                    logger.info(f"  [SKIP] Eval cache exists: {eval_cache_path}")
                    cached = load_json(eval_cache_path)
                    all_results.append(cached)
                    all_retrained_distances.append(cached.get("retrained_distance", 0.0))
                    continue

                metric_cache_path = os.path.join(
                    config.RESULTS_DIR,
                    f"metrics_{dataset_name}_{method_name}_seed{seed}.json"
                )
                metrics = evaluate_all_metrics(
                    unlearned_model   = unlearned_model,
                    retrained_model   = retrained_model,
                    processor         = processor,
                    forget_set        = data["forget"],
                    retain_set        = data["retain"],
                    metric_cache_path = metric_cache_path,
                )

                ret_dist = compute_retrained_distance(
                    unlearned_model, retrained_model, processor, data["forget"]
                )

                result = {
                    "dataset":            dataset_name,
                    "method":             method_name,
                    "seed":               seed,
                    **metrics,
                    "retrained_distance": ret_dist,
                }
                all_results.append(result)
                all_retrained_distances.append(ret_dist)

                # Save immediately — so crashes don't lose completed metrics
                save_json(result, eval_cache_path)
                logger.info(f"  Cached → {eval_cache_path}")
                logger.info(f"  {metrics}")

                # Free GPU memory
                del unlearned_model
                import torch; torch.cuda.empty_cache()

    save_json(all_results,
              os.path.join(config.RESULTS_DIR, "multimodal_results.json"))
    logger.info(f"Stage 3 complete. {len(all_results)} results saved.")
    return all_results


# ─── Stage 4: Analysis + Figures ─────────────────────────────────────────────
def stage_analyse():
    """Load saved results and run full analysis pipeline."""
    import numpy as np
    from analysis.findings import run_full_analysis
    from analysis.visualise import generate_all_figures
    from evaluation.uqs import derive_uqs_weights, uqs_ablation, build_ranking_table

    # Load results
    mm_path  = os.path.join(config.RESULTS_DIR, "multimodal_results.json")
    uni_path = os.path.join(config.RESULTS_DIR, "unimodal_results.json")

    if not os.path.exists(mm_path):
        logger.error(f"Multimodal results not found: {mm_path}")
        return

    mm_results  = load_json(mm_path)
    uni_results = load_json(uni_path) if os.path.exists(uni_path) else []

    ret_dists = [r.get("retrained_distance", 0.0) for r in mm_results]

    # Run all findings
    analysis = run_full_analysis(mm_results, uni_results, ret_dists)

    # Derive UQS weights
    weights = derive_uqs_weights(
        mm_results, ret_dists,
        save_path=os.path.join(config.RESULTS_DIR, "uqs_weights.json"),
    )

    # Update config with derived weights
    config.UQS_WEIGHTS = [weights[m] for m in config.METRIC_NAMES]

    # Build ranking table for paper
    # Aggregate mean per (method, dataset)
    from collections import defaultdict
    method_agg = defaultdict(lambda: defaultdict(list))
    for r in mm_results:
        for m in config.METRIC_NAMES:
            method_agg[r["method"]][m].append(r[m])

    mean_results = {
        method: {m: float(np.mean(vals)) for m, vals in metrics.items()}
        for method, metrics in method_agg.items()
    }

    ranking_table = build_ranking_table(mean_results, weights)
    save_json(ranking_table,
              os.path.join(config.RESULTS_DIR, "ranking_table.json"))

    # UQS ablation
    ablation = uqs_ablation(mean_results)
    save_json(ablation, os.path.join(config.RESULTS_DIR, "uqs_ablation.json"))

    # Generate all figures
    generate_all_figures(analysis, ablation)

    logger.info("Stage 4 (Analyse) complete.")
    return analysis


# ─── Stage 5: Unimodal baseline ───────────────────────────────────────────────
def stage_unimodal():
    from scripts.unimodal_baseline import run_unimodal_pipeline
    run_unimodal_pipeline()
    logger.info("Stage 5 (Unimodal) complete.")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Unlearning Metric Evaluation Pipeline"
    )
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "train", "unlearn", "evaluate",
                                 "analyse", "unimodal"],
                        help="Which stage to run")
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="Dataset(s) to use (default: all from config)")
    parser.add_argument("--method", nargs="+", default=None,
                        help="Method(s) to use (default: all from config)")
    parser.add_argument("--seed", nargs="+", type=int, default=None,
                        help="Seed(s) to use (default: all from config)")

    args = parser.parse_args()

    datasets = args.dataset
    methods  = args.method
    seeds    = args.seed

    logger.info(f"Pipeline started | stage={args.stage} | "
                f"datasets={datasets or 'all'} | "
                f"methods={methods or 'all'} | "
                f"seeds={seeds or 'all'}")

    t_start = time.time()

    if args.stage in ("all", "train"):
        stage_train(datasets, seeds)

    if args.stage in ("all", "unlearn"):
        stage_unlearn(datasets, methods, seeds)

    if args.stage in ("all", "evaluate"):
        stage_evaluate(datasets, methods, seeds)

    if args.stage in ("all", "unimodal"):
        stage_unimodal()

    if args.stage in ("all", "analyse"):
        stage_analyse()

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nPipeline complete in {elapsed:.1f} minutes.")


if __name__ == "__main__":
    main()