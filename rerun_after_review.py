"""
rerun_after_review.py  —  NeurIPS 2026 Reviewer Fix Script
===========================================================
Run from project root:
    cd <your project dir>
    py rerun_after_review.py

BEFORE running, replace these 4 files in your project:
    evaluation/metrics.py    <- updated MIA (3-feature + CV)
    evaluation/uqs.py        <- updated normalization (exp(-AD/100))
    analysis/findings.py     <- updated analysis (discriminative tau)
    neurips_paper/main.tex   <- updated paper (all 5 fixes documented)

Time estimate: ~8-12 hours (full re-evaluate due to cache clear)
"""

import os, glob, subprocess, sys, json, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS  = os.path.join(BASE_DIR, "outputs")


def run(cmd, desc):
    print(f"\n{'='*70}\n  {desc}\n  > {cmd}\n{'='*70}\n")
    t0 = time.time()
    r  = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    elapsed = (time.time() - t0) / 60
    if r.returncode != 0:
        print(f"\n  FAILED after {elapsed:.1f} min — fix the error and re-run.")
        sys.exit(1)
    print(f"  Done in {elapsed:.1f} min")


def clear_eval_caches():
    print("\n" + "="*70)
    print("  Clearing evaluation caches (checkpoints preserved)")
    print("="*70)
    killed = 0
    for pattern in ["eval_*.json", "metrics_*.json", "uqs_*.json",
                     "analysis_results.json", "ranking_table.json", "uqs_ablation.json"]:
        for f in glob.glob(os.path.join(OUTPUTS, pattern)):
            os.remove(f)
            print(f"  deleted: {os.path.basename(f)}")
            killed += 1
    print(f"\n  {killed} cache files cleared.")


def verify_files():
    """Confirm the 4 updated files are in place before running."""
    print("\n" + "="*70)
    print("  Verifying updated files are in place...")
    print("="*70)
    required = [
        ("evaluation/metrics.py", "cross_val_predict",   "3-feature CV MIA"),
        ("evaluation/uqs.py",     "exp(-",               "cohort-independent AD"),
        ("analysis/findings.py",  "non_degenerate",      "discriminative tau"),
    ]
    ok = True
    for path, marker, label in required:
        full = os.path.join(BASE_DIR, path)
        if not os.path.exists(full):
            print(f"  MISSING FILE: {path}")
            ok = False
        elif marker not in open(full).read():
            print(f"  WRONG VERSION: {path} — marker '{marker}' not found")
            print(f"  Did you replace with the updated file? ({label})")
            ok = False
        else:
            print(f"  OK: {path} ({label})")
    if not ok:
        print("\n  Fix the above then re-run.")
        sys.exit(1)


def print_results():
    """Print key numbers after re-run."""
    path = os.path.join(OUTPUTS, "analysis_results.json")
    if not os.path.exists(path):
        print("  (no results yet)")
        return
    r = json.load(open(path))

    print("\n" + "="*70)
    print("  KEY RESULTS AFTER FIXES")
    print("="*70)

    # Full tau
    tau_rows = r.get("finding1_tau_matrix", {}).get("tau_rows", [])
    print("\n  Tau matrix (all data):")
    for row in tau_rows:
        print(f"    {row}")

    # Discriminative tau
    tau_d = r.get("finding1_tau_matrix_discriminative", {})
    d_rows = tau_d.get("tau_rows", [])
    if d_rows != tau_rows:
        print("\n  Tau matrix (MMUBench only — discriminative):")
        for row in d_rows:
            print(f"    {row}")

    # Per-dataset
    per_ds = r.get("finding1_per_dataset_tau", {})
    if per_ds:
        print("\n  Per-dataset mean pairwise tau:")
        for ds, v in per_ds.items():
            print(f"    {ds}: {v.get('mean_tau', '?'):.3f}")

    # Finding 2
    f2 = r.get("finding2_modality_compare", {})
    print(f"\n  F2: multimodal={f2.get('multimodal_mean_tau','?'):.3f}  "
          f"unimodal={f2.get('unimodal_mean_tau','?'):.3f}  "
          f"delta={f2.get('delta','?'):.3f}")

    # Finding 3
    f3 = r.get("finding3_reliability", {})
    print("\n  F3: reliability:")
    for m, v in f3.items():
        if isinstance(v, dict):
            print(f"    {m}: rho={v.get('spearman_r','?'):.3f}  p={v.get('p_value','?'):.4f}")

    print(f"\n  Degenerate datasets: {r.get('degenerate_datasets', [])}")


if __name__ == "__main__":
    print("""
  NeurIPS 2026 - Reviewer Fix Re-Evaluation
  ==========================================
  Fix 1: Oracle justification (paper)
  Fix 2: exp(-AD/100) cohort-independent normalization (uqs.py)
  Fix 3: 3-feature CV MIA, 100 samples (metrics.py)
  Fix 4: Discriminative-only + per-dataset tau (findings.py)
  Fix 5: Scope reframed as methodology (paper)
""")

    verify_files()
    clear_eval_caches()
    run("py main.py --stage evaluate",
        "Re-evaluate all 36 models")
    run("py main.py --stage analyse",
        "Re-run analysis with discriminative tau")
    print_results()

    print("""
  DONE. Next steps:
    1. Copy outputs/figures/*.pdf  ->  neurips_paper/figures/
    2. Update Table 5 p-values in main.tex
    3. Update Table 3 tau matrix with new values
    4. Paste MMUBench-only tau into the paper paragraph
""")
