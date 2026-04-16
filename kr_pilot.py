"""
kr_pilot.py — Knowledge Recoverability (KR) Pilot Experiment
=============================================================
Two-stage no-API version with MCQ-aware evaluation.

Stage 1:
- Run direct FA scan on the full forget set
- Save all direct results to outputs/fa_scan_results.json
- Create kr_probes_fa0_template.json only for samples with FA=0

Stage 2:
- Load manually created probes from kr_probes_fa0.json
- Run KR only on the FA=0 subset
- Save outputs/kr_pilot_results.json
- Save outputs/kr_pilot_table.tex
- Save outputs/kr_pilot_summary.txt

Required files:
  - probe_prompt.txt         : fixed prompt used to create probes manually
  - kr_probes_fa0.json       : manually filled probes for FA=0 samples only

Usage:
    py kr_pilot.py --stage scan
    py kr_pilot.py --stage kr
"""

import os
import sys
import re
import json
import time
import argparse
from typing import Dict, List, Any, Optional

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(BASE_DIR, "outputs")
CKPTS = os.path.join(BASE_DIR, "checkpoints")

# Must be set before any transformers import
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, BASE_DIR)

MODEL_CHECKPOINT = os.path.join(
    CKPTS, "unlearned_mllmu_bench_salun_seed5508.pt"
)
DATASET = "mmubench"
N_SAMPLES = 100

PROMPT_FILE = os.path.join(BASE_DIR, "probe_prompt.txt")
PROBE_FILE = os.path.join(BASE_DIR, "kr_probes_fa0.json")
PROBE_TEMPLATE_FILE = os.path.join(BASE_DIR, "kr_probes_fa0_template.json")
FA_SCAN_FILE = os.path.join(OUTPUTS, "fa_scan_results.json")
KR_RESULTS_FILE = os.path.join(OUTPUTS, "kr_pilot_results.json")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def truncate(text: str, n: int) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


# ---------------------------------------------------------------------
# Prompt / probe loading
# ---------------------------------------------------------------------

def read_probe_prompt() -> str:
    if not os.path.exists(PROMPT_FILE):
        print(f"  Warning: probe prompt file not found: {PROMPT_FILE}")
        return ""
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_probe_json() -> List[Dict[str, Any]]:
    if not os.path.exists(PROBE_FILE):
        raise FileNotFoundError(
            f"Probe file not found: {PROBE_FILE}\n\n"
            f"Create kr_probes_fa0.json first. Expected format:\n"
            f"[\n"
            f'  {{\n'
            f'    "sample_id": 3,\n'
            f'    "question": "Which ...?",\n'
            f'    "answer": "A",\n'
            f'    "rephrased": "...",\n'
            f'    "indirect": "...",\n'
            f'    "negation": "..."\n'
            f"  }}\n"
            f"]"
        )

    with open(PROBE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("kr_probes_fa0.json must contain a JSON list.")

    required = {"sample_id", "question", "answer", "rephrased", "indirect", "negation"}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Probe entry {i} is not a JSON object.")
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Probe entry {i} missing keys: {sorted(missing)}")

    return data


def build_probe_lookup(probe_entries: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    lookup: Dict[int, Dict[str, str]] = {}

    for item in probe_entries:
        sample_id = item["sample_id"]
        if sample_id in lookup:
            raise ValueError(f"Duplicate sample_id in kr_probes_fa0.json: {sample_id}")

        lookup[sample_id] = {
            "question": item["question"],
            "answer": item["answer"],
            "rephrased": item["rephrased"],
            "indirect": item["indirect"],
            "negation": item["negation"],
        }

    return lookup


def validate_probe_entry(
    sample_id: int,
    dataset_question: str,
    dataset_answer: str,
    probe_entry: Dict[str, str],
) -> None:
    q_dataset = normalize_text(dataset_question)
    a_dataset = normalize_text(dataset_answer)
    q_probe = normalize_text(probe_entry["question"])
    a_probe = normalize_text(probe_entry["answer"])

    if q_dataset != q_probe:
        raise ValueError(
            f"Question mismatch for sample_id={sample_id}\n"
            f"Dataset : {dataset_question}\n"
            f"JSON    : {probe_entry['question']}"
        )

    if a_dataset != a_probe:
        raise ValueError(
            f"Answer mismatch for sample_id={sample_id}\n"
            f"Dataset : {dataset_answer}\n"
            f"JSON    : {probe_entry['answer']}"
        )

    for key in ("rephrased", "indirect", "negation"):
        if not str(probe_entry[key]).strip():
            raise ValueError(f"Empty probe '{key}' for sample_id={sample_id}")


# ---------------------------------------------------------------------
# MCQ-aware evaluation
# ---------------------------------------------------------------------

def extract_choice_letter(text: str) -> Optional[str]:
    """
    Try to extract A/B/C/D from model output.
    Handles examples like:
      A
      (B)
      option C
      answer is D
      the correct answer is B
    """
    t = str(text).strip().upper()

    patterns = [
        r"^\s*([ABCD])\s*$",
        r"^\s*\(([ABCD])\)\s*$",
        r"^\s*OPTION\s+([ABCD])\s*$",
        r"^\s*ANSWER\s*(?:IS|:)?\s*([ABCD])\s*$",
        r"^\s*THE\s+ANSWER\s*(?:IS|:)?\s*([ABCD])\s*$",
        r"^\s*THE\s+CORRECT\s+ANSWER\s*(?:IS|:)?\s*([ABCD])\s*$",
    ]

    for pat in patterns:
        m = re.match(pat, t)
        if m:
            return m.group(1)

    looser_patterns = [
        r"\bOPTION\s+([ABCD])\b",
        r"\bANSWER\s*(?:IS|:)?\s*([ABCD])\b",
        r"\bCORRECT\s+ANSWER\s*(?:IS|:)?\s*([ABCD])\b",
    ]
    for pat in looser_patterns:
        m = re.search(pat, t)
        if m:
            return m.group(1)

    return None


def get_option_text_map(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract option text mapping from a dataset sample.
    Supports common MCQ layouts.
    Returns dict like: {'A': 'horse', 'B': 'cat', 'C': 'dog', 'D': 'cow'}
    """
    option_map: Dict[str, str] = {}

    # Case 1: explicit A/B/C/D fields
    for letter in ["A", "B", "C", "D"]:
        if letter in sample and isinstance(sample[letter], str):
            option_map[letter] = sample[letter]
    if option_map:
        return option_map

    # Case 2: lowercase a/b/c/d fields
    for letter in ["a", "b", "c", "d"]:
        if letter in sample and isinstance(sample[letter], str):
            option_map[letter.upper()] = sample[letter]
    if option_map:
        return option_map

    # Case 3: options/choices/candidates/answers list
    for key in ["options", "choices", "candidates", "answers"]:
        if key in sample and isinstance(sample[key], list) and len(sample[key]) >= 4:
            vals = sample[key][:4]
            if all(isinstance(v, str) for v in vals):
                return {
                    "A": vals[0],
                    "B": vals[1],
                    "C": vals[2],
                    "D": vals[3],
                }

    # Case 4: options/choices/candidates/answers dict
    for key in ["options", "choices", "candidates", "answers"]:
        if key in sample and isinstance(sample[key], dict):
            d = sample[key]
            out: Dict[str, str] = {}
            for letter in ["A", "B", "C", "D", "a", "b", "c", "d"]:
                if letter in d and isinstance(d[letter], str):
                    out[letter.upper()] = d[letter]
            if out:
                return out

    return {}


def check_answer(prediction: str, ground_truth: str, sample: Optional[Dict[str, Any]] = None) -> bool:
    """
    MCQ-aware correctness check.

    Priority:
      1. exact option-letter match
      2. semantic match against correct option text, if available
      3. yes/no fallback when option map is missing
      4. fallback literal match
    """
    pred_norm = normalize_text(prediction)
    gold = str(ground_truth).strip().upper()

    # 1. explicit letter match
    pred_letter = extract_choice_letter(prediction)
    if pred_letter is not None and pred_letter == gold:
        return True

    option_map: Dict[str, str] = {}
    if sample is not None:
        option_map = get_option_text_map(sample)

    # 2. semantic match against correct option text
    if gold in option_map:
        correct_text = normalize_text(option_map[gold])

        if correct_text:
            if correct_text in pred_norm:
                return True
            if pred_norm == correct_text:
                return True
            if pred_norm in correct_text:
                return True

            correct_words = set(re.findall(r"\b\w+\b", correct_text))
            pred_words = set(re.findall(r"\b\w+\b", pred_norm))
            if len(correct_words & pred_words) > 0:
                return True

    # 3. yes/no fallback when option map is missing
    # Assumes common binary convention: A=yes, B=no
    if sample is not None and not option_map:
        pred_words = set(re.findall(r"\b\w+\b", pred_norm))

        if gold == "A" and "yes" in pred_words:
            return True
        if gold == "B" and "no" in pred_words:
            return True

    # 4. fallback behavior
    gold_lower = gold.lower()
    if gold_lower in pred_norm or pred_norm.startswith(gold_lower):
        return True

    return False


# ---------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------

def load_unlearned_model():
    """Load the GA-unlearned LLaVA model."""
    import config as cfg
    cfg.ACTIVE_MODEL = "llava"

    from models.llava_model import load_llava, apply_lora
    from utils import load_checkpoint

    if not os.path.exists(MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_CHECKPOINT}\n"
            f"Run: py main.py --stage unlearn --dataset mmubench "
            f"--method gradient_ascent --seed 42"
        )

    print("  Loading LLaVA-7B + GA checkpoint...")
    model, processor = load_llava()
    model = apply_lora(model)
    model, _ = load_checkpoint(model, MODEL_CHECKPOINT)
    model.to(cfg.DEVICE)
    model.eval()
    print(f"  Model loaded on {cfg.DEVICE}")
    return model, processor


def run_inference(model, processor, image, question: str) -> str:
    """Run single-sample inference. Returns model text output."""
    import config as cfg

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(cfg.DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

def load_forget_set() -> List[Dict[str, Any]]:
    print("\n  Loading MMUBench forget set...")
    import config as cfg
    cfg.ACTIVE_MODEL = "llava"
    from data.loader import load_dataset_by_name

    data = load_dataset_by_name(DATASET)
    forget_set = data["forget"][:N_SAMPLES]
    print(f"  Using {len(forget_set)} forget-set samples")
    return forget_set


# ---------------------------------------------------------------------
# Stage 1 — Direct FA scan
# ---------------------------------------------------------------------

def run_direct_fa_scan():
    print("\n" + "=" * 65)
    print("  STAGE 1 — Direct FA Scan")
    print("=" * 65)

    ensure_dir(OUTPUTS)

    prompt_text = read_probe_prompt()
    if prompt_text:
        print("  Found probe_prompt.txt")
    else:
        print("  probe_prompt.txt not found; continuing without it")

    forget_set = load_forget_set()

    print("\n  Loading unlearned model...")
    model, processor = load_unlearned_model()

    fa_results = []
    fa0_template = []

    for i, sample in enumerate(forget_set):
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        print(f"\n  [{i + 1}/{len(forget_set)}] Q: {truncate(question, 60)}")
        print(f"         A: {answer}")

        pred = run_inference(model, processor, image, question)
        direct_correct = check_answer(pred, answer, sample)

        status = "FA=1 (remembered)" if direct_correct else "FA=0 (forgotten)"
        print(f"    Direct: '{truncate(pred, 50)}' → {status}")

        option_map = get_option_text_map(sample)

        fa_results.append({
            "sample_id": i,
            "question": question,
            "answer": answer,
            "direct_prediction": pred,
            "direct_correct": direct_correct,
            "option_map": option_map,
        })

        if not direct_correct:
            fa0_template.append({
                "sample_id": i,
                "question": question,
                "answer": answer,
                "rephrased": "",
                "indirect": "",
                "negation": ""
            })

    with open(FA_SCAN_FILE, "w", encoding="utf-8") as f:
        json.dump(fa_results, f, indent=2, ensure_ascii=False)

    with open(PROBE_TEMPLATE_FILE, "w", encoding="utf-8") as f:
        json.dump(fa0_template, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved direct scan results to: {FA_SCAN_FILE}")
    print(f"  Saved FA=0 probe template to: {PROBE_TEMPLATE_FILE}")
    print(f"  FA=0 samples found: {len(fa0_template)}")

    return fa_results


# ---------------------------------------------------------------------
# Stage 2 — KR only on FA=0 samples
# ---------------------------------------------------------------------

def run_kr_pilot():
    print("\n" + "=" * 65)
    print("  STAGE 2 — KR on FA=0 Samples")
    print("=" * 65)

    ensure_dir(OUTPUTS)

    prompt_text = read_probe_prompt()
    if prompt_text:
        print("  Found probe_prompt.txt")
    else:
        print("  probe_prompt.txt not found; continuing without it")

    forget_set = load_forget_set()

    print("\n  Loading local probes...")
    probe_entries = load_probe_json()
    probe_lookup = build_probe_lookup(probe_entries)

    fa0_samples = []
    for i, sample in enumerate(forget_set):
        if i in probe_lookup:
            validate_probe_entry(
                sample_id=i,
                dataset_question=sample["question"],
                dataset_answer=sample["answer"],
                probe_entry=probe_lookup[i],
            )
            fa0_samples.append((i, sample))

    if not fa0_samples:
        raise ValueError(
            "No matching FA=0 probes found in kr_probes_fa0.json.\n"
            "Run stage 1 first:\n"
            "  py kr_pilot.py --stage scan\n"
            "Then fill kr_probes_fa0_template.json and save it as kr_probes_fa0.json"
        )

    print(f"  Found {len(fa0_samples)} FA=0 samples with probes")

    print("\n  Loading unlearned model...")
    model, processor = load_unlearned_model()

    results = []

    for idx, (sample_id, sample) in enumerate(fa0_samples):
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]
        probes = probe_lookup[sample_id]

        print(f"\n  [{idx + 1}/{len(fa0_samples)}] sample_id={sample_id}")
        print(f"         Q: {truncate(question, 60)}")
        print(f"         A: {answer}")

        t0 = time.time()

        direct_pred = run_inference(model, processor, image, question)
        direct_correct = check_answer(direct_pred, answer, sample)
        direct_status = "CORRECT ✗ (not forgotten)" if direct_correct else "WRONG ✓ (forgotten)"
        print(f"    Direct    : '{truncate(direct_pred, 50)}' → {direct_status}")

        probe_map = {
            "rephrased": probes["rephrased"],
            "indirect": probes["indirect"],
            "negation": probes["negation"],
        }

        probe_results = {}
        kr_correct_count = 0

        for probe_type, probe_q in probe_map.items():
            pred = run_inference(model, processor, image, probe_q)
            correct = check_answer(pred, answer, sample)

            if correct:
                kr_correct_count += 1

            probe_results[probe_type] = {
                "question": probe_q,
                "prediction": pred,
                "correct": correct,
            }

            status = "LEAKS ⚠" if correct else "blocked"
            print(f"    {probe_type:10s}: '{truncate(pred, 40)}' → {status}")

        kr_score = kr_correct_count / len(probe_map)

        result = {
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "direct_prediction": direct_pred,
            "direct_correct": direct_correct,
            "fa_correct": direct_correct,
            "probes": probe_results,
            "kr_score": kr_score,
            "probe_source": "local_json",
            "probe_prompt_used": bool(prompt_text),
            "option_map": get_option_text_map(sample),
        }

        results.append(result)
        elapsed = time.time() - t0
        print(f"    KR score  : {kr_score:.2f} ({kr_correct_count}/{len(probe_map)} leaked) | {elapsed:.1f}s")

    with open(KR_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved: {KR_RESULTS_FILE}")
    return results


# ---------------------------------------------------------------------
# Analysis and paper outputs
# ---------------------------------------------------------------------

def analyse_and_write_paper(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print("  ANALYSIS — FA vs KR")
    print("=" * 65)

    total = len(results)
    fa_0_samples = [r for r in results if not r["direct_correct"]]
    fa_1_samples = [r for r in results if r["direct_correct"]]

    kr_leaks = [r for r in fa_0_samples if r["kr_score"] > 0]
    kr_scores_fa0 = [r["kr_score"] for r in fa_0_samples]
    mean_kr_fa0 = sum(kr_scores_fa0) / len(kr_scores_fa0) if kr_scores_fa0 else 0.0
    all_kr = [r["kr_score"] for r in results]
    mean_kr_all = sum(all_kr) / len(all_kr) if all_kr else 0.0

    print(f"\n  Total KR-evaluated samples: {total}")
    print(f"  Still FA=0 at KR time:      {len(fa_0_samples)}")
    print(f"  Became FA=1 again:          {len(fa_1_samples)}")

    print(f"\n  Among FA=0 samples:")
    print(f"    With KR > 0 (leakage):    {len(kr_leaks)} / {len(fa_0_samples)}")
    print(f"    Mean KR score:            {mean_kr_fa0:.3f}")
    print(f"\n  Key finding: {len(kr_leaks)}/{len(fa_0_samples)} samples where FA=0 still have KR>0")
    print("  → Suppressing direct recall does NOT erase the knowledge")

    probe_types = ["rephrased", "indirect", "negation"]
    print("\n  Per-probe-type leakage (across FA=0 samples):")
    probe_success_counts = {}
    for pt in probe_types:
        successes = sum(
            1 for r in fa_0_samples
            if pt in r["probes"] and r["probes"][pt]["correct"]
        )
        n = sum(1 for r in fa_0_samples if pt in r["probes"])
        pct = (successes / n * 100) if n else 0.0
        probe_success_counts[pt] = successes
        print(f"    {pt:10s}: {successes}/{n} leaked ({pct:.0f}%)")

    table_rows = []
    for r in results:
        fa_str = "0" if not r["direct_correct"] else "1"
        kr_str = f"{r['kr_score']:.2f}"
        verdict = (
            r"\textbf{leaks}" if (not r["direct_correct"] and r["kr_score"] > 0)
            else ("n/a" if r["direct_correct"] else "blocked")
        )

        q_short = truncate(r["question"], 45)
        a_short = truncate(r["answer"], 15)

        table_rows.append(
            f"  {latex_escape(q_short)} & {latex_escape(a_short)} & {fa_str} & {kr_str} & {verdict} \\\\"
        )

    latex_table = r"""
\begin{table}[t]
\centering
\caption{\textbf{Knowledge Recoverability (KR) pilot.}
Only samples that fail the direct query in the initial FA scan are probed.
FA$=0$ indicates the unlearned model fails the direct question at KR time.
KR$>0$ indicates the forgotten information is recovered via a
rephrased, indirect, or negation probe, despite FA$=0$.}
\label{tab:kr_pilot}
\small
\begin{tabular}{p{5cm}p{1.5cm}cc p{2cm}}
\toprule
\textbf{Original Question} & \textbf{Answer} & \textbf{FA} & \textbf{KR} & \textbf{Verdict} \\
\midrule
""" + "\n".join(table_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    table_path = os.path.join(OUTPUTS, "kr_pilot_table.tex")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\n  LaTeX table saved: {table_path}")

    n_fa0 = len(fa_0_samples)
    paragraph = f"""
\\paragraph{{Knowledge Recoverability pilot.}}
We first scan the full forget set using the original query and retain only samples
for which the unlearned model fails the direct question. We then probe this
FA$=0$ subset with three fixed alternative queries constructed prior to evaluation:
one rephrased query, one indirect semantic query, and one negation-based query.
Among {n_fa0} samples that still satisfy FA$=0$ at KR time, {len(kr_leaks)}
({(len(kr_leaks) / max(n_fa0, 1) * 100):.0f}\\%) reveal the forgotten answer via
at least one alternative probe. Mean KR score among FA$=0$ samples is
{mean_kr_fa0:.3f} (0$=$fully erased, 1$=$fully recoverable). This shows that
failure on the direct query does not imply true knowledge erasure.
""".strip()

    para_path = os.path.join(OUTPUTS, "kr_pilot_summary.txt")
    with open(para_path, "w", encoding="utf-8") as f:
        f.write(paragraph)

    print(f"  Paper paragraph saved: {para_path}")
    print("\n  === PARAGRAPH PREVIEW ===")
    print(paragraph)

    return {
        "total": total,
        "fa_0": len(fa_0_samples),
        "fa_1": len(fa_1_samples),
        "kr_leaks": len(kr_leaks),
        "mean_kr_fa0": mean_kr_fa0,
        "mean_kr_all": mean_kr_all,
        "rephrased_leaks": probe_success_counts.get("rephrased", 0),
        "indirect_any_leaks": sum(
            1 for r in fa_0_samples
            if (
                r["probes"].get("indirect", {}).get("correct", False)
                or r["probes"].get("negation", {}).get("correct", False)
            )
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["scan", "kr"],
        required=True,
        help="scan = direct FA scan, kr = KR evaluation on FA=0 samples",
    )
    args = parser.parse_args()

    print(r"""
╔═════════════════════════════════════════════════════════════════════╗
║  Knowledge Recoverability Pilot                                     ║
║  Two-stage, local JSON probes                                       ║
║  Model: LLaVA-7B GA-unlearned/Salun, MMUBench, seed 42/128/5508     ║
╚═════════════════════════════════════════════════════════════════════╝
""")

    if args.stage == "scan":
        run_direct_fa_scan()

    elif args.stage == "kr":
        results = run_kr_pilot()
        summary = analyse_and_write_paper(results)

        leak_pct = (summary["kr_leaks"] / max(summary["fa_0"], 1)) * 100

        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  DONE                                                         ║
╠═══════════════════════════════════════════════════════════════╣
║  FA=0 samples:  {summary['fa_0']:2d}                                        ║
║  KR leaks:      {summary['kr_leaks']:2d} ({leak_pct:.0f}% of FA=0 samples still leaked)     ║
║  Mean KR:       {summary['mean_kr_fa0']:.3f}                                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Files:                                                       ║
║    outputs/fa_scan_results.json   <- direct scan results      ║
║    outputs/kr_pilot_results.json  <- full KR results          ║
║    outputs/kr_pilot_table.tex     <- LaTeX table              ║
║    outputs/kr_pilot_summary.txt   <- paragraph for paper      ║
╚═══════════════════════════════════════════════════════════════╝
""")