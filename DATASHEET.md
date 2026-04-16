# Datasheet for the Multimodal Unlearning Metric Consistency Benchmark

Following the Datasheets for Datasets framework (Gebru et al., 2021).

---

## Motivation

**For what purpose was the dataset created?**
This benchmark was created to systematically study *metric inconsistency* in multimodal machine unlearning. Existing unlearning benchmarks evaluate methods using multiple metrics independently but have not studied whether those metrics agree with each other. This benchmark provides evaluation results for 4 unlearning methods × 5 metrics × 3 datasets × 3 seeds on LLaVA-1.5-7B, together with analysis code and figures.

**Who created the dataset and on behalf of which entity?**
Created by the paper authors for the NeurIPS 2026 Datasets & Benchmarks Track submission.

**Who funded the creation of the dataset?**
University research funding. No commercial funding.

---

## Composition

**What does the dataset represent?**
The benchmark contains *evaluation results* (numerical metric scores), not raw images or text. The underlying visual-question-answering data comes from three existing, publicly licensed benchmarks:

| Source Dataset | License | HuggingFace | Reference |
|---|---|---|---|
| MLLMU-Bench | Apache 2.0 | franciscoliu/MLLMU-Bench | Liu et al., NAACL 2025 |
| UnLOK-VQA | CC-BY 4.0 | vpatil24/unlok-vqa | Patil et al., arXiv 2505.01456 |
| MMUBench | Apache 2.0 | linhx/MMUBench | Li et al., NeurIPS 2024 |

The **new contribution** of this dataset is the evaluation results, analysis scripts, and derived Unified Quality Score (UQS) weights.

**How many instances are there?**
- Multimodal results: 4 methods × 3 datasets × 3 seeds = 36 evaluation runs
- Unimodal results: 4 methods × 1 dataset × 3 seeds = 12 evaluation runs
- Total: 48 rows in `multimodal_results.json` + `unimodal_results.json`

**Does the dataset contain all possible instances, or is it a sample?**
It is a complete enumeration of all method × dataset × seed combinations defined in `config.py`.

**Is there a label or target associated with each instance?**
Each instance includes 5 metric scores (FA, RA, MIA, AD, JS) and a retrained model distance (ground truth).

**Is any information missing from individual instances?**
No. All 48 rows have all 6 numeric fields.

**Are relationships between instances made explicit?**
Yes. Each row includes `dataset`, `method`, and `seed` fields allowing grouping and comparison.

**Are there any errors, noise, or redundancies?**
- Metric values are computed programmatically from model checkpoints with fixed seeds. Numerical precision is float32.
- The same model is re-evaluated across datasets, introducing correlation within methods.

**Is the dataset self-contained, or does it link to external resources?**
The results JSON files are self-contained. Reproducing the results requires downloading LLaVA-1.5-7B from HuggingFace and the three source datasets. All links are provided in README.md.

**Does the dataset contain data that might be considered confidential?**
No. All source datasets are public. No personal or sensitive information is included.

**Does the dataset contain data that might be considered offensive or inappropriate?**
The source VQA datasets may contain general-domain images. We are not aware of offensive content in MLLMU-Bench, UnLOK-VQA, or MMUBench.

---

## Collection Process

**How was the data collected?**
Evaluation results were produced by running `main.py` with three fixed seeds (42, 123, 5508) on an NVIDIA RTX 4090 (24GB). All hyperparameters are in `config.py`. The full pipeline is reproducible from scratch with `python main.py --stage all`.

**Who collected the data?**
The paper authors using the code in this repository.

**Over what timeframe was the data collected?**
Experiments were run between January–April 2026 for the NeurIPS 2026 submission.

**Were any ethical review processes conducted?**
The work involves no human subjects, no personal data, and no sensitive applications. No ethics review was required.

---

## Preprocessing / Cleaning / Labelling

**Was any preprocessing / cleaning / labelling done?**
- Forget/retain splits are created by random sampling 10% of training data as the forget set (seed-controlled, see `data/loader.py`).
- Metric values are averaged over multiple batches and normalised where required.

**Is the software used to preprocess available?**
Yes. All preprocessing is in `data/loader.py`, `data/dataset.py`, and `evaluation/metrics.py`.

**Is the preprocessing reproducible?**
Yes, with fixed seeds via `utils.set_seed()`.

---

## Uses

**Has the dataset been used for any tasks already?**
It is introduced in the accompanying NeurIPS 2026 paper.

**What (other) tasks could the dataset be used for?**
- Studying whether other composite scoring approaches resolve metric inconsistency
- Benchmarking new unlearning methods in multimodal settings
- Studying how forget/retain split ratio affects metric agreement
- Extending to other VLMs (e.g. LLaVA-1.6, InstructBLIP)

**Is there anything about the composition of the dataset or the way it was collected and preprocessed that might affect future uses?**
- Results are specific to LLaVA-1.5-7B with LoRA. Other models may show different metric correlations.
- The forget/retain split is random (10%). Different ratios may affect metric agreement.
- Source datasets have specific coverage (factual VQA, knowledge unlearning). Results may not generalise to other domains.

**Are there tasks for which the dataset should not be used?**
This benchmark should not be used to make claims about LLaVA-1.5's safety or privacy without proper evaluation; metric scores measure unlearning quality under the specific evaluation protocol.

---

## Distribution

**How will the dataset be distributed?**
Via the GitHub repository accompanying the paper. Results JSONs and analysis code are in `outputs/` and `analysis/`.

**When will the dataset be distributed / is it already distributed?**
Upon paper acceptance. The code repository is public at submission time (single-blind submission).

**Will the dataset be distributed under a copyright or other IP license?**
Apache License 2.0. See `LICENSE`.

**Have any third parties imposed IP-based or other restrictions on the data?**
The underlying VQA datasets have their own licenses (see table above). Our derived evaluation results are Apache 2.0.

---

## Maintenance

**Who is supporting / hosting / maintaining the dataset?**
The paper authors. The repository will be maintained on GitHub.

**How can the owner / curator / manager of the dataset be contacted?**
Via GitHub Issues on the repository.

**Will the dataset be updated?**
We plan to add results for additional unlearning methods and VLMs as they become available. Version tags will be used.

**If the dataset is no longer supported, will there be an archival version?**
Results JSONs will be submitted to a persistent archive (Zenodo) upon camera-ready.

---

## References

- Gebru et al. (2021). "Datasheets for Datasets." *Communications of the ACM*.
- Liu et al. (2025). "MLLMU-Bench." NAACL 2025.
- Patil et al. (2025). "UnLOK-VQA." arXiv:2505.01456.
- Li et al. (2024). "MMUBench." NeurIPS 2024.
- Fan et al. (2024). "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation." ICLR 2024.
