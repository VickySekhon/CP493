## Twelve-Week Plan for “GPT Log-Probabilities Predict OCR Quality”

This plan assumes one term (12 weeks) of undergraduate research on the BLN600 newspaper corpus, using GPT-4o logprobs and the existing `scan2latex_entropy.py` prototype. By Week 12, the student should have:

* A reproducible pipeline (`loader.py`, `logprobs_client.py`, `entropy.py`, `predict_quality.py`, figure scripts).
* A clean artifact bundle (page-level CSVs, cached API outputs, plots).
* Draft manuscript sections (literature review, experimental setup, results) extending the CCWC short paper to a journal paper.

Throughout, we **only** use the GPT-4o API’s `logprobs` / `top_logprobs` signal—no empirical probability estimation, no semantic entropy, no multiple stochastic passes.

---

### Before Week 1: High-Level Context (Student Homework)

* Skim the BLN600 paper and dataset description to understand what data we have: 600 19th-century London crime newspaper excerpts with images, machine OCR, and manual transcriptions. ([White Rose Research Online][1])
* Read the OCR-D CER spec (definition as normalized Levenshtein). ([ocr-d.de][2])
* Skim an approachable CER overview to build intuition. ([WaterCrawl][3])
* Glance at OpenAI/logprobs docs to see what the API returns when `logprobs=True`, `top_logprobs=k`. ([OpenAI Cookbook][4])

---

## Week 1 – Orientation, Constraints, and Script Forensics

**Theory**

* Carefully read the syllabus and the CCWC short paper draft (Introduction + Background + Method sections).

* Discuss the conceptual pipeline:

  1. Historical newspaper page image → GPT-4o transcript + token-level top-k logprobs.
  2. Top-k+tail Shannon entropy per token (in **bits**).
  3. Aggregate to average bits per token per page.
  4. Compare with CER / Levenshtein distance vs GT.
  5. Use entropy as a **triage** score (accept vs review).

* Make explicit the constraints:

  * **Single pass only:** one GPT-4o decoding run per page.
  * **Native logprobs only:** no empirical probability estimation from multiple samples.
  * **No semantic entropy:** no “rephrase N times and measure disagreement.”

**Python / Code**

* Create a **private repo** (e.g., GitHub) for the project.

  * Add `README.md`, `LICENSE` (to be finalized before artifact release), and basic folder structure:
    `src/`, `notebooks/`, `data/`, `cache/`, `figures/`.

* Set up a Python virtual environment and pin dependencies:

  ```text
  pandas, numpy, rapidfuzz[levenshtein], matplotlib, seaborn (optional),
  scikit-learn, python-dotenv (or similar), openai
  ```

* Copy `scan2latex_entropy.py` into the repo under `scripts/` and perform a **code-reading pass**:

  * Confirm **how Shannon entropy is computed**:

    * For each token, loop over `info.top_logprobs[:TOP_K]`.
    * Convert logprobs (natural logs) to probabilities via `p = exp(logprob)`.
    * Accumulate entropy in *bits* using
      `H_pos += -p * alt.logprob / LOG2` where `LOG2 = ln(2)`
      → this is (-p \log_2 p).
    * Sum probability mass into `mass`, define tail as `p_tail = max(0, 1-mass)` and add
      (-p_{\text{tail}}\log_2 p_{\text{tail}}) when `p_tail>0`.
    * No renormalisation beyond this; tail is a single coarse-grained state.
  * Note **defaults**:

    * `TOP_K = 5` (via CLI `--top-k`, default 5).
    * `window-size W = 20` tokens (CLI default).
    * `TOP_M = 3` windows reported.
  * Confirm **sliding-window aggregation**:

    * Compute running sum of per-token entropies with stride 1.
    * For each contiguous window of length `W`, store `(avg_entropy, start_index)`.
    * Use `heapq.nlargest(TOP_M, windows)` to get the top-M highest-entropy windows.
    * This is **mean** entropy per window; no max/sum variant.
  * Note token filtering:

    * Tokens in `EXCLUDE_TOKENS` (`"```", "python", "", " ", "\n", "latex", "json", "tag"`) and those with `token.strip()==""` are dropped.
  * Note GPT-4o call configuration:

    * `temperature=0.5`, `top_p=0.9`, `seed=12345`, `max_tokens=10000`, `logprobs=True`, `top_logprobs=TOP_K`.

**Deliverables**

* Short **one-page “code autopsy” note** summarising:

  * Entropy formula, units (bits), top-k+tail design.
  * Sliding-window details (window size, stride, mean aggregation).
  * Current default hyperparameters and tokens excluded.

---

## Week 2 – BLN600 Ingestion and CER/Levenshtein Baselines

**Theory**

* Deep read the BLN600 paper & dataset description: understand file layout, what “excerpt” means (image + machine OCR + manual transcription). ([White Rose Research Online][1])
* Study CER formally from OCR-D: CER is Levenshtein edit distance over characters divided by GT string length. ([ocr-d.de][2])
* Discuss pitfalls:

  * Whitespace and line breaks.
  * Punctuation and curly vs straight quotes, dashes.
  * Page length effects: same absolute distance gives very different CER for short vs long pages.

**Python / Code**

* Implement `loader.py` with (at minimum):

  * `load_bln600_metadata(...)` → DataFrame with one row per excerpt/page, including IDs, file paths for images, machine OCR text, and human GT text.
  * `load_text_pair(page_id)` → `(gt_text, machine_ocr_text)` with consistent normalization.

* Implement `normalization.py`:

  * Functions for **configurable** normalization (each as a toggle):

    ```python
    def normalize_whitespace(text): ...
    def normalize_quotes_and_dashes(text): ...
    def strip_punctuation(text): ...
    def lowercase(text): ...
    ```

  * A master function that applies a chosen pipeline (e.g., whitespace + quote/dash normalization, but not punctuation stripping).

* Implement `metrics.py`:

  * `levenshtein_distance(a, b)` using `rapidfuzz` or similar.
  * `cer(a, b)` = `levenshtein_distance / len(gt)` with careful handling of empty strings.

* Run **baseline evaluation** on 10–20 BLN600 pages (machine OCR vs GT):

  * Compute length, Levenshtein, CER with your chosen normalization pipeline.
  * Simple histograms of CER and length.

**Writing**

* Start a **“Related Work” bullet outline**:

  * OCR evaluation metrics and CER/WER.
  * OCR confidence measures and post-OCR correction.
  * Logprobs and uncertainty in LLMs / VLMs.

**Deliverables**

* Jupyter notebook or script `notebooks/01_bln600_baseline.ipynb` (or `.py`) demonstrating:

  * Loading ~20 pages.
  * Normalisation examples (before/after text).
  * CER/Levenshtein distributions (simple plots).

---

## Week 3 – GPT-4o Logprobs Client for Newspaper Pages

**Theory**

* Revisit the Method section of the conference paper, especially the top-k+tail entropy definition and the coarse-graining lower-bound property.

* Connect to API behaviour:

  * From OpenAI docs / cookbook: `logprobs=True` and `top_logprobs=k` return logprobs for the chosen token plus up to `k` most likely alternatives. ([OpenAI Cookbook][4])
  * Interpret logprob values as natural logs of probabilities.

* Clarify **surprisal vs entropy**:

  * Surprisal: (-\log_2 P(y_i)) of the chosen token (top-1).
  * Entropy: expectation of surprisal over the distribution ((H_i = \mathbb{E}[-\log_2 P(Y_i)])).
  * Our method uses **entropy**, approximated via top-k+tail; surprisal is optional ablation later.

**Python / Code**

* Create `logprobs_client.py`:

  * Function `transcribe_with_logprobs(image_path, *, top_k=5, model="gpt-4o")` that:

    1. Encodes the page image to base64 (reusing `encode_image`).

    2. Uses a **new system/user prompt** appropriate for **newspaper text**, not LaTeX:

       * System: “You are an OCR engine for historical English newspapers. Transcribe the page faithfully as plain text…”
       * User: specific instructions about preserving layout minimally and not using code fences.

    3. Calls GPT-4o with the same randomness settings as the script (`temperature`, `top_p`, `seed`), plus `logprobs=True`, `top_logprobs=top_k`.

    4. Returns `(transcript_text, token_logprobs)` where `token_logprobs` is a list/dict capturing for each token:

       ```python
       {
         "token": str,
         "logprobs": [float ln p_j for top-k],
         "alts": [{"token": str, "logprob": float}, ...]
       }
       ```

* Implement simple **JSON caching**:

  * Cache key: `(page_id, model, top_k, prompt_version)` → JSON with transcript + token logprobs.
  * Before calling the API, check if a cache file exists; if yes, load it instead of calling GPT-4o.

* Run on **3–5 BLN600 pages** as a dry run:

  * Add a short script / notebook showing:

    * Example transcript snippet.
    * First 5 tokens with their top-k alternatives and probabilities.

**Deliverables**

* `src/logprobs_client.py` with docstrings and error handling.
* A short **“API logprobs” note** summarizing:

  * Data structure returned.
  * How many tokens per page.
  * Any quirks observed (e.g., unusual tokens, whitespace).

---

## Week 4 – Clean Entropy Module and Verification Against Prototype

**Theory**

* Work through the entropy equations in the conference paper:

  * Token-level top-k+tail entropy in bits:
    [
    \widehat H_i =
    -\sum_{j=1}^k p_{i,j}\log_2 p_{i,j}
    -p_{i,\text{tail}}\log_2 p_{i,\text{tail}},
    \quad p_{i,\text{tail}} = 1 - \sum_{j=1}^k p_{i,j}.
    ]
  * Page-level totals:
    [
    H_{\text{page}} = \sum_{i=1}^n \widehat H_i,
    \quad
    \overline H_{\text{page}} = \frac{1}{n} \sum_{i=1}^n \widehat H_i.
    ]
* Prove the **coarse-graining lower-bound property** (sketch):

  * Show that merging tail tokens into a single state cannot increase entropy; reference the standard result from information theory (entropy is Schur-concave). ([ResearchGate][5])

**Python / Code**

* Implement `entropy.py`:

  * `topk_tail_entropy(ln_probs: list[float]) -> float`:

    * Assume `ln_probs` are natural logs of top-k probabilities for a single position.
    * Compute `p_j = exp(lnp)`, `mass = sum(p_j)`, `p_tail = max(0.0, 1.0 - mass)`.
    * `H = sum(-p_j * lnp / LOG2)` + tail term (if `p_tail>0`).
    * Return `H` in **bits**.

  * `token_entropies_from_logprobs(token_logprobs, top_k)`:

    * Accept the cached GPT-4o logprobs structure.
    * Handle **token filtering** similar to `EXCLUDE_TOKENS` in the script.
    * Return list of entropies (one per kept token).

  * Optional: `sliding_window_entropies(pos_entropies, window_size, stride=1)` returning list of `(avg_entropy, start_idx)`.

* **Verification against `scan2latex_entropy.py`:**

  * For a small synthetic example, manually construct a pseudo-`logprobs` structure and verify:

    * `topk_tail_entropy` returns 0 bits when probability mass is 1 on a single token.
    * For two tokens with equal probability, returns 1 bit.
  * On a real cached page (from Week 3), run both:

    * A small wrapper that mimics the original script.
    * Your new module.
    * Confirm that **total bits** and **average bits/token** match up to floating-point tolerance.

**Writing**

* Add a **Methods subsection draft** describing:

  * How logprobs are requested from GPT-4o.
  * The top-k+tail entropy computation (formula, bits, lower-bound explanation).
  * Statement that the implementation matches a validated prototype.

**Deliverables**

* `src/entropy.py` with unit tests (e.g., `tests/test_entropy.py`).
* Short verification report (table with prototype vs new module results for 2–3 pages).

---

## Week 5 – End-to-End Pilot: BLN600 Subset

**Theory**

* Discuss aggregation and correlation:

  * Why use **average bits per token** vs total bits or median.
  * Pearson vs Spearman correlation; how each captures linear vs monotone relationships.
  * Bootstrapped confidence intervals (conceptual overview).

**Python / Code**

* Implement `predict_quality.py` (first draft) that, for a **subset** of BLN600 (e.g., 100 pages):

  1. Loads GT text and image path with `loader.py`.
  2. Uses `logprobs_client.transcribe_with_logprobs` (with `top_k=5`) to obtain GPT-4o transcript and logprobs (using cache).
  3. Uses `entropy.token_entropies_from_logprobs` → list of per-token entropies.
  4. Computes:

     * `avg_bits_per_token` (main predictor)
     * `total_bits`
     * `n_tokens`
  5. Computes CER and Levenshtein distance between **GPT-4o transcript** and GT using `metrics.py` with your chosen normalization pipeline.
  6. Stores a row per page into a DataFrame and writes `results_subset.csv` with columns:

     ```text
     page_id, avg_bits_per_token, total_bits, n_tokens,
     cer, levenshtein, gt_length, normalization_profile
     ```

* Visualisation (pilot):

  * Scatter plot: CER vs `avg_bits_per_token` with a simple loess or linear fit.
  * Histogram: distribution of `avg_bits_per_token`.
  * Optionally, scatter of Levenshtein vs `avg_bits_per_token`.

* Simple quantitative summary:

  * Compute Pearson `r` and Spearman `ρ` for CER vs average bits/token.
  * Use basic bootstrap (resample pages with replacement) to estimate 95% CIs for `r` (single script, no need for a full stats framework).

**Writing**

* Write a **short “preliminary results” paragraph** describing:

  * Shape of the scatter.
  * Direction and approximate strength of correlation.
  * Any visible outliers.

**Deliverables**

* `src/predict_quality.py` (subset mode).
* `figures/entropy_vs_cer_subset.png`.
* `results_subset.csv`.

**Milestone:** end-to-end path from BLN600 page → GPT-4o logprobs → entropy → CER for ~100 pages.

---

## Week 6 – Full-Corpus Run and Robustness Across (k)

**Theory**

* Examine how the top-k+tail approximation depends on **k**:

  * When top-k captures most probability mass, tail is small and approximation is tight.
  * When tail is large, approximation is coarser, but still a lower bound.

* Discuss computational trade-offs:

  * Larger `k` returns more logprobs per token (slightly more data), but the GPT-4o call cost is dominated by tokens, not by `k`.

**Python / Code**

* Extend `predict_quality.py` to support **configurable k** and full-corpus processing:

  * CLI arguments: `--top-k`, `--max-pages` (for debugging), `--output`.
  * The cache keys must include `top_k` to avoid mixing runs.

* Run experiments on **all available BLN600 pages** (or as many as budget allows) for:

  * `k = 5`
  * `k = 10`
  * `k = 15`

  For each k, produce a `results_k{K}.csv`.

* For each k:

  * Compute Pearson and Spearman correlations (CER vs avg bits/token).
  * Compute bootstrapped CIs for those correlations.
  * Compute simple linear regression slopes (CER ~ avg_bits/token).

* Build a **robustness table** (for the eventual paper):

  | k  | Pearson r (CER) | 95% CI | Spearman ρ (CER) | 95% CI |
  | -- | --------------- | ------ | ---------------- | ------ |
  | 5  |                 |        |                  |        |
  | 10 |                 |        |                  |        |
  | 15 |                 |        |                  |        |

* Visualisations:

  * Overlaid scatter plots (or density plots) for different k values.
  * Plot correlation coefficients vs k.

**Writing**

* Draft a short **“Ablation: sensitivity to k”** subsection:

  * Note whether correlations are stable across k.
  * Interpret any changes (e.g., diminishing returns beyond k=10).

**Deliverables**

* Full-corpus CSVs: `results_k5.csv`, `results_k10.csv`, `results_k15.csv`.
* `figures/entropy_vs_cer_k_comparison.png`.
* A filled robustness table (values to be added by the student).

**Milestone (Midterm):** complete results package for Weeks 1–6, ready for midterm checkpoint.

---

## Week 7 – Calibration and Entropy-Based Triage Thresholds

**Theory**

* Introduce **binary classification** on pages:

  * Define a `good_page` label: e.g., CER ≤ 1% (primary) and CER ≤ 2% (secondary sensitivity analysis).
  * Discuss AUROC, sensitivity, specificity, and trade-offs.
  * Explain **Youden’s J statistic** and how to choose an operating point.

**Python / Code**

* Use one `results_kX.csv` (pick the “default k”, e.g., 10) to build a calibration dataset:

  * Split pages into **train/validation** sets (e.g., 80/20, fixed random seed).

  * On the train set, fit a simple **logistic regression**:

    ```python
    good_page ~ avg_bits_per_token
    ```

  * On the validation set:

    * Compute predicted probabilities `p_hat`.
    * Compute AUROC for `good_page`.
    * Compute ROC curve and PR curve.
    * Compute a few candidate thresholds in entropy:

      * Threshold minimizing misclassification error.
      * Threshold maximizing Youden’s J.
      * Optionally, threshold that fixes a desired false-negative rate.

* Construct:

  * ROC plot (entropy-based classifier).
  * A small table of operating points with:

    | Threshold (bits/token) | Sensitivity | Specificity | AUROC (global) |

* Check whether using **CER threshold 1% vs 2%** changes the recommended entropy threshold substantially.

**Writing**

* Draft a **“Calibration and triage”** subsection:

  * Describe procedure (train/validation split, logistic model, metrics).
  * Report AUROC and chosen operating threshold(s).
  * Emphasise the **triage role**: high-entropy pages → send to manual review/post-correction.

**Deliverables**

* `notebooks/02_calibration_and_roc.ipynb`.
* `figures/roc_entropy_good_page.png`.
* Short text summary of recommended entropy threshold(s).

**Milestone:** Figure for ROC + chosen threshold, ready to become “Figure 2” in the journal paper.

---

## Week 8 – Error Analysis, Stratification, and Optional Surprisal Ablation

**Theory**

* Discuss **error analysis** in OCR:

  * How page length and layout complexity can confound the entropy–CER relationship.
  * Why we should inspect cases where entropy is high but CER is low, and vice versa.

* Explain **stratification**:

  * Group pages by length or CER ranges, look for patterns within strata.

**Python / Code**

* Extend results tables to include more features:

  * `page_length_chars` (len of GT text).
  * `page_length_tokens` (n_tokens).
  * A simple layout proxy if available (columns, etc.; optional).

* Stratified analysis:

  * Split pages into quartiles by `page_length_chars`.
  * For each quartile, recompute correlations (CER vs entropy) and plot separate scatters.
  * Check whether correlation weakens/strengthens for short vs long pages.

* Normalisation toggles:

  * Recompute CER under a different normalization (e.g., stripping punctuation).
  * Compare correlations under each normalization, focusing on whether the entropy signal is robust.

* **Optional surprisal ablation (must respect constraints):**

  * Using the existing `top_logprobs`, compute per-token **surprisal** for the actually emitted GPT-4o token whenever its logprob is present in the returned `top_logprobs` list:

    [
    s_i = -\log_2 p(y_i)
    ]

  * Aggregate to average surprisal per token per page.

  * Compare:

    * Correlation of CER vs mean entropy vs mean surprisal.
    * ROC performance using surprisal-based predictor vs entropy.

  * This uses only the same logprob data—no extra samples or semantic entropy.

* Visualisations:

  * Residual plots: CER – f(entropy) vs page length.
  * Scatter: entropy vs surprisal, mark misclassified triage decisions.

**Writing**

* Expand the **Results section** with:

  * Stratified plots / tables summarizing length effects.
  * A paragraph on robustness to normalization choices.
  * If surprisal ablation is done: short discussion comparing entropy vs surprisal as predictors.

**Deliverables**

* Additional figures (e.g., `figures/entropy_vs_cer_by_length.png`, `figures/surprisal_vs_entropy.png`).
* Updated results CSVs with extra columns.

---

## Week 9 – Reproducible Packaging and One-Command Pipeline

**Theory**

* Discuss **reproducible research**: data cards, environment specs, one-command rebuilds, seeds, and caching.
* Look briefly at other OCR/post-OCR work using BLN600 (e.g., BART-based post-OCR correction releases) to understand community expectations for artifact quality. ([arXiv][6])

**Python / Code**

* Finalise `predict_quality.py` as the **main CLI**:

  * Accepts arguments: dataset root, output directory, top_k, model, normalization profile, and page subset filters.
  * Handles caching robustly (no accidental overwrites).
  * Logs progress and basic stats.

* Add a **Makefile** or `tox`/`nox` config enabling:

  * `make run-all` → runs GPT-4o transcription (if needed), entropy computation, and evaluation, producing final CSVs.
  * `make figs` → runs figure-generation scripts/notebooks and writes all plots into `figures/`.
  * `make check` → runs unit tests.

* Consolidate **figure generation** into 1–2 scripts / notebooks:

  * `figures/01_entropy_vs_cer.py`
  * `figures/02_roc_thresholds.py`
  * `figures/03_stratified_analysis.py`

* Freeze environment:

  * `requirements.txt` generated from venv (curated, not raw `pip freeze` if possible).
  * Optional: `environment.yml` for Conda.

**Writing**

* Finalise the **Experimental Setup** section:

  * Data description (BLN600).
  * GPT-4o settings (model name, temperature, top_p, top_k, logprobs).
  * Entropy computation, CER definition, normalization pipeline.
  * Train/validation split details and seeds.
  * Hardware / runtime notes.

**Deliverables**

* Working one-command rebuild of all results and figures.
* Draft of Experimental Setup suitable for journal submission.

**Milestone:** “Artifact complete” from a reproducibility standpoint.

---

## Week 10 – Literature Review Section (Student-Owned)

**Theory**

* Systematically collect and summarise relevant literature:

  * OCR evaluation metrics & error analysis, with focus on CER. ([ocr-d.de][2])
  * Historical OCR and BLN600 use cases. ([White Rose Research Online][1])
  * VLM/LLM-based OCR and post-OCR correction on newspapers. ([Gale Review][7])
  * Logprobs and uncertainty in LLMs/VLMs (classification, calibration, etc.). ([Together.ai Docs][8])

* Map how the current project fits:

  * Page-level, scalar uncertainty based on GPT-4o logprobs.
  * Black-box use of GPT-4o as OCR engine.

**Writing**

* Draft a **2–3 page Literature Review** section structured along:

  1. OCR evaluation and CER.
  2. Historical OCR for newspapers and BLN600 context.
  3. Uncertainty in LLMs/VLMs and logprob-based metrics.
  4. Positioning of this work (page-level entropy triage).

* Make sure to:

  * Cite BLN600 paper properly (dataset and its uses).
  * Cite at least one OCR-metric survey paper.
  * Cite several recent logprob/uncertainty papers/tools.

**Python / Code (light)**

* Clean up figure styles (font sizes, labels, legends) to be publication-ready.
* Ensure all figures referenced in the text are generated from scripts, not manually.

**Deliverables**

* Draft Literature Review section (LaTeX or Markdown as per supervisor’s workflow).
* Updated, publication-style figures.

---

## Week 11 – Experiments, Results, and Limitations (Student-Owned)

**Theory**

* Discuss **best practices** in reporting experimental results:

  * Clear description of metrics, datasets, and splits.
  * Avoid over-claiming; focus on monotone trends and practical interpretability.
  * Explicitly state limitations (e.g. dataset, model dependence, absence of localisation).

**Writing**

* Draft **Experimental Results** section:

  * **Quantitative:** correlation tables, robustness across k, ROC metrics, stratified analyses.
  * **Qualitative:** description of representative low-entropy/low-CER and high-entropy/high-CER pages, plus interesting outliers.
  * **Triage rule:** present the chosen entropy threshold(s) and discuss operational meaning (e.g., expected proportion of pages flagged).

* Draft a concise **Limitations** subsection (likely as part of Results or Discussion), emphasising:

  * Dependence on GPT-4o as the only recogniser.
  * BLN600’s specific domain (19th-century London newspapers).
  * No error localisation (page-level only).
  * Top-k+tail approximation and tail-mass effects.

**Python / Code**

* Ensure all tables and figures used in the Results section can be regenerated via your scripts (no ad-hoc spreadsheet edits).

**Deliverables**

* Draft Experimental Results + Limitations sections.
* Confirmed linkage between text and auto-generated figures/tables.

---

## Week 12 – Final Polish, Release Tag, and Handoff

**Python / Code**

* Tag a **release** in version control, e.g., `v1.0-journal-submission`:

  * Verify that `make run-all` + `make figs` work from a clean clone.
  * Archive:

    * Page-level CSVs (entropy, CER, Levenshtein, etc.).
    * Cached API outputs (or a representative subset if size is large, with instructions to regenerate).
    * Figure-generation scripts/notebooks.

* Prepare an **artifact release folder** ready for GitHub / OSF / similar:

  * `code/`, `data/` (or data links and scripts to download), `results/`, `figures/`, `README.md`.

**Writing**

* Final pass over **student-owned sections**:

  * Literature Review.
  * Experimental Setup (student-relevant parts).
  * Experimental Results + Limitations.

* Hand off to supervisor:

  * All student sections in LaTeX format.
  * A short **“implementation note”** explaining any technical caveats, known bugs, or open TODOs.

**Deliverables**

* Finalised student sections and artifact bundle.
* One-page summary of what was implemented and where to find everything in the repo.

**Final Milestone:** student work is complete and fully documented; supervisor can integrate remaining sections (Introduction framing, broader Discussion, Conclusion) and proceed to journal submission.

[1]: https://eprints.whiterose.ac.uk/id/eprint/217296/1/2024.lrec-main.219.pdf?utm_source=chatgpt.com "BLN600: A parallel corpus of machine/human transcribed ..."
[2]: https://ocr-d.de/en/spec/ocrd_eval.html?utm_source=chatgpt.com "Quality Assurance in OCR-D"
[3]: https://watercrawl.dev/blog/Character-Error-Rate?utm_source=chatgpt.com "Character Error Rate (CER): A Friendly, No-Nonsense Guide"
[4]: https://cookbook.openai.com/examples/using_logprobs?utm_source=chatgpt.com "Using logprobs"
[5]: https://www.researchgate.net/publication/355818046_A_survey_of_OCR_evaluation_tools_and_metrics?utm_source=chatgpt.com "A survey of OCR evaluation tools and metrics | Request PDF"
[6]: https://arxiv.org/pdf/1604.06225?utm_source=chatgpt.com "OCR Error Correction Using Character ..."
[7]: https://review.gale.com/2024/09/03/using-large-language-models-for-post-ocr-correction/?utm_source=chatgpt.com "Using Large Language Models for Post-OCR Correction ..."
[8]: https://docs.together.ai/docs/logprobs?utm_source=chatgpt.com "Getting Started with Logprobs"
