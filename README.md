## Requirements

## Steps

1. historical newspaper page image -> GPT-4o transcript + token-level top-k logprobs
2. find shannon entropy (top-k+tail) per token
3. compute the average entropy per page
4. compute CER (levenshtein distance from GPT-4o transcript to GT)
5. use entropy score as a triage to determine whether to accept or reject a GPT-4o-generated transcript

## Constraints

1. one GPT-4o decoding run per page
2. use native logprobs only
3. 'k' defaults to 5

# Code Structure

1. private github repo
2. README, LICENSE, src/, notebooks/, data/, cache/, figures/
3. virtual environment
     - pin dependencies:
          - pandas
          - numpy
          - rapidfuzz[levenshtein]
          - matplotlib
          - seaborn (optional)
          - scikit-learn
          - python-dotenv (or similar)
          - openai

# My Notes
- logprobs returns logprob = ln(probability) for a given token. To get the actual probability back (0-1 range) we take e^logprob.
e.g.

1.
ln(10) = 2.30
    |       |
original p  logprob

2.
e^2.30 = 10

- the way logprobs work is 0 = 100% probability, <0 = lower probabilities. Whereas, the way entropy works is 0 = 0 uncertainty probability, 1 = alot of uncertainty from equally likely outcomes.
- top-logprobs is restricted to max of 5.
- can plot to understand relationship (ln(x) = lobprob, e^x = probability)

Correlations
- Pearson
    * Measures linear relationship and answers the question: "if x increases, does y increase proportionally?"
    * Outputs 1 for perfect positive line, 0 for no linear relationship, and -1 for perfect negative line
- Spearman
    * Instead of looking at raw values, they are assigned a rank
    * Answers the question: "if x increases, do y generally go up or down (not necessarily linearly)?"
    * Outputs 1 for perfect positive association of ranks, 0 for no monotonic association, and -1 for perfect negative association of ranks

**In summary, the pearson correlation helps us understand if CER (y) *linearly* increases when Entropy (x) increases, while spearman helps us understand if CER (y) increases when Entropy (x) increases**

Confidence Intervals
- Bootstrap CI
    * Instead of assuming data follows normal distribution, you re-sample points and compute a statistic on them, and then compute the tails based on your CI to get the range in which 'CI' amount of your computed statistic values lie

TESTS:
- https://www.diffchecker.com/szQ35Wnw/ (normalized)
- https://www.diffchecker.com/8wySsLjv/ (anomaly difference)

CER:
0.0 – 0.05 → excellent OCR (modern, clean documents)
0.05 – 0.15 → decent OCR (some noise/degradation)
0.15 – 0.40 → poor OCR (historical, degraded, low quality scans)
0.40+ → very poor (heavily damaged, unusual scripts, bad scans)

OUTLIER:
* GPT-4 failure mode called repetition loop causing the 8.1 CER anomaly