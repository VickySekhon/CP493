"""
Runs the entire pipeline from start to finish, generating a single CSV file.
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loader import load_text_pair
from logprobs_client import transcribe_with_logprobs
from entropy import token_entropies_from_logprobs, surprisal_from_logprobs
from metrics import cer, levenshtein_distance
from normalization import normalize_text
from utils import (
    get_page_id_from_image,
    write_anomalies,
    compute_pearson,
    compute_spearman,
)

NORMALIZATION_TYPE = "all"


def predict_subset(top_k, max_pages, output):
    image_folder = os.path.join(os.getcwd(), "data/images")
    page_ids = np.array(
        [get_page_id_from_image(image) for image in os.listdir(image_folder)]
    )
    page_total = len(page_ids)

    # Get random sample
    np.random.shuffle(page_ids)

    data, i = [], 0
    for page_id in page_ids:
        if i == max_pages:
            break

        image_path, ground_truth_text = load_text_pair(page_id)
        generated_transcript_text, token_logprobs = transcribe_with_logprobs(
            image_path, top_k
        )

        token_surprisals = surprisal_from_logprobs(token_logprobs)
        avg_surprisal_per_token = sum(token_surprisals) / len(token_surprisals)

        token_entropies = token_entropies_from_logprobs(token_logprobs)
        total_bits = sum(token_entropies)
        n_tokens = len(token_entropies)
        avg_bits_per_token = total_bits / n_tokens

        generated_transcript_text_norm, ground_truth_text_norm = normalize_text(
            generated_transcript_text, ground_truth_text, NORMALIZATION_TYPE
        )

        calculated_cer = cer(generated_transcript_text_norm, ground_truth_text_norm)

        if calculated_cer > 1:
            print(
                f"Found an anomaly! OCR for {page_id} has a CER of {calculated_cer}. Skipping it and going directly to page: {i+2}"
            )
            write_anomalies(
                page_id, generated_transcript_text_norm, ground_truth_text_norm
            )
            continue

        calculated_levenshtein = levenshtein_distance(
            generated_transcript_text_norm, ground_truth_text_norm
        )

        row = {
            "page_id": page_id,
            "avg_bits_per_token": avg_bits_per_token,
            "avg_surprisal_per_token": avg_surprisal_per_token,
            "total_bits": total_bits,
            "n_tokens": n_tokens,
            "cer": calculated_cer,
            "levenshtein": calculated_levenshtein,
            "gt_length": len(ground_truth_text_norm),
            "normalization_profile": NORMALIZATION_TYPE,
        }
        data.append(row)
        print(f"Processed {i+1}/{page_total} pages (max = {max_pages})...")
        i += 1

    df = pd.DataFrame(data)
    df.to_csv(f"{output}/results_k_{top_k}.csv")
    return df


def visualize_correlation_coefficient(x, y, coefficient, top_k):
    plt.figure(figsize=(10, 6))
    ax = sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
    params = {
        "xlabel": "Token",
        "ylabel": f"{coefficient}",
        "title": f"Computed {coefficient} Correlation Coefficient Across Iterations",
    }
    ax.set(**params)
    plt.savefig(f"figures/{coefficient.lower()}_k_{top_k}")


def compute_bootstrap_confidence_interval(
    df: pd.DataFrame, resample_count, sample_size, top_k, indicator
):
    total_r, total_p = [], []
    for _ in range(resample_count):
        sample = df.sample(sample_size, replace=True)
        x, y = sample[f"{indicator}"], sample["cer"]
        r = compute_pearson(x, y)
        p = compute_spearman(x, y)
        total_r.append(r)
        total_p.append(p)
    r_ci_lower_bound = np.percentile(total_r, 2.5)
    r_ci_upper_bound = np.percentile(total_r, 97.5)

    p_ci_lower_bound = np.percentile(total_p, 2.5)
    p_ci_upper_bound = np.percentile(total_p, 97.5)

    # Plot correlation values across bootstrap iterations.
    x = [i for i in range(1, resample_count + 1)]
    visualize_correlation_coefficient(x, total_r, "Pearson", top_k)
    visualize_correlation_coefficient(x, total_p, "Spearman", top_k)

    return r_ci_lower_bound, r_ci_upper_bound, p_ci_lower_bound, p_ci_upper_bound


def main(indicator):
    parser = argparse.ArgumentParser(
        description=("Run prediction pipeline on entire BLN600 dataset")
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="How many top logprobs to consider when computing entropy",
    )
    parser.add_argument(
        "--max-pages", type=int, help="Maximum number of pages to process"
    )
    parser.add_argument("--output", type=str, help="Path (folder) to store output")

    args = parser.parse_args()
    top_k = args.top_k or 10
    max_pages = args.max_pages or 100
    output = args.output or "csvs"

    print(f"Using **{indicator}** as an indicator of CER")
    df = predict_subset(top_k, max_pages, output)


if __name__ == "__main__":
    main("avg_bits_per_token")
