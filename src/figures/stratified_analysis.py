"""
Performs stratified analysis on text data, computing correlations and generating visualizations
to explore the relationship between entropy, surprisal, and CER across different length quartiles.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utils import compute_pearson, compute_spearman
from regression import get_misclassified_triage_decisions

from utils import YOUDEN_J


def stratify_df(df: pd.DataFrame, quartiles=4):
    df["length_quartile"] = pd.qcut(
        df["gt_length"], q=quartiles, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    return df


# Checks if length of an excerpt affects the relationship between Entropy/Surprisal and CER
def compute_stratified_correlations(
    stratified_df: pd.DataFrame, indicator="avg_bits_per_token"
):
    for quartile, group in stratified_df.groupby("length_quartile"):
        x, y = group[f"{indicator}"], group["cer"]
        r = compute_pearson(x, y)
        p = compute_spearman(x, y)
        print(f"{quartile} (n={len(group)}): Pearson={r:.3f}, Spearman={p:.3f}")


def visualize_entropy_and_cer_across_page_lengths(stratified_df: pd.DataFrame, top_k):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (quartile, group) in zip(
        axes.flatten(), stratified_df.groupby("length_quartile")
    ):
        ax.scatter(group["avg_bits_per_token"], group["cer"])
        ax.set_title(f"{quartile} n={len(group)}")
        ax.set_xlabel("Entropy (Average Bits Per Token)")
        ax.set_ylabel("CER")
    plt.suptitle("The Relationship Between CER and Entropy Across Different Lengths")
    fig.tight_layout(pad=2.0)
    plt.savefig(f"figures/entropy_vs_cer_stratified_k_{top_k}.png")


def visualize_entropy_vs_surprisal_as_predictor(
    df: pd.DataFrame, top_k, threshold_type, use_primary=False
):
    correct, val_indices = get_misclassified_triage_decisions(
        top_k, use_primary, threshold_type
    )
    # Validation set is only 20% of entire dataframe
    val_df = df.loc[val_indices]
    plt.figure(figsize=(10, 6))
    plt.scatter(
        val_df[correct]["avg_surprisal_per_token"],
        val_df[correct]["avg_bits_per_token"],
        color="green",
        label="Correct",
        alpha=0.6,
    )
    plt.scatter(
        val_df[~correct]["avg_surprisal_per_token"],
        val_df[~correct]["avg_bits_per_token"],
        color="red",
        label="Misclassified",
        alpha=0.6,
    )
    plt.legend()
    plt.xlabel("Surprisal (Average)")
    plt.ylabel("Entropy (Average Bits Per Token)")
    plt.title("The Correlation Between Entropy and Surprisal")
    plt.savefig(f"figures/surprisal_vs_entropy_k_{top_k}.png")


def main():
    parser = argparse.ArgumentParser(
        description=("Generate stratified scatter plot and surprisal vs entropy plot")
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="How many top logprobs were considered when computing entropy",
    )

    args = parser.parse_args()
    top_k = args.top_k or 10

    path_to_csv = Path(f"csvs/results_k_{top_k}.csv")
    assert (
        path_to_csv.exists()
    ), f"{path_to_csv} does not exist, please run `make run-all` first to generate it."

    df = pd.read_csv(f"csvs/results_k_{top_k}.csv")
    stratified_df = stratify_df(df)
    compute_stratified_correlations(stratified_df)
    visualize_entropy_and_cer_across_page_lengths(stratified_df, top_k)
    visualize_entropy_vs_surprisal_as_predictor(df, top_k, YOUDEN_J)


if __name__ == "__main__":
    main()
