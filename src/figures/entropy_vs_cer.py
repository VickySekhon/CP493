"""
Script to visualize the relationship between entropy and CER using plots from CSV data.
"""

import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_cer(df, top_k, indicator="avg_bits_per_token"):
    x, y = df[f"{indicator}"], df["cer"]
    if indicator == "avg_surprisal_per_token":
        indicator = "Surprisal"
    else:
        indicator = "Entropy"
    plt.figure(figsize=(10, 6))
    ax = sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
    params = {
        "xlabel": f"{indicator}",
        "ylabel": "CER",
        "title": f"The Relationship Between {indicator} and CER For K = {top_k}",
    }
    ax.set(**params)
    plt.savefig(f"figures/{indicator.lower()}_vs_cer_k_{top_k}.png")


def visualize_entropy_distribution(df, top_k):
    data = df["avg_bits_per_token"]
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=len(df))
    plt.xlabel("Entropy (Average Bits Per Token)")
    plt.ylabel("Frequency of Entropy Level")
    plt.title(f"Distribution of Entropy Across Data (K = {top_k})")
    plt.savefig(f"figures/entropy_distribution_k_{top_k}.png")


def main():
    parser = argparse.ArgumentParser(
        description=("Generate a plot of the relationship between CER and Entropy")
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
    visualize_cer(df, top_k)
    visualize_entropy_distribution(df, top_k)


if __name__ == "__main__":
    main()
