"""
Generates a plot of ROC curves for regression results based on a specified top-k value for logprobs and reads data from a CSV file and uses the regression module to create the visualization.
"""

import argparse
from pathlib import Path

from regression import main


def _main():
    parser = argparse.ArgumentParser(description=("Generate a plot of ROC"))

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

    main(top_k)


if __name__ == "__main__":
    _main()
