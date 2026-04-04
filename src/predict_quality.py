import os, argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from loader import load_text_pair
from logprobs_client import transcribe_with_logprobs
from entropy import token_entropies_from_logprobs, surprisal_from_logprobs
from metrics import cer, levenshtein_distance
from normalization import normalize_text
from regression import get_misclassified_triage_decisions
from utils import get_page_id_from_image, is_repetitive, write_anomalies

from utils import YOUDEN_J

NORMALIZATION_TYPE = "all"

def predict_subset(top_k, max_pages, output):
     image_folder = os.path.join(os.getcwd(), "data/images")
     page_ids = np.array([get_page_id_from_image(image) for image in os.listdir(image_folder)])
     page_total = len(page_ids)
     # Get random sample
     # np.random.shuffle(page_ids)
     
     data, i = [], 0
     for page_id in page_ids:
          if i == max_pages: break
          
          image_path, ground_truth_text = load_text_pair(page_id)
          generated_transcript_text, token_logprobs = transcribe_with_logprobs(image_path, top_k)
          
          # # GPT-4o sometimes reaches a failure mode called repetition loop causing it to repeat phrases nonsensically. These generations should not be included in our observation
          # if is_repetitive(generated_transcript_text):
          #      # i + 2 since i + 1 represents current page
          #      print(f"Found an anomaly! Skipping it and going directly to page: {i+2}")
          #      write_anomalies(page_id, generated_transcript_text, ground_truth_text)
          #      continue
          
          token_surprisals = surprisal_from_logprobs(token_logprobs)
          avg_surprisal_per_token = sum(token_surprisals) / len(token_surprisals)
          
          token_entropies = token_entropies_from_logprobs(token_logprobs)
          total_bits = sum(token_entropies)
          n_tokens = len(token_entropies)
          avg_bits_per_token = total_bits / n_tokens
               
          generated_transcript_text_norm, ground_truth_text_norm = normalize_text(generated_transcript_text, ground_truth_text, NORMALIZATION_TYPE)
          
          calculated_cer = cer(generated_transcript_text_norm, ground_truth_text_norm)
          
          if calculated_cer > 1:
               print(f"Found an anomaly! OCR for {page_id} has a CER of {calculated_cer}. Skipping it and going directly to page: {i+2}")
               write_anomalies(page_id, generated_transcript_text_norm, ground_truth_text_norm)
               continue

          calculated_levenshtein = levenshtein_distance(generated_transcript_text_norm, ground_truth_text_norm)
          
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

def visualize_cer(df, top_k, indicator):
     x, y = df[f"{indicator}"], df["cer"]
     if indicator == "avg_surprisal_per_token":
          indicator = "Surprisal"
     else:
          indicator = "Entropy"
     plt.figure(figsize=(10,6))
     ax = sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
     params = {
          "xlabel":f"{indicator}", 
          "ylabel":"CER",
          "title":f"The Relationship Between {indicator} and CER For K = {top_k}",
     }
     ax.set(**params)
     plt.savefig(f"figures/{indicator.lower()}_vs_cer_k_{top_k}.png")
    
def visualize_entropy_distribution(df, top_k):
     data = df["avg_bits_per_token"]
     plt.figure(figsize=(10,6))
     plt.hist(data, bins=len(df), edgecolor="black")
     plt.xlabel("Entropy (Average Bits Per Token)")
     plt.ylabel("Frequency of Entropy Level")
     plt.title(f"Distribution of Entropy Across Data (K = {top_k})")
     plt.savefig(f"figures/entropy_distribution_k_{top_k}.png")
     
def visualize_correlation_coefficient(x, y, coefficient, top_k):
     plt.figure(figsize=(10,6))
     ax = sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
     params = {
          "xlabel":"Token", 
          "ylabel":f"{coefficient}",
          "title":f"Computed {coefficient} Correlation Coefficient Across Iterations",
     }
     ax.set(**params)
     plt.savefig(f"figures/{coefficient.lower()}_k_{top_k}")
     
def compute_pearson(x, y):
     statistic, _ = stats.pearsonr(x, y)
     return statistic

def compute_spearman(x, y):
     statistic, _ = stats.spearmanr(x, y)
     return statistic

def compute_bootstrap_confidence_interval(df: pd.DataFrame, resample_count, sample_size, top_k, indicator):
     total_r, total_p = [], []
     for _ in range(resample_count):
          sample = df.sample(sample_size, replace=True)
          x, y = sample[f"{indicator}"], sample["cer"]
          r = compute_pearson(x, y)
          p = compute_spearman(x,y)
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

def stratify_df(df: pd.DataFrame, quartiles=4):
     df["length_quartile"] = pd.qcut(df["gt_length"], q=quartiles, labels=["Q1", "Q2", "Q3", "Q4"])
     return df

# Checks if length of an excerpt affects the relationship between Entropy/Surprisal and CER
def compute_stratified_correlations(stratified_df: pd.DataFrame, indicator):
     for quartile, group in stratified_df.groupby("length_quartile"):
          x, y = group[f"{indicator}"], group["cer"]
          r = compute_pearson(x, y)
          p = compute_spearman(x, y)
          print(f"{quartile} (n={len(group)}): Pearson={r:.3f}, Spearman={p:.3f}")
          
def visualize_entropy_and_cer_across_page_lengths(stratified_df: pd.DataFrame):
     fig, axes = plt.subplots(2, 2, figsize=(10,6))
     for ax, (quartile, group) in zip(axes.flatten(), stratified_df.groupby("length_quartile")):
          ax.scatter(group["avg_bits_per_token"], group["cer"])
          ax.set_title(f"{quartile} n={len(group)}")
          ax.set_xlabel("Entropy (Average Bits Per Token)")
          ax.set_ylabel("CER")
     fig.tight_layout(pad=2.0)
     plt.savefig("figures/entropy_vs_cer_by_length.png")
          
def visualize_entropy_vs_surprisal_as_predictor(df: pd.DataFrame, top_k, threshold_type, use_primary=True):
     correct, val_indices = get_misclassified_triage_decisions(top_k, use_primary, threshold_type)
     # Validation set is only 20% of entire dataframe
     val_df = df.loc[val_indices]
     plt.figure(figsize=(10,6))
     plt.scatter(val_df[correct]["avg_surprisal_per_token"], val_df[correct]["avg_bits_per_token"], color="green", label="Correct", alpha=0.6)
     plt.scatter(val_df[~correct]["avg_surprisal_per_token"], val_df[~correct]["avg_bits_per_token"], color="red", label="Misclassified", alpha=0.6)
     plt.legend()
     plt.xlabel("Surprisal (Average)")
     plt.ylabel("Entropy (Average Bits Per Token)")
     plt.title("Entropy Vs Surprisal Scatter")
     plt.savefig(f"figures/surprisal_vs_entropy.png")

def main(indicator):
     parser = argparse.ArgumentParser(description=("Run prediction pipeline on entire BLN600 dataset"))
     
     parser.add_argument("--top-k", type=int, help="How many top logprobs to consider when computing entropy")
     parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process")
     parser.add_argument("--output", type=str, help="Path (folder) to store output")
     
     args = parser.parse_args()
     top_k = args.top_k
     max_pages = args.max_pages
     output = args.output
     
     print(f"Running with top-k set to {top_k}")
     
     print(f"Using **{indicator}** as an indicator of CER")
     # df = pd.read_csv("results_subset.csv")
     df = predict_subset(top_k, max_pages, output)
     visualize_cer(df, top_k, indicator)
     visualize_entropy_vs_surprisal_as_predictor(df, top_k, YOUDEN_J, True)
     #visualize_entropy_distribution(df)
     x, y = df[f"{indicator}"], df["cer"]
     r = compute_pearson(x, y)
     p = compute_spearman(x, y)
     print(f"Pearson Correlation Coefficient: {r:.3f}\nSpearman Correlation Coefficient: {p:.3f}")
     stratified_df = stratify_df(df)
     compute_stratified_correlations(stratified_df, indicator)
     visualize_entropy_and_cer_across_page_lengths(stratified_df)
     resample_count, sample_size = 1000, len(df)
     r_ci_lower_bound, r_ci_upper_bound, p_ci_lower_bound, p_ci_upper_bound = compute_bootstrap_confidence_interval(df, resample_count, sample_size, top_k, indicator)
     print(f"Across {resample_count} resamples of size {sample_size}, 95% of the computed 'r' values lie between range ({r_ci_lower_bound:.3f}, {r_ci_upper_bound:.3f})\nThe original computed value of 'r' on {sample_size:.3f} samples was {r}")
     print(f"Across {resample_count} resamples of size {sample_size}, 95% of the computed 'p' values lie between range ({p_ci_lower_bound:.3f}, {p_ci_upper_bound:.3f})\nThe original computed value of 'r' on {sample_size:.3f} samples was {p}")
     
if __name__ == "__main__":
     main("avg_bits_per_token")
     main("avg_surprisal_per_token")