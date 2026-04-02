import os, argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loader import load_text_pair
from logprobs_client import transcribe_with_logprobs, flush_cache_updates
from entropy import token_entropies_from_logprobs
from metrics import cer, levenshtein_distance
from normalization import normalize_text
from utils import get_page_id_from_image, is_repetitive, write_anomalies, load_cache_json

NORMALIZATION_TYPE = "all"
_anomaly_lock = Lock()

def process_page(page_id, top_k, cache_snapshot):
     local_cache = {}
     image_path, ground_truth_text = load_text_pair(page_id)
     generated_transcript_text, token_logprobs = transcribe_with_logprobs(
          image_path,
          top_k,
          cache_snapshot=cache_snapshot,
          local_cache=local_cache,
          persist_cache=False,
     )

     # # GPT-4o sometimes reaches a failure mode called repetition loop causing it to repeat phrases nonsensically.
     # if is_repetitive(generated_transcript_text):
     #      with _anomaly_lock:
     #           write_anomalies(page_id, generated_transcript_text, ground_truth_text)
     #      return None, local_cache

     token_entropies = token_entropies_from_logprobs(token_logprobs)
     n_tokens = len(token_entropies)
     if n_tokens == 0:
          return None, local_cache

     total_bits = sum(token_entropies)
     avg_bits_per_token = total_bits / n_tokens

     generated_transcript_text_norm, ground_truth_text_norm = normalize_text(
          generated_transcript_text,
          ground_truth_text,
          NORMALIZATION_TYPE,
     )

     calculated_cer = cer(generated_transcript_text_norm, ground_truth_text_norm)
     calculated_levenshtein = levenshtein_distance(generated_transcript_text_norm, ground_truth_text_norm)
     
     if calculated_cer > 1:
          print(f"Found an anomaly! OCR for {page_id} has a CER of {calculated_cer}")
          with _anomaly_lock:
               write_anomalies(page_id, generated_transcript_text_norm, ground_truth_text_norm)
          return None, local_cache

     row = {
          "page_id": page_id,
          "avg_bits_per_token": avg_bits_per_token,
          "total_bits": total_bits,
          "n_tokens": n_tokens,
          "cer": calculated_cer,
          "levenshtein": calculated_levenshtein,
          "gt_length": len(ground_truth_text_norm),
          "normalization_profile": NORMALIZATION_TYPE,
     }
     return row, local_cache


def predict_subset(top_k, max_pages, output, workers):
     image_folder = os.path.join(os.getcwd(), "data/images")
     page_ids = np.array([get_page_id_from_image(image) for image in os.listdir(image_folder)])
     page_total = len(page_ids)
     # Get random sample
     np.random.shuffle(page_ids)

     # Read once; workers treat this as read-only cache snapshot.
     cache_snapshot = load_cache_json()

     target_page_ids = list(page_ids[:max_pages]) if max_pages is not None else list(page_ids)
     max_target = len(target_page_ids)

     data = []
     cache_updates = {}
     processed_count = 0
     with ThreadPoolExecutor(max_workers=workers) as executor:
          futures = {
               executor.submit(process_page, int(page_id), top_k, cache_snapshot): int(page_id)
               for page_id in target_page_ids
          }
          for future in as_completed(futures):
               page_id = futures[future]
               try:
                    row, local_cache = future.result()
               except Exception as e:
                    print(f"Failed page {page_id}: {e}")
                    continue

               processed_count += 1
               print(f"Processed {processed_count}/{max_target} pages (workers={workers}, available={page_total})...")
               if local_cache:
                    cache_updates.update(local_cache)
               if row is not None:
                    data.append(row)

     successful = flush_cache_updates(cache_updates)
     if not successful:
          print("Warning: failed to flush cache updates to cache/cache.json")
     
     df = pd.DataFrame(data)
     df.to_csv(f"{output}/results_k_{top_k}.csv")
     return df

def visualize_cer(df, top_k):
     x, y = df["avg_bits_per_token"], df["cer"]
     plt.figure(figsize=(10,6))
     ax = sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
     params = {
          "xlabel":"Entropy (Average Bits Per Token)", 
          "ylabel":"CER",
          "title":f"The Relationship Between Entropy (Average Bits Per Token) and CER For K = {top_k}",
     }
     ax.set(**params)
     plt.savefig(f"figures/entropy_vs_cer_k_{top_k}.png")
    
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
     plt.savefig(f"figures/{coefficient.lower()}_k_{top_k}.png")
     
def compute_pearson(x, y):
     statistic, _ = stats.pearsonr(x, y)
     return statistic

def compute_spearman(x, y):
     statistic, _ = stats.spearmanr(x, y)
     return statistic

def compute_bootstrap_confidence_interval(df: pd.DataFrame, resample_count, sample_size, top_k):
     total_r, total_p = [], []
     for _ in range(resample_count):
          sample = df.sample(sample_size, replace=True)
          x, y = sample["avg_bits_per_token"], sample["cer"]
          r = compute_pearson(x, y)
          p = compute_spearman(x,y)
          total_r.append(r)
          total_p.append(p)
     r_ci_lower_bound = np.percentile(total_r, 2.5)
     r_ci_upper_bound = np.percentile(total_r, 97.5)
     
     p_ci_lower_bound = np.percentile(total_p, 2.5)
     p_ci_upper_bound = np.percentile(total_p, 97.5)
     
     # Plot correlation values across bootstrap iterations.
     x = [i for i in range(1, len(total_r) + 1)]
     visualize_correlation_coefficient(x, total_r, "Pearson", top_k)
     visualize_correlation_coefficient(x, total_p, "Spearman", top_k)
     
     return r_ci_lower_bound, r_ci_upper_bound, p_ci_lower_bound, p_ci_upper_bound
     
def main():
     parser = argparse.ArgumentParser(description=("Run prediction pipeline on entire BLN600 dataset"))
     
     parser.add_argument("--top-k", type=int, help="How many top logprobs to consider when computing entropy")
     parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process")
     parser.add_argument("--output", type=str, help="Path (folder) to store output")
     parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for page-level parallelism")
     
     args = parser.parse_args()
     top_k = args.top_k if args.top_k is not None else 5
     max_pages = args.max_pages
     output = args.output if args.output is not None else "csvs"
     workers = max(1, args.workers)

     os.makedirs(output, exist_ok=True)
     os.makedirs("figures", exist_ok=True)
     
     print(f"Running with top-k={top_k}, max-pages={max_pages}, output={output}, workers={workers}")
     # df = pd.read_csv("results_subset.csv")
     df = predict_subset(top_k, max_pages, output, workers)
     visualize_cer(df, top_k)
     # visualize_entropy_distribution(df)
     x, y = df["avg_bits_per_token"], df["cer"]
     r = compute_pearson(x, y)
     p = compute_spearman(x, y)
     print(f"Pearson Correlation Coefficient: {r:.3f}\nSpearman Correlation Coefficient: {p:.3f}")
     resample_count, sample_size = 1000, len(df)
     r_ci_lower_bound, r_ci_upper_bound, p_ci_lower_bound, p_ci_upper_bound = compute_bootstrap_confidence_interval(df, resample_count, sample_size, top_k)
     print(f"Across {resample_count} resamples of size {sample_size}, 95% of the computed 'r' values lie between range ({r_ci_lower_bound:.3f}, {r_ci_upper_bound:.3f})\nThe original computed value of 'r' on {sample_size:.3f} samples was {r}")
     print(f"Across {resample_count} resamples of size {sample_size}, 95% of the computed 'p' values lie between range ({p_ci_lower_bound:.3f}, {p_ci_upper_bound:.3f})\nThe original computed value of 'r' on {sample_size:.3f} samples was {p}")
     
if __name__ == "__main__":
     main()