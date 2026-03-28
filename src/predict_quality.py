import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from loader import load_text_pair
from logprobs_client import transcribe_with_logprobs
from entropy import token_entropies_from_logprobs
from metrics import cer, levenshtein_distance
from normalization import normalize_text
from utils import get_page_id_from_image, is_repetitive, write_anomalies

PAGES_TO_LOAD = 100
NORMALIZATION_TYPE = "all"
SCATTER_PLOT_CONFIG = {
     "s": 20,
     "marker": "o",
     "color": "green",
     "alpha": 0.6,
}

def predict_subset():
     image_folder = os.path.join(os.getcwd(), "data/images")
     page_ids = np.array([get_page_id_from_image(image) for image in os.listdir(image_folder)])
     
     # Get random sample
     # np.random.shuffle(page_ids)
     
     data = []
     for i in range(PAGES_TO_LOAD):
          page_id = int(page_ids[i])
          
          image_path, ground_truth_text = load_text_pair(page_id)
          generated_transcript_text, token_logprobs = transcribe_with_logprobs(image_path)
          
          # GPT-4o sometimes reaches a failure mode called repetition loop causing it to repeat phrases nonsensically. These generations should not be included in our observation
          if is_repetitive(generated_transcript_text):
               # i + 2 since i + 1 represents current page
               print(f"Found an anomaly! Skipping it and going directly to page: {i+2}")
               write_anomalies(page_id, generated_transcript_text, ground_truth_text)
               continue
          
          token_entropies = token_entropies_from_logprobs(token_logprobs)
          
          total_bits = sum(token_entropies)
          n_tokens = len(token_entropies)
          avg_bits_per_token = total_bits / n_tokens
               
          generated_transcript_text_norm, ground_truth_text_norm = normalize_text(generated_transcript_text, ground_truth_text, NORMALIZATION_TYPE)
          
          calculated_cer = cer(generated_transcript_text_norm, ground_truth_text_norm)

          calculated_levenshtein = levenshtein_distance(generated_transcript_text_norm, ground_truth_text_norm)
          
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
          data.append(row)
          print(f"Processed {i+1}/{PAGES_TO_LOAD} pages...")
     
     df = pd.DataFrame(data)
     df.to_csv("results_subset.csv")
     return df

def visualize_cer(df):
     x_label, y_label = df["avg_bits_per_token"], df["cer"]
     plt.figure(figsize=(10,6))
     plt.scatter(x_label, y_label, **SCATTER_PLOT_CONFIG)
     plt.xlabel("Entropy (bits per token)")
     plt.ylabel("CER")
     plt.title("The Relationship Between Entropy and CER in OCR")
     plt.show()
    
def visualize_entropy_distribution(df):
     data = df["avg_bits_per_token"]
     plt.figure(figsize=(10,6))
     plt.hist(data, bins=PAGES_TO_LOAD, edgecolor="black")
     plt.xlabel("Entropy Levels")
     plt.ylabel("Frequency")
     plt.title("Frequency of Entropy Across Tokens")
     plt.show()
     
def compute_pearson(x, y):
     statistic, _ = stats.pearsonr(x, y)
     return statistic

def compute_spearman(x, y):
     statistic, _ = stats.spearmanr(x, y)
     return statistic

def compute_bootstrap_confidence_interval():
     
     return
     
def main():
     # df = predict_subset()
     df = pd.read_csv("results_subset.csv")
     visualize_cer(df)
     visualize_entropy_distribution(df)
     x, y = df["avg_bits_per_token"], df["cer"]
     pearson_coefficient = compute_pearson(x, y)
     spearman_coefficient = compute_spearman(x, y)
     print(f"Pearson Correlation Coefficient: {pearson_coefficient:.3f}\nSpearman Correlation Coefficient: {spearman_coefficient:.3f}")
     
     
if __name__ == "__main__":
     main()