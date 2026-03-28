"""
Calculates per-token entropy and entropy for all tokens of an excerpt.
"""
from utils import calculate_shannon_entropy, load_cache_json, get_probability

# Calculates the entropy of a single token
def topk_tail_entropy(logprobs: list[float]) -> float:
     assert len(logprobs) != 0 and logprobs is not None, "Cannot calculate entropy because no logprobs were passed"
     
     probs = [get_probability(p) for p in logprobs]
     mass = sum(probs)
     tail = max(0.0, 1.0 - mass)
     
     if tail == 0:
          tail_H = 0
     else:
          tail_H = calculate_shannon_entropy(tail)
          
     H = sum([calculate_shannon_entropy(p) for p in probs])
     return H + tail_H

# Calculates per-token entropy for an entire excerpt
def token_entropies_from_logprobs(token_logprobs):
     assert token_logprobs != {}, "Cannot calculate entropies from an empty token_logprobs object"
     token_entropies = []     
     for token_logprob in token_logprobs:
          token_entropy = topk_tail_entropy(token_logprob["logprobs"])
          token_entropies.append(token_entropy)
     return token_entropies


if __name__ == "__main__":
     cache = load_cache_json()
     token_logprobs = cache["3207643520_gpt-4o_5_1"]["token_logprobs"]
     entropies = token_entropies_from_logprobs(token_logprobs)
     
     total_bits = sum(entropies)
     average_bits_per_token = total_bits / len(entropies)
     
     print(f"Total bits: {total_bits}")
     print(f"Average bits per token: {average_bits_per_token}")
     