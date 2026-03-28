"""
Tests per-token entropy and entropy for all tokens of output in entropy.py by comparing it to results from scan2latex_entropy.py.
"""
from utils import load_cache_json
from entropy import token_entropies_from_logprobs
from scan2latex_entropy import calculate_entropy

def test_page_id_3207643520():
     cache_key = "3207643520_gpt-4o_5_1"
     
     cache = load_cache_json()
     token_logprobs = cache[cache_key]["token_logprobs"]
     
     x1 = token_entropies_from_logprobs(token_logprobs)
     x1 = (sum(x1), sum(x1)/len(x1))
     # Positional entropies are not needed remove them
     x2 = calculate_entropy(token_logprobs)[:-1]
     
     assert x1 == x2, f"Entropy does not match. entropy.py produced: {x1} whereas scan2latex_entropy.py produced {x2}"

if __name__ == "__main__":
     test_page_id_3207643520()
     