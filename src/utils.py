import os, math, base64, json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

from loader import load_text_pair

MODEL = "gpt-4o"
# Safe defaults, otherwise these are read from input
TOP_K = 5
WINDOW_SIZE = 5
TOP_M = 10
TOKEN_PRINT_LIMIT = 3
EXCLUDE_TOKENS = {"```", "python", "", " ", "\n", "\n\n" "latex", "json", "tag", "\\"}
CACHE_PATH = "cache/cache.json"

def init_openai_client():
     load_dotenv()
     
     api_key = os.getenv("OPENAI_API_KEY")
     if api_key is None:
          raise ValueError("Please set the OPENAI_API_KEY environment variable.")
     client = OpenAI(api_key=api_key)
     return client

def encode_image(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")

def pretty(alts):
    return ", ".join(f"{a.token!r}:{math.exp(a.logprob):.3f}" for a in alts)

def calculate_shannon_entropy(p):
    return -p * math.log(p, 2)

def get_probability(logprob):
    return math.exp(logprob)

def get_page_id_from_path(path: Path) -> int:
    return int(str(path).split("/")[-1][:-4]) # trim ".tif", ".jpg", ".tex"

def get_page_id_from_image(image: str) -> int:
     return int(image[:-4])

def load_ground_truth(page_id: int) -> str:
    pair = load_text_pair(page_id)
    if pair is None:
        raise ValueError(f"Ground truth returned None for page: {page_id}")
    _, gt = pair
    return gt.array[0]

def get_cache_key(page_id, model, top_k, prompt_version):
     return f"{page_id}_{model}_{top_k}_{prompt_version}"

def load_cache_json() -> dict:
     current_path = os.getcwd()
     cache_file = os.path.join(current_path, CACHE_PATH)

     try:
          with open(cache_file, "r", encoding="utf-8") as file:      
               cache = json.load(file)
     except json.JSONDecodeError:
          print("Cache is empty, returning empty dictionary instead")
          return {}
     return cache

def write_cache_json(mutated_obj) -> bool:
     current_path = os.getcwd()
     cache_file = os.path.join(current_path, CACHE_PATH)
     
     try:
          with open(cache_file, "w", encoding="utf-8") as file:
               json.dump(mutated_obj, file, indent=4)
     except Exception as e:
          print(f"Error writing to cache: {e}")
          return False
     return True

def get_token_logprobs(choice, top_k):
     token_logprobs = []
     # Iterate through all tokens of response
     for logprob_obj in choice.logprobs.content:
          if logprob_obj.token in EXCLUDE_TOKENS or not logprob_obj.token.strip() or logprob_obj.token.endswith("\n"):
               continue
               
          obj, logprobs, alts = {}, [0] * top_k, []
          top_logprobs = logprob_obj.top_logprobs
          
          # Iterate through all alternatives for this token
          for i in range(top_k):
               top_logprob = top_logprobs[i]
               
               token = top_logprob.token
               probability = top_logprob.logprob
               
               logprobs[i] = probability
               
               alt = {"token": token, "logprob": probability}
               alts.append(alt)
          
          obj["token"] = logprob_obj.token
          obj["logprobs"] = logprobs
          obj["alts"] = alts
          token_logprobs.append(obj)
     return token_logprobs

def get_average_bits_per_token(token_entropies):
     return sum(token_entropies) / len(token_entropies)

def convert_all_tif_to_jpg():
     image_folder = os.path.join(os.getcwd(), "data/images")
     original_filecount = len(os.listdir(image_folder))
     
     for image_filename in os.listdir(image_folder):
          image_path = os.path.join(image_folder, image_filename)
          convert_tif_to_jpg(image_path)
     
     new_filecount = len(os.listdir(image_folder))
     if original_filecount != new_filecount:
          print(f"Some images were not correctly converted resulting in loss (filecount before: {original_filecount}, filecount after: {new_filecount})")
          return
     
     print(f"Successfully converted")

def convert_tif_to_jpg(file_path):
     file_path = Path(file_path)
     if file_path.suffix.lower() != ".tif":
          print("Input file is not of .tif format. Did not convert it to JPG.")
          return
     try:
          with Image.open(file_path) as file:
               if file.mode != "RGB":
                    file = file.convert("RGB")
               new_path = file_path.with_suffix(".jpg")
               # 90 quality is basically lossless
               file.save(new_path, "JPEG", quality=90)
               # Only delete .tif after .jpg is saved
               file_path.unlink()
     except OSError as e:
          print(f"Error converting {file_path}: {e}")
     

def make_full_latex(latex_output: str) -> str:
    """
    Normalise the model’s LaTeX output.

    1.  Strip any ```latex … ``` (or plain ```) fence placed by the model.
    2.  If the cleaned text already contains a \\documentclass directive
        (i.e. it is a complete document), return it unchanged.
    3.  Otherwise, wrap the body in a minimal pre-amble and \\end{document}
        footer so that the result compiles as a stand-alone file.
    """
    import re

    cleaned = latex_output.strip()

    # remove ``` fences
    fence_prefixes = ("```latex", "```tex", "```")
    for prefix in fence_prefixes:
        if cleaned.lower().startswith(prefix):
            # drop the opening fence line
            cleaned = cleaned[len(prefix) :].lstrip()
            break

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    # if already a full document, just return it
    # We look for \documentclass anywhere (typically near the top).
    if re.search(r"\\documentclass", cleaned):
        return cleaned

    # otherwise add minimal header & footer
    header = (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage{amsmath,amssymb,amsthm}\n\n"
        "\\begin{document}\n\n"
    )
    footer = "\n\n\\end{document}\n"

    return header + cleaned + footer