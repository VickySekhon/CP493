"""
Loads a page excerpt, converts it into latex, and then runs sliding window analysis to determine areas with most entropy.

Usage: <python3 src/scan2latex_entropy.py --top-k 1 --window-size 5 --top-m 10 --norm all data/images/3200797037.jpg>
"""

import os, sys, time, argparse, heapq, dotenv, io
from pathlib import Path
from datetime import datetime
from openai import OpenAI

from normalization import *
from metrics import cer
from utils import MODEL, TOKEN_PRINT_LIMIT, EXCLUDE_TOKENS
from utils import (
    pretty,
    get_page_id_from_path,
    load_ground_truth,
    make_full_latex,
    encode_image,
    calculate_shannon_entropy,
    get_probability,
)

class TeeOutput:
    """Captures all print output while also displaying to console."""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original_stdout.flush()

    def get_log(self):
        return self.buffer.getvalue()
def chat(
    msgs,
    client,
    model,
    top_k,
    temperature=0.5,
    top_p=0.9,
    n=1,
    seed=12345,
    max_tokens=10_000,
    retries=10,
):
    for attempt in range(retries):
        try:
            params = {
                "model": model,
                "messages": msgs,
                # High = more creative, Low = more focused
                "temperature": temperature,
                # Model looks at the top 90% of tokens it generates
                "top_p": top_p,
                # Controls how much model repeats itself (-2 - 2, with '+' value being penalize for repetition
                "n": n,
                "seed": seed,
                "max_tokens": max_tokens,
                "logprobs": True,
                "top_logprobs": top_k,
            }
            return client.chat.completions.create(**params)
        except Exception as e:
            print(f"Error: {e} – retrying in 5 seconds ({attempt + 1}/{retries})")
            time.sleep(5)
    raise RuntimeError("Max retries exceeded for chat function.")

def calculate_entropy(tok_infos, N, top_k):
    total_H = 0.0
    # Store entropy of each position for sliding window
    pos_entropy = []

    # A given token
    for info in tok_infos:
        H_pos, mass = 0.0, 0.0

        top_k_logprobs = info.top_logprobs[:top_k]
        # loop through top k logprobs
        for alt in top_k_logprobs:

            # Convert to probability value
            p = get_probability(alt.logprob)
            mass += p

            # log(0) is undefined
            if p == 0.0:
                continue

            H_pos += calculate_shannon_entropy(p)

        # Residual probability
        p_tail = 1.0 - mass
        if p_tail > 0.0:
            H_pos += calculate_shannon_entropy(p_tail)

        total_H += H_pos
        pos_entropy.append(H_pos)

    avg_H = total_H / N

    return total_H, avg_H, pos_entropy

def sliding_window(pos_entropy, N, W, top_m):
    # running window sum for O(N) computation
    window_sum = sum(pos_entropy[:W])
    # (average entropy within window, window index)
    windows = [(window_sum / W, 0)]

    # avoid overlapping top windows
    i = W
    while i < N:
        window_sum = 0
        window_sum = sum(pos_entropy[i : i + W])
        windows.append((window_sum, i))
        i += W

    top_windows = heapq.nlargest(top_m, windows, key=lambda x: x[0])
    return top_windows


if __name__ == "__main__":
    print("Starting program...")

    # Redirect stdout to capture all print statements
    _tee = TeeOutput(sys.stdout)
    sys.stdout = _tee
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Convert a scanned page to LaTeX (auto-adds a standard "
            "document header/footer unless the output already starts "
            "with \\documentclass) and analyse token-level Shannon "
            "entropy (whole sequence + sliding windows)."
        )
    )
    parser.add_argument(
        "image", type=str, help="Path to the JPEG/PNG/PDF page to process"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of alternative tokens to request (1–20).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        metavar="W",
        help="Sliding-window size in tokens.",
    )
    parser.add_argument(
        "--top-m",
        type=int,
        default=3,
        metavar="M",
        help="Show the M windows with the largest average entropy.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["none", "all", "interactive"],
        default="all",
        metavar="N",
        help="Type of normalization (none/all/interactive) to apply to ocr and ground_truth text excerpts.",
    )
    args = parser.parse_args()
    # K logprobs to consider, W window size, TOP_M windows considered
    TOP_K = args.top_k
    W = args.window_size
    TOP_M = args.top_m
    NORM_TYPE = args.norm

    IMAGE_PATH = Path(os.path.join(os.getcwd(), args.image))
    if not IMAGE_PATH.exists():
        sys.exit(f"Error reading image. Path: '{IMAGE_PATH}' was not found.")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are an expert text processor specialized in converting a scanned "
        "document into properly formatted LaTeX. Identify every mathematical "
        "expression and repair typical OCR artifacts. Preserve prose verbatim. "
        "Return only the converted document inside one ```latex … ``` fence."
    )

    user_text = (
        "Convert the attached scanned page into LaTeX.\n"
        "• Detect every math expression—inline or displayed—and typeset it in "
        "valid LaTeX, fixing OCR distortions.\n"
        "• Preserve all non-mathematical content exactly.\n"
        "• Return ONLY the LaTeX code inside a single ```latex … ``` fence."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(IMAGE_PATH)}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    resp = chat(messages, client, MODEL, TOP_K)
    choice = resp.choices[0]
    reply = choice.message.content.strip()
    print("\nAssistant reply (LaTeX only expected):\n")
    print(reply, "\n")

    # save LaTeX file next to the image
    full_latex = make_full_latex(reply)
    tex_file_path = IMAGE_PATH.with_suffix(".tex")
    try:
        tex_file_path.write_text(full_latex, encoding="utf-8")
        print(f"\nLaTeX output saved to: {tex_file_path}")
    except Exception as e:
        print(f"\nError writing LaTeX file: {e}")
    # ─────────────── collect & filter tokens ──────────────────────
    tok_infos = [
        t
        for t in choice.logprobs.content
        if t.token not in EXCLUDE_TOKENS and t.token.strip()
    ]

    # For debugging: write log probabilities to a file for review
    write_path = "data/log-probs/1.txt"
    with open(write_path, "w", encoding="utf-8") as f2:
        f2.writelines(str(tok_infos))

    N = len(tok_infos)
    if N == 0:
        sys.exit("No tokens to analyse.")

    total_H, avg_H, pos_entropy = calculate_entropy(tok_infos, N, TOP_K)
    print(f"Token count: {N}")
    print(
        f"Total entropy across all {TOP_K} top-k for all {N} tokens: {total_H:.6f} bits"
    )
    print(f"Average entropy (page-level): {avg_H:.6f} bits/token")
    if W <= 0:
        sys.exit("Window size W must be positive.")
    elif N < W:
        print(
            f"\nWindow size W={W} exceeds sequence length N={N}; "
            "skipping sliding-window analysis."
        )
    else:
        top_windows = sliding_window(pos_entropy, N, W, TOP_M)

        print(
            f"\nTop {len(top_windows)} windows (size W={W}) "
            f"with highest average entropy:"
        )
        for rank, (avg_w, start) in enumerate(top_windows, 1):
            end = start + W - 1
            all_tokens_in_window = "".join(
                tok_infos[i].token for i in range(start, end + 1)
            )
            print(
                f"{rank:>2}. [{start:>3}–{end:>3}]  "
                f'avg H = {avg_w:.4f} bits/token  →  "{all_tokens_in_window}"'
            )
    head = tok_infos[:TOKEN_PRINT_LIMIT]
    tail = tok_infos[-TOKEN_PRINT_LIMIT:] if N > TOKEN_PRINT_LIMIT else []

    if head:
        print(f"\nFirst {len(head)} tokens (+ top-k probs):")
        for t in head:
            print(f"  {t.token!r} → {pretty(t.top_logprobs)}")
    if tail:
        print(f"\nLast  {len(tail)} tokens (+ top-k probs):")
        for t in tail:
            print(f"  {t.token!r} → {pretty(t.top_logprobs)}")

    log_dir = Path(os.getcwd()) / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"log_{timestamp}.txt"

    try:
        log_file.write_text(_tee.get_log(), encoding="utf-8")
        print(f"\nLog saved to: {log_file}")
    except Exception as e:
        print(f"\nError writing log file: {e}")
    PAGE_ID = get_page_id_from_path(IMAGE_PATH)

    gt = load_ground_truth(PAGE_ID)
    ocr, gt = normalize_text(full_latex, gt, NORM_TYPE)
    _cer = cer(ocr, gt)

    print(
        f"The CER between ocr and gt text excerpts for page: '{PAGE_ID}' is: {_cer:.2f}"
    )
