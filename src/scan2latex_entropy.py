# python3 src/scan2latex_entropy.py --top-k 1 --window-size 5 --top-m 10 data/images/3200797037.jpg

# python scan2latex_entropy.py -h

#!/usr/bin/env python3
import os, sys, time, math, base64, argparse, heapq, dotenv
from pathlib import Path
from openai import OpenAI

# ─────────────── configuration ───────────────────────────────
MODEL = "gpt-4o"  # must support logprobs
TOKEN_PRINT_LIMIT = 3  # diagnostics: how many tokens to show
EXCLUDE_TOKENS = {"```", "python", "", " ", "\n", "latex", "json", "tag"}
LOG2 = math.log(2)  # ln 2  (nat → bit)

dotenv.load_dotenv()

# ──────────────── CLI ────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=(
        "Convert a scanned page to LaTeX (auto-adds a standard "
        "document header/footer unless the output already starts "
        "with \\documentclass) and analyse token-level Shannon "
        "entropy (whole sequence + sliding windows)."
    )
)

parser.add_argument("image", type=str, help="Path to the JPEG/PNG/PDF page to process")
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
args = parser.parse_args()

TOP_K = args.top_k
# Window size stuff
W = args.window_size
TOP_M = args.top_m

IMAGE_PATH = Path(os.path.join(os.getcwd(), args.image))
# IMAGE_PATH = Path(os.path.join(os.getcwd(), "data/images/3200797037.jpg"))
if not IMAGE_PATH.exists():
    sys.exit(f"❌ File not found: {IMAGE_PATH}")


# ─────────────── helpers ──────────────────────────────────────
def encode_image(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def pretty(alts):
    return ", ".join(f"{a.token!r}:{math.exp(a.logprob):.3f}" for a in alts)


def make_full_latex(latex_output: str) -> str:
    """
    Normalise the model’s LaTeX output.

    1.  Strip any ```latex … ``` (or plain ```) fence placed by the model.
    2.  If the cleaned text already contains a \documentclass directive
        (i.e. it is a complete document), return it unchanged.
    3.  Otherwise, wrap the body in a minimal pre-amble and \end{document}
        footer so that the result compiles as a stand-alone file.
    """
    import re

    cleaned = latex_output.strip()

    # ── remove ``` fences ─────────────────────────────────────────────────
    fence_prefixes = ("```latex", "```tex", "```")
    for prefix in fence_prefixes:
        if cleaned.lower().startswith(prefix):
            # drop the opening fence line
            cleaned = cleaned[len(prefix) :].lstrip()
            break

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    # ── if already a full document, just return it ───────────────────────
    # We look for \documentclass anywhere (typically near the top).
    if re.search(r"\\documentclass", cleaned):
        return cleaned

    # ── otherwise add minimal header & footer ─────────────────────────────
    header = (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage{amsmath,amssymb,amsthm}\n\n"
        "\\begin{document}\n\n"
    )
    footer = "\n\n\\end{document}\n"

    return header + cleaned + footer


# ─────────────── OpenAI client ────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

# ─────────────── system & user messages with image ────────────

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


# ─────────────── chat wrapper ─────────────────────────────────
def chat(msgs):
    while True:
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=msgs,
                temperature=0.5,
                top_p=0.9,
                n=1,
                seed=12345,
                max_tokens=10_000,
                logprobs=True,
                top_logprobs=TOP_K,
            )
        except Exception as e:
            print(f"Error: {e} – retrying in 5 s")
            time.sleep(5)


# ─────────────── main single request ──────────────────────────
resp = chat(messages)
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
N = len(tok_infos)
if N == 0:
    sys.exit("No tokens to analyse.")

# ─────────────── per-token entropy + totals ───────────────────
total_H = 0.0
pos_entropy = []  # store H of each position for windows

for info in tok_infos:
    H_pos, mass = 0.0, 0.0
    for alt in info.top_logprobs[:TOP_K]:
        p = math.exp(alt.logprob)
        if p == 0.0:
            continue
        H_pos += -p * alt.logprob / LOG2  # −p ln p  (→ bits)
        mass += p

    p_tail = max(0.0, 1.0 - mass)  # residual prob.
    if p_tail > 0.0:
        H_pos += -p_tail * math.log(p_tail) / LOG2

    total_H += H_pos
    pos_entropy.append(H_pos)

avg_H = total_H / N
print(f"Observed tokens: {N}")
print(f"Top-k entropy  (k={TOP_K}): {total_H:.6f} bits")
print(f"Average bits/token         : {avg_H:.6f} bits/token")

# ─────────────── sliding-window entropy ───────────────────────
if W <= 0:
    sys.exit("Window size W must be positive.")
if N < W:
    print(
        f"\nWindow size W={W} exceeds sequence length N={N}; "
        "skipping sliding-window analysis."
    )
else:
    # running window sum for O(N) computation
    window_sum = sum(pos_entropy[:W])
    windows = [(window_sum / W, 0)]  # (avg, start_idx)

    for start in range(1, N - W + 1):
        window_sum += pos_entropy[start + W - 1] - pos_entropy[start - 1]
        windows.append((window_sum / W, start))

    # pick top-M windows with largest average entropy
    top_windows = heapq.nlargest(TOP_M, windows, key=lambda x: x[0])

    print(
        f"\nTop {len(top_windows)} windows (size W={W}) "
        f"with highest average entropy:"
    )
    for rank, (avg_w, start) in enumerate(top_windows, 1):
        end = start + W - 1
        snippet = "".join(tok_infos[i].token for i in range(start, end + 1))
        print(
            f"{rank:>2}. [{start:>3}–{end:>3}]  "
            f'avg H = {avg_w:.4f} bits/token  →  "{snippet}"'
        )

# ─────────────── diagnostics: first / last tokens ─────────────
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
