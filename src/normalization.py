"""
Shared pipeline of functions applied to OCR and GT to maintain fair and equal CER calculation.
"""

import re

from utils import load_cache_json
from loader import load_text_pair
from metrics import cer


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (spaces/newlines/tabs) to single spaces and strip ends."""
    return " ".join(text.split()).strip()


def normalize_quotes_and_dashes(text: str) -> str:
    """
    Normalize typographic quotes/dashes to simple ASCII forms.
    Safe to run on both GT and OCR to make comparisons stable.
    """
    # quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("„", '"').replace("‟", '"')
    text = text.replace("``", '"').replace("''", '"')

    text = text.replace("`", "'")  # backtick -> apostrophe
    text = text.replace("‘", "'").replace("’", "'")

    # dashes: normalize em/en/triple-hyphen to a single hyphen
    text = text.replace("---", "-").replace("—", "-").replace("–", "-")

    return text


def strip_punctuation(text: str) -> str:
    """
    Remove ASCII punctuation. Use only if CER should ignore punctuation.
    If you want to preserve apostrophes/hyphens, adjust the regex.
    """
    return re.sub(r"[^\w\s]", "", text)


def lowercase(text: str) -> str:
    """Lowercase text (use if CER should be case-insensitive)."""
    return text.lower()


def normalize_text(ocr, ground_truth, normalization_type):
    if normalization_type == "none":
        return ocr, ground_truth

    if normalization_type == "all":
        return normalize_text_all(ocr, ground_truth)

    if normalization_type == "interactive":
        user_input = input(
            """Enter the type of normalization to apply to both ocr and gt ('q' to quit):
    1. Whitespaces
    2. Quotes and dashes
    3. Punctuation
    4. Lowercase\nEntry: """
        )
        while user_input != "q":
            if not user_input.isdigit() or int(user_input) < 1 or int(user_input) > 4:
                print("Invalid option. You should enter a number from 1 to 4")
                user_input = input(
                    """Enter the type of normalization to apply to both ocr and gt ('q' to quit):
    1. Whitespaces
    2. Quotes and dashes
    3. Punctuation
    4. Lowercase\nEntry: """
                )
                continue
            user_input = int(user_input)
            if user_input == 1:
                ocr = normalize_whitespace(ocr)
                ground_truth = normalize_whitespace(ground_truth)
            elif user_input == 2:
                ocr = normalize_quotes_and_dashes(ocr)
                ground_truth = normalize_quotes_and_dashes(ground_truth)
            elif user_input == 3:
                ocr = strip_punctuation(ocr)
                ground_truth = strip_punctuation(ground_truth)
            else:
                ocr = lowercase(ocr)
                ground_truth = lowercase(ground_truth)

            user_input = input(
                """Enter the type of normalization to apply to both ocr and gt ('q' to quit):
    1. Whitespaces
    2. Quotes and dashes
    3. Punctuation
    4. Lowercase\nEntry: """
            )
        return ocr, ground_truth


def normalize_text_all(ocr, ground_truth):
    ocr = normalize_whitespace(ocr)
    ground_truth = normalize_whitespace(ground_truth)
    ocr = normalize_quotes_and_dashes(ocr)
    ground_truth = normalize_quotes_and_dashes(ground_truth)
    ocr = strip_punctuation(ocr)
    ground_truth = strip_punctuation(ground_truth)
    ocr = lowercase(ocr)
    ground_truth = lowercase(ground_truth)
    return ocr, ground_truth


def main():
    page_id = 3206244093
    cache_key = "3206244093_gpt-4o_5_1"

    cache = load_cache_json()
    generated_text = cache[cache_key]["transcript"]

    _, ground_truth_text = load_text_pair(page_id)

    _cer = cer(generated_text, ground_truth_text)
    compare = f"Generated:\n{generated_text}\n\nGround Truth:\n{ground_truth_text}\n\nCER:\n{_cer}"

    with open("compare.txt", "w") as file:
        file.write(compare)
    return


if __name__ == "__main__":
    main()
