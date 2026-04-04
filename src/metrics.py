"""
Calculates CER between OCR and GT.
"""

from rapidfuzz.distance import Levenshtein


def levenshtein_distance(ocr, ground_truth):
    return Levenshtein.distance(ocr, ground_truth)


def cer(ocr, ground_truth):
    assert len(ground_truth) != 0, "Cannot compute CER for empty GT text."
    edits_needed = levenshtein_distance(ocr, ground_truth)
    return edits_needed / len(ground_truth)
