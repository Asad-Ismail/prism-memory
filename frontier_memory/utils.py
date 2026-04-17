from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "now",
    "of",
    "on",
    "or",
    "our",
    "she",
    "still",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "with",
    "you",
    "your",
}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s']", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize(text: str, *, keep_stopwords: bool = False) -> List[str]:
    tokens = normalize_text(text).split()
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = normalize_text(text)
    return any(normalize_text(phrase) in lowered for phrase in phrases)


def extract_numbers(text: str) -> List[int]:
    return [int(value) for value in re.findall(r"\d+", text)]


def extract_capitalized_phrases(text: str) -> List[str]:
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return dedupe_preserve_order(candidates)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def beta_mean(alpha: float, beta: float, success_count: int, failure_count: int) -> float:
    return (alpha + success_count) / (alpha + beta + success_count + failure_count)


def geometric_mean(values: Sequence[float]) -> float:
    filtered = [max(value, 1e-9) for value in values]
    if not filtered:
        return 0.0
    return math.exp(sum(math.log(value) for value in filtered) / len(filtered))


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
