import re


def clip(x, left, right):
    if x < left:
        return left
    if x > right:
        return right
    return x


def mean(lst):
    return sum(lst) / len(lst)


def extract_number_from_text(text):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def normalize_text_for_exact_match(text):
    """Normalize text for exact match comparison.

    Args:
        text: Input text string

    Returns:
        Normalized text string (lowercased, stripped, whitespace normalized)
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
