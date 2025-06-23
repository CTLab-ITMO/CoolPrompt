from typing import Tuple


def extract_answer(
    answer: str,
    tags: Tuple[str, str],
    format_mismatch_label: int | str = -1
) -> str | int:
    """Extract label from model output string containing XML-style tags.

    Args:
        answer (str): Model output string potentially containing format tags
        tags (Tuple[str, str]): XML-style tags
        format_mismatch_label (int | str):
            label corresponding to parsing failure.
            Defaults to -1

    Returns:
        str | int: Extracted answer or format_mismatch_label if parsing fails
    """

    start_tag, end_tag = tags
    start_idx = answer.rfind(start_tag)

    if start_idx == -1:
        return format_mismatch_label

    content_start = start_idx + len(start_tag)
    end_idx = answer.find(end_tag, content_start)

    if end_idx == -1:
        return format_mismatch_label

    label = answer[content_start:end_idx]
    return label
