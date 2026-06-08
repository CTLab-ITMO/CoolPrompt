from dirtyjson import DirtyJSONLoader
from typing import Tuple

from langchain_core.language_models.base import BaseLanguageModel


def extract_answer(
    answer: str, tags: Tuple[str, str], format_mismatch_label: int | str = -1
) -> str | int:
    """Extract label from model output string containing XML-style tags.

    Args:
        answer (str): Model output string potentially containing format tags
        tags (Tuple[str, str]): XML-style tags
        format_mismatch_label (int | str):
            label corresponding to parsing failure.
            Defaults to -1

    Returns:
        label (str | int): Extracted answer or format_mismatch_label
            if parsing fails
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


def safe_template(template: str, **kwargs) -> str:
    """Safely formats the `template` with vars from `kwargs`.

    Args:
        template (str): template string.
        kwargs: template's vers (maybe with '{', '}').
    Returns:
        str: `template` formatted with `kwargs`, where '{' and '}' escaped
            for safety.
    """

    escaped = {
        k: str(v).replace("{", "{{").replace("}", "}}")
        for k, v in kwargs.items()
    }
    return template.format(**escaped)


def extract_json(text: str) -> dict | list | None:
    """Extracts the first valid JSON (object or array) from the text.

    Args:
        text (str): text with JSON-like substrings.
    Returns:
        result (dict | list | None): dict or list from JSON or None
            (if no valid JSON substrings found).
    """

    if isinstance(text, dict):
        return text

    loader = DirtyJSONLoader(text)

    pos = 0
    while pos < len(text):
        # Find both { and [
        start_pos = text.find("{", pos)
        bracket_pos = text.find("[", pos)

        # Get earliest position
        if start_pos == -1 and bracket_pos == -1:
            break
        elif start_pos == -1:
            search_pos = bracket_pos
        elif bracket_pos == -1:
            search_pos = start_pos
        else:
            search_pos = min(start_pos, bracket_pos)

        try:
            result = loader.decode(start_index=search_pos)
            if isinstance(result, dict):
                return dict(result)
            elif isinstance(result, list):
                return list(result)
        except Exception:
            pass

        pos = search_pos + 1

    return None


def parse_assistant_response(answer: str) -> str:
    """Extracts the answer from the assistant's response.

    Args:
        answer (str): assistant's response. May contain special format and
            reasoning tokens (e.g. <|im_start|>, <think>).
    Returns:
        str: extracted answer or empty string if there is no final answer
            (the response is not completed).
    """

    if answer.startswith("<|im_start|>"):
        # Qwen output case
        start_tag = "<|im_start|>assistant\n"
        think_start = "<think>"
        think_end = "</think>"

        pos = answer.find(start_tag)
        if pos == -1:
            return ""

        answer_after = answer[pos + len(start_tag) :]

        think_pos = answer_after.find(think_start)
        if think_pos != -1:
            think_end_pos = answer_after.find(think_end)
            if think_end_pos == -1:
                return ""
            else:
                return answer_after[think_end_pos + len(think_end) :].strip()
        return answer_after.strip()
    else:
        return answer.strip()


from typing import Tuple


def get_model_answer_extracted(
    llm: BaseLanguageModel,
    prompt: str,
    n: int = 1,
    temperature=None,
):
    """Invoke a model and return parsed assistant text for one or many generations."""
    if temperature is not None:
        llm = llm.bind(temperature=temperature)

    if n == 1:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return parse_assistant_response(text)

    if hasattr(llm, "generate"):
        try:
            llm_n = llm.bind(n=n)
            result = llm_n.generate([prompt])
            gens = result.generations[0]

            outputs = []
            for g in gens:
                text = getattr(g, "text", str(g))
                outputs.append(parse_assistant_response(text))

            if len(outputs) >= n:
                return outputs[:n]
        except Exception:
            pass

    duplicated = [prompt] * n
    responses = llm.batch(duplicated)

    outputs = []
    for r in responses:
        text = r.content if hasattr(r, "content") else str(r)
        outputs.append(parse_assistant_response(text))
    outputs = list(dict.fromkeys(outputs))  # hard deduplication

    return outputs
