from dirtyjson import DirtyJSONLoader
from typing import Tuple
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage


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


def extract_json(text: str) -> dict | None:
    """Extracts the first valid JSON with one text value from the `text`.

    Args:
        text (str): text with JSON-lke substrings.
    Returns:
        result (dict | None): dict from JSON or None
            (if no valid JSON substrings found).
    """

    if isinstance(text, dict):
        return text

    loader = DirtyJSONLoader(text)

    pos = 0
    while pos < len(text):
        start_pos = text.find("{", pos)
        if start_pos == -1:
            break
        try:
            return dict(loader.decode(start_index=start_pos))
        except:
            pos = start_pos + 1

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


def get_model_answer_extracted(llm: BaseLanguageModel, prompt: str) -> str:
    """Gets `llm`'s response for the `prompt` and extracts the answer.

    Args:
        llm (BaseLanguageModel): LangChain language model.
        prompt (str): prompt for the model.
    Returns:
        str: extracted answer or empty string if there is no final answer.
    """

    answer = llm.invoke(prompt)

    if isinstance(answer, AIMessage):
        answer = answer.content

    answer = parse_assistant_response(answer)

    return answer
