import re
from typing import List, Tuple

from coolprompt.utils.logging_config import logger


FREEZE_START_TAG = "<freeze>"
FREEZE_END_TAG = "</freeze>"

FREEZE_PATTERN = re.compile(
    re.escape(FREEZE_START_TAG) + r"(.*?)" + re.escape(FREEZE_END_TAG),
    re.DOTALL
)


def validate_freeze_tags(prompt: str) -> None:
    if not prompt:
        return

    start_count = prompt.count(FREEZE_START_TAG)
    end_count = prompt.count(FREEZE_END_TAG)

    if start_count != end_count:
        raise ValueError(
            f"Found {start_count} opening tags "
            f"<freeze> but {end_count} closing tags </freeze>. "
            f"Each opening tag must have closing tag."
        )


def split_prompt(prompt: str) -> Tuple[str, str]:
    if not prompt:
        return "", ""

    validate_freeze_tags(prompt)

    frozen_parts = FREEZE_PATTERN.findall(prompt)
    optimizable_part = FREEZE_PATTERN.sub("", prompt)
    optimizable_part = re.sub(r"\s+", " ", optimizable_part).strip()

    if frozen_parts:
        frozen_part = " ".join(frozen_parts).strip()
        logger.debug(
            f"Found {len(frozen_parts)} frozen part(s). "
            f"Frozen content: {frozen_part[:100]}..."
        )
    else:
        frozen_part = ""

    return optimizable_part, frozen_part


def merge_prompt(optimized_part: str, frozen_part: str) -> str:
    if not frozen_part:
        return optimized_part

    if optimized_part:
        return f"{optimized_part} {frozen_part}"
    else:
        return frozen_part


def extract_frozen_parts(prompt: str) -> List[str]:
    if not prompt:
        return []

    validate_freeze_tags(prompt)

    return FREEZE_PATTERN.findall(prompt)


def remove_freeze_tags(prompt: str) -> str:
    if not prompt:
        return prompt

    validate_freeze_tags(prompt)

    result = FREEZE_PATTERN.sub(r"\1", prompt)
    return result.strip()


def has_freeze_tags(prompt: str) -> bool:
    if not prompt:
        return False

    return FREEZE_START_TAG in prompt and FREEZE_END_TAG in prompt
