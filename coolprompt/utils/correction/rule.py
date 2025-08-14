from abc import ABC
from typing import Any
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.utils.prompt_templates.correction_templates import (
    ANSWER_EXTRACTION_TEMPLATE,
    LANGUAGE_DETECTION_TEMPLATE,
    TRANSLATION_TEMPLATE,
    safe_template,
)


def extract_answer(text: str, start_tag: str, end_tag: str) -> str:
    """Extracts answer bracketed in `start_tag` and `end_tag` from the
    `text`."""

    return text[(text.rfind(start_tag) + len(start_tag)) : text.rfind(end_tag)]


class Rule(ABC):
    """Base class for rules which will be checked and fixed by a corrector."""

    @property
    def is_guaranteed_after_first_fix(self) -> bool:
        """Indicates whether the rule is guaranteed to pass check after first
        fix.

        Returns:
            bool: True if rule always pass check after first fix, False
                otherwise.
        """
        return False

    def check(self, prompt: str, **kwargs) -> tuple[bool, dict[str, Any]]:
        """Checks if the prompt follows the rule.

        Args:
            prompt (str): prompt to check.
            kwargs: other data explicit for the rule.
        Returns:
            result (tuple[bool, dict[str, Any]]): tuple of flag (correctness)
                and meta data for fixing.
        """
        pass

    def fix(self, prompt: str, meta: dict[str, Any]) -> str:
        """Fixes the prompt.

        Args:
            prompt (str): prompt to fix.
            meta (dict[str, Any]): meta data from the `check` function.
        Returns:
            result (str): fixed prompt.
        """
        pass


class LanguageRule(Rule):
    """The rule which checks if the final prompt and the start prompt are in
    the same languages."""

    def __init__(self, llm: BaseLanguageModel) -> None:
        """Initializes with LangChain language model."""
        self.llm = llm

    @property
    def is_guaranteed_after_first_fix(self):
        return True

    def check(
        self, final_prompt: str, start_prompt: str
    ) -> tuple[bool, dict[str, Any]]:
        """Checks if the final prompt and the start prompt are in the same
        languages.

        Args:
            final_prompt (str): enhanced prompt.
            start_prompt (str): original prompt.
        Returns:
            result (tuple[bool, dict[str, Any]]): tuple of flag (correctness)
                and meta data with the target language.
        """

        def detect_language(text: str) -> str:
            """Detects the provided text's language using the `llm` model."""

            template = safe_template(LANGUAGE_DETECTION_TEMPLATE, text=text)
            return extract_answer(self.llm.invoke(template), "<ans>", "</ans>")

        start_prompt_lang = detect_language(start_prompt)
        final_prompt_lang = detect_language(final_prompt)

        if start_prompt_lang != final_prompt_lang:
            return False, {
                "type": "translation",
                "to_lang": start_prompt_lang,
            }
        else:
            return True, {}

    def fix(self, final_prompt: str, meta: dict[str, Any]) -> str:
        """Performs a translation for `final_prompt` from its language to
        the start prompt's one via `llm` model.

        Args:
            final_prompt (str): enhanced prompt to fix.
            meta (dict[str, Any]): meta data with prompt languages.
        Returns:
            result (str): fixed prompt.
        """

        template = safe_template(
            TRANSLATION_TEMPLATE,
            target_lang=meta["to_lang"],
            user_prompt=final_prompt,
        )
        return extract_answer(self.llm.invoke(template), "<ans>", "</ans>")


class FormatRule(Rule):
    """If the final prompt has to be between tags, checks the format."""

    def __init__(self, llm: BaseLanguageModel) -> None:
        """Initializes with LangChain language model."""
        self.llm = llm

    @property
    def is_guaranteed_after_first_fix(self):
        return False

    def check(
        self, prompt: str, start_tag: str, end_tag: str, context: str
    ) -> tuple[bool, dict[str, Any]]:
        """Checks if the `prompt` is between tags.

        Args:
            prompt (str): prompt to check.
            start_tag (str): start tag (with brackets).
            end_tag (str): end tag (with brackets).
            context (str): context for which the `prompt` was produced.
        Returns:
            result (tuple[bool, dict[str, Any]]): tuple of flag (correctness)
                and meta data with context, start and end tags.
        """

        if start_tag in prompt and end_tag in prompt:
            return True, {}

        return False, {
            "type": "Extraction",
            "start_tag": start_tag,
            "end_tag": end_tag,
            "context": context,
        }

    def fix(self, prompt: str, meta: dict[str, Any]) -> str:
        """Performs a call to `llm` for extracting the final answer bracketed
        in tags.

        Args:
            prompt (str): prompt to fix.
            meta (dict[str, Any]): meta data with open and close tags.
        Returns:
            result (str): fixed prompt.
        """

        start_tag = meta["start_tag"]
        end_tag = meta["end_tag"]
        context = meta["context"]

        template = safe_template(
            ANSWER_EXTRACTION_TEMPLATE,
            start_tag=start_tag,
            end_tag=end_tag,
            context=context,
            thought=prompt,
        )

        return self.llm.invoke(template)
