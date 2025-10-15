from abc import ABC
from typing import Any
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.utils.prompt_templates.correction_templates import (
    TRANSLATION_TEMPLATE,
)
from coolprompt.utils.language_detection import detect_language
from coolprompt.utils.parsing import (
    extract_json,
    get_model_answer_extracted,
    safe_template,
)


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

        start_prompt_lang = detect_language(start_prompt, self.llm)
        final_prompt_lang = detect_language(final_prompt, self.llm)

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

        prompt = safe_template(
            TRANSLATION_TEMPLATE,
            user_prompt=final_prompt,
            to_lang=meta["to_lang"],
        )

        answer = get_model_answer_extracted(self.llm, prompt)

        result = extract_json(answer)

        return result["translated_text"] if result else final_prompt
