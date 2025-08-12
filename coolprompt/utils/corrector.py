from abc import ABC
from langdetect import detect, DetectorFactory
from translate import Translator
from typing import Any


class Rule(ABC):
    """Base class for rules which will be checked and fixed by a corrector."""

    def check(
        self, final_prompt: str, start_prompt: str
    ) -> tuple[bool, dict[str, Any]]:
        """Checks if the final prompt follows the rule.

        Args:
            final_prompt (str): enhanced prompt.
            start_prompt (str): original prompt.
        Returns:
            result (tuple[bool, dict[str, Any]]): tuple of flag (correctness)
                and meta data for fixing.
        """
        pass

    def fix(self, final_prompt: str, meta: dict[str, Any]) -> str:
        """Fixes the final prompt.

        Args:
            final_prompt (str): enhanced prompt to fix.
            meta (dict[str, Any]): meta data (explicit for rule).
        Returns:
            result (str): fixed prompt.
        """
        pass


class LanguageRule(Rule):
    """The rule which checks if the final prompt and the start prompt are in
    the same languages."""

    def check(self, final_prompt, start_prompt):
        DetectorFactory.seed = 0

        start_prompt_lang = detect(start_prompt)
        final_prompt_lang = detect(final_prompt)

        if start_prompt_lang != final_prompt_lang:
            return False, {
                "type": "translation",
                "to_lang": start_prompt_lang,
                "from_lang": final_prompt_lang,
            }
        else:
            return True, {}

    def fix(self, final_prompt, meta):
        translator = Translator(
            to_lang=meta["to_lang"],
            from_lang=meta["from_lang"],
        )
        return translator.translate(final_prompt)


class Corrector:
    """Corrector class which implements a correction loop."""

    def __init__(self, rules: list[Rule]):
        """Initializes with the list of rules."""

        self.rules = rules

    def run(
        self, final_prompt: str, start_prompt: str, max_attempts: int = 3
    ) -> str:
        """Running a correction loop. All the rules will be checked
        and, if need to, fixed sequentially. Loop will end if all rules
        are correct or after `max_attempts` attempts.

        Args:
            final_prompt (str): enhanced prompt.
            start_prompt (str): original prompt.
            max_attempts (optional, int): number of attempts the loop will end
                after. Defaults to 3.
        Returns:
            result (str): corrected final prompt.
        """

        for _ in range(max_attempts):
            all_ok = True
            issues = []

            for rule in self.rules:
                ok, meta = rule.check(final_prompt, start_prompt)
                if not ok:
                    all_ok = False
                    issues.append((rule, meta))

            if all_ok:
                return final_prompt

            for rule, meta in issues:
                final_prompt = rule.fix(final_prompt, meta)

        return final_prompt
