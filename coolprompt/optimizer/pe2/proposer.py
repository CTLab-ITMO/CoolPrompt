"""PE2 proposer: generates refined prompts from failure analysis."""

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.pe2.node import Node
from coolprompt.utils.parsing import (
    extract_answer,
    get_model_answer_extracted,
)
from coolprompt.utils.prompt_templates.pe2_templates import (
    PE2_REASONING_TEMPLATE,
    PE2_REFINEMENT_TEMPLATE,
)


class Proposer:
    """Wraps LLM calls for PE2 prompt refinement.

    Args:
        model (BaseLanguageModel): LLM for reasoning and refinement.
        prompt_max_tokens (int): Max token budget hint for the new prompt.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        prompt_max_tokens: int = 300,
    ) -> None:
        self.model = model
        self.prompt_max_tokens = prompt_max_tokens

    def propose(
        self,
        node: Node,
        examples_str: str,
        full_template: str,
        batch_size: int,
    ) -> tuple[str, str]:
        """Proposes a refined prompt by analyzing failures.

        Args:
            node (Node): Current beam node whose prompt is being refined.
            examples_str (str): Formatted failure examples string.
            full_template (str): The full prompt template in use.
            batch_size (int): Number of failure examples shown.

        Returns:
            tuple[str, str]: (new_prompt, reasoning) where new_prompt is
                the refined prompt text and reasoning is the analysis.
        """
        reasoning_prompt = PE2_REASONING_TEMPLATE.format(
            batch_size=batch_size,
            prompt=node.prompt,
            full_template=full_template,
            examples=examples_str,
        )
        reasoning = get_model_answer_extracted(self.model, reasoning_prompt)

        refinement_prompt = PE2_REFINEMENT_TEMPLATE.format(
            reasoning=reasoning,
            prompt=node.prompt,
            max_tokens=self.prompt_max_tokens,
        )
        refinement_answer = get_model_answer_extracted(
            self.model, refinement_prompt
        )

        new_prompt = extract_answer(
            refinement_answer,
            ("<prompt>", "</prompt>"),
            format_mismatch_label=node.prompt,
        )

        return new_prompt.strip(), reasoning
