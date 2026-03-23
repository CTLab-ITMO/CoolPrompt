"""APE proposer: generates prompt variations via paraphrasing."""

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.pe2.node import Node
from coolprompt.utils.parsing import (
    extract_answer,
    get_model_answer_extracted,
)
from coolprompt.utils.prompt_templates.ape_templates import (
    APE_PARAPHRASE_TEMPLATE,
)


class APEProposer:
    """Generates prompt variations by paraphrasing.

    Unlike PE2's proposer, APE ignores failure examples
    and simply paraphrases the current prompt.

    Args:
        model (BaseLanguageModel): LLM for paraphrasing.
        prompt_max_tokens (int): Max token budget hint
            for the new prompt.
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
        """Proposes a paraphrased prompt variation.

        Args:
            node (Node): Current beam node whose prompt
                is being varied.
            examples_str (str): Ignored by APE.
            full_template (str): Ignored by APE.
            batch_size (int): Ignored by APE.

        Returns:
            tuple[str, str]: (new_prompt, "paraphrase").
        """
        prompt = APE_PARAPHRASE_TEMPLATE.format(
            prompt=node.prompt,
            max_tokens=self.prompt_max_tokens,
        )
        answer = get_model_answer_extracted(self.model, prompt)

        new_prompt = extract_answer(
            answer,
            ("<prompt>", "</prompt>"),
            format_mismatch_label=node.prompt,
        )

        return new_prompt.strip(), "paraphrase"
