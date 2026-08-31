from typing import Callable
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.reflective_prompt.prompt import Prompt, PromptOrigin
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.prompt_compressor.compressor import PromptCompressor


class CompressorOperator(Operator):
    """Compress a prompt while preserving its intended behavior."""

    def __init__(self, model: BaseLanguageModel, **kwargs) -> None:
        """Create a prompt compressor backed by the supplied model.

        Args:
            model (BaseLanguageModel): model used for compression.
            **kwargs (Any): base-operator arguments such as ``logger``.
        """

        super().__init__(**kwargs)
        self.compressor = PromptCompressor(model)

    def run(
        self,
        iteration: int,
        prompt: Prompt,
        evaluate_fn: Callable[[Prompt, str], None],
    ) -> Prompt:
        """Compress and evaluate a prompt.

        Args:
            iteration (int): optimization iteration used for logging.
            prompt (Prompt): prompt to compress.
            evaluate_fn (Callable[[Prompt, str], None]): prompt evaluator.

        Returns:
            Prompt: evaluated compressed prompt, or a zero-scored failure
                placeholder when compression raises an exception.
        """

        try:
            compressed = self.compressor.compress(prompt.text)
            compressed = Prompt(compressed, origin=PromptOrigin.COMPRESSED)
            evaluate_fn(compressed, "train")
        except Exception:
            compressed = Prompt("failed to compress", origin=PromptOrigin.COMPRESSED)
            compressed.set_score(0)

        if self.logger is not None:
            self.logger.log_mutation(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=compressed.text,
                mutated_score=compressed.score,
                file_name="compressions",
            )

        return compressed
