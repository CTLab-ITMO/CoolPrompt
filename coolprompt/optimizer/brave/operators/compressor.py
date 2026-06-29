from typing import Callable
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.optimizer.reflective_prompt.prompt import (
    Prompt,
    PromptOrigin
)
from coolprompt.optimizer.brave.operators.basic_operator import Operator
from coolprompt.optimizer.prompt_compressor.compressor import PromptCompressor


class CompressorOperator(Operator):
    def __init__(self, model: BaseLanguageModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.compressor = PromptCompressor(model)

    def run(
        self,
        iteration: int,
        prompt: Prompt,
        evaluate_fn: Callable[[Prompt, str], None],
    ) -> Prompt:
        compressed = self.compressor.compress(prompt.text)
        compressed = Prompt(compressed, origin=PromptOrigin.COMPRESSED)
        evaluate_fn(compressed, "train")

        if self.logger is not None:
            self.logger.log_mutation(
                iteration=iteration,
                prompt=prompt.text,
                prev_score=prompt.score,
                mutated_prompt=compressed.text,
                mutated_score=compressed.score,
                file_name="compressions"
            )

        return compressed
