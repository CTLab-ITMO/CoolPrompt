from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.prompt_compressor import PromptCompressor


class CompressorMethod(AutoPromptingMethod):
    def __init__(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        return_metadata: bool = False,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.return_metadata = return_metadata

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split=None,
        evaluator=None,
        problem_description=None,
        **kwargs,
    ):
        compressor = PromptCompressor(
            model=model,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            **kwargs,
        )

        result = compressor.compress(
            prompt=initial_prompt,
            return_metadata=self.return_metadata,
        )

        if self.return_metadata:
            return result.final_prompt

        return result

    def is_data_driven(self) -> bool:
        return False

    @property
    @override
    def name(self) -> str:
        return "compress"