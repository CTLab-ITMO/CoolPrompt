from typing import override

from coolprompt.optimizer.apmethod import AutoPromptingMethod
from coolprompt.optimizer.prompt_compressor import PromptCompressor


class CompressorMethod(AutoPromptingMethod):
    """Prompt compression method for auto‑prompting.

    This method uses a `PromptCompressor` to shorten an initial prompt
    while preserving essential information. It can optionally return
    compression metadata.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        return_metadata: bool = False,
    ):
        """Initialize the CompressorMethod.

        Args:
            system_prompt (str | None): Optional system‑level instruction
                for the compression model. Defaults to None.
            user_prompt (str | None): Optional user‑level instruction
                for the compression model. Defaults to None.
            return_metadata (bool): If True, the `optimize` method will
                return only the final compressed prompt (string) instead
                of the full result object. Defaults to False.
        """
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
        """Compress the initial prompt using the PromptCompressor.

        Args:
            model: The language model used for prompt compression.
            initial_prompt (str): The prompt text to compress.
            dataset_split: Unused by this method. Defaults to None.
            evaluator: Unused by this method. Defaults to None.
            problem_description: Unused by this method. Defaults to None.
            **kwargs: Additional keyword arguments passed to the
                `PromptCompressor` constructor.

        Returns:
            If `self.return_metadata` is True, returns the compressed prompt
            as a string. Otherwise, returns the full result object from
            `compressor.compress()`.
        """
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
