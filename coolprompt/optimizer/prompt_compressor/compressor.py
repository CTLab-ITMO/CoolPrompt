from typing import List, Optional, Union, override

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.utils.logging_config import logger
from coolprompt.utils.prompt_templates.compress_templates import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)


class CompressedPromptResponse(BaseModel):
    """Structure for LLM answer."""

    reasoning: str = Field(
        description="Анализ задачи и вопроса в исходном промпте"
    )
    prompt_input_context: str = Field(
        description="Выделенный входной контекст задачи в одном предложении"
    )
    prompt_task: str = Field(description="Выделенное предложение самой задачи")
    final_prompt: str = Field(description="Финальный сжатый промпт")


class PromptCompressor:
    """
    Prompt compressor using LLM and structured output.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model: LangChain language model.
            system_prompt: System prompt for compression (if None, the default is used).
            user_prompt: User query template with {prompt} placeholder.
        """
        self.model = model
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.user_prompt = user_prompt or USER_PROMPT

        self.structured_model = model.with_structured_output(
            CompressedPromptResponse, method="json_schema"
        )

    def _build_messages(self, prompt: str) -> List[dict]:
        """Create LLM input in the form of a list of messages."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(prompt=prompt)},
        ]

    def compress(
        self, prompt: str, return_metadata: bool = False
    ) -> Union[str, CompressedPromptResponse]:
        """
        Compress a single prompt synchronously.

        Args:
            prompt: Original prompt.
            return_metadata: If True, returns the full CompressedPromptResponse object,
                             otherwise only the final_prompt string.

        Returns:
            Compressed prompt (string) or full object with metadata.
        """
        messages = self._build_messages(prompt)
        response = self.structured_model.invoke(messages)

        logger.debug(
            f"Compressed prompt from '{prompt[:50]}...' -> '{response.final_prompt[:50]}...'"
        )
        return response if return_metadata else response.final_prompt


class CompressorMethod(AutoPromptingMethod):
    """Prompt compression method for auto‑prompting."""

    def __init__(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        return_metadata: bool = False,
    ) -> None:
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
        *,
        use_structured_output: bool = False,
        **kwargs,
    ):
        """Compress ``initial_prompt``.

        Note:
            :class:`PromptCompressor` is intrinsically built on top of
            ``with_structured_output`` and cannot operate without it.
            The ``use_structured_output`` flag is accepted here for
            interface uniformity with other methods, but passing ``False``
            raises ``NotImplementedError`` because the compressor does
            not support a non-structured execution path.

        Raises:
            NotImplementedError: If ``use_structured_output`` is ``False``.
        """
        if not use_structured_output:
            raise NotImplementedError(
                "PromptCompressor is built on top of structured output "
                "and cannot run with use_structured_output=False"
            )

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

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
        *,
        use_structured_output: bool = False,
    ) -> str:
        mc = ctx.config.get("method", {})
        method = CompressorMethod(
            system_prompt=mc.get("system_prompt", self.system_prompt),
            user_prompt=mc.get("user_prompt", self.user_prompt),
            return_metadata=mc.get("return_metadata", False),
        )
        return method.optimize(
            ctx.model,
            start_prompt,
            use_structured_output=use_structured_output,
        )

    def is_data_driven(self) -> bool:
        return False

    @property
    @override
    def name(self) -> str:
        return "compress"
