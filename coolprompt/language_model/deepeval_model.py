from typing import Optional, Type, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel


class DeepEvalLangChainModel(DeepEvalBaseLLM):
    """DeepEval LLM wrapper for a LangChain ``BaseLanguageModel``.

    The wrapper exposes the ``DeepEvalBaseLLM`` interface (``generate`` /
    ``a_generate``) on top of an arbitrary LangChain chat model so that
    DeepEval metrics (e.g. ``GEval``) can drive the same model that the
    rest of CoolPrompt uses.

    Args:
        model: The underlying LangChain language model to delegate to.
        use_structured_output: If ``True`` **and** DeepEval supplies a
            pydantic ``schema`` to ``generate`` / ``a_generate`` (per
            the ``DeepEvalBaseLLM`` contract), the call is routed
            through ``model.with_structured_output(schema,
            method="json_schema")``. Otherwise the wrapper falls back
            to the legacy plain ``invoke`` + ``AIMessage`` → ``str``
            extraction. Defaults to ``False`` (always legacy
            behaviour).
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        use_structured_output: bool = False,
    ):
        self.model = model
        self.use_structured_output = use_structured_output

    def load_model(self) -> BaseLanguageModel:
        return self.model

    @staticmethod
    def _extract_text(result) -> str:
        """Coerce a LangChain ``invoke`` result into a plain string."""
        if isinstance(result, AIMessage):
            return (
                result.content
                if isinstance(result.content, str)
                else str(result.content)
            )
        return str(result)

    def _structured_runner(self, schema: Type[BaseModel]):
        """Return the LangChain runnable with structured output bound."""
        return self.model.with_structured_output(
            schema, method="json_schema"
        )

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """Generate a synchronous text response for DeepEval.

        Args:
            prompt: Prompt text to send to the underlying model.
            schema: Optional pydantic schema passed by DeepEval to
                request a structured response. Honoured only when
                :attr:`use_structured_output` is ``True``.

        Returns:
            * When :attr:`use_structured_output` is ``True`` **and** a
              ``schema`` is supplied: a populated pydantic instance of
              that schema.
            * Otherwise: the raw text response (legacy behaviour).
        """
        chat_model = self.load_model()
        if self.use_structured_output and schema is not None:
            runner = self._structured_runner(schema)
            return runner.invoke(prompt)

        result = chat_model.invoke(prompt)
        return self._extract_text(result)

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """Generate an asynchronous text response for DeepEval."""
        chat_model = self.load_model()
        if self.use_structured_output and schema is not None:
            runner = self._structured_runner(schema)
            return await runner.ainvoke(prompt)

        result = await chat_model.ainvoke(prompt)
        return self._extract_text(result)

    def get_model_name(self) -> str:
        return "CoolPrompt DeepEval LangChain Model"
