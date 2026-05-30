from typing import Optional, Type, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from coolprompt.utils.structured_schemas.language_model import (
    DeepEvalJudgeResponse,
)


class DeepEvalLangChainModel(DeepEvalBaseLLM):
    """DeepEval LLM wrapper for a LangChain ``BaseLanguageModel``.

    The wrapper exposes the ``DeepEvalBaseLLM`` interface (``generate`` /
    ``a_generate``) on top of an arbitrary LangChain chat model so that
    DeepEval metrics (e.g. ``GEval``) can drive the same model that the
    rest of CoolPrompt uses.

    Args:
        model: The underlying LangChain language model to delegate to.
        use_structured_output: If ``True``, calls to ``generate`` /
            ``a_generate`` are routed through
            ``model.with_structured_output(schema, method="json_schema")``
            instead of a plain ``invoke``. The ``schema`` argument
            forwarded by DeepEval (per the ``DeepEvalBaseLLM`` contract)
            is used as-is; if the caller does not provide one, the
            default :class:`coolprompt.utils.structured_schemas.\
language_model.DeepEvalJudgeResponse` schema is used and its
            ``response`` field is returned as a string to preserve the
            legacy unstructured contract. Defaults to ``False``
            (legacy behaviour: plain ``invoke`` + ``AIMessage`` →
            ``str`` extraction).
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
        """Synchronously generate a response for ``prompt``.

        Args:
            prompt: Prompt text to send to the underlying model.
            schema: Optional pydantic schema passed by DeepEval to
                request a structured response. Honoured only when
                :attr:`use_structured_output` is ``True``.

        Returns:
            * When :attr:`use_structured_output` is ``False``:
              the raw text response (legacy behaviour).
            * When :attr:`use_structured_output` is ``True`` **and** a
              ``schema`` is supplied: a populated pydantic instance of
              that schema.
            * When :attr:`use_structured_output` is ``True`` **and** no
              schema is supplied: the model is invoked with the
              fallback :class:`DeepEvalJudgeResponse` schema and only
              its ``response`` string field is returned, so DeepEval
              code expecting a ``str`` keeps working.
        """
        chat_model = self.load_model()
        if self.use_structured_output:
            if schema is not None:
                runner = self._structured_runner(schema)
                return runner.invoke(prompt)
            runner = self._structured_runner(DeepEvalJudgeResponse)
            parsed: DeepEvalJudgeResponse = runner.invoke(prompt)
            return parsed.response

        result = chat_model.invoke(prompt)
        return self._extract_text(result)

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """Asynchronous counterpart of :meth:`generate`."""
        chat_model = self.load_model()
        if self.use_structured_output:
            if schema is not None:
                runner = self._structured_runner(schema)
                return await runner.ainvoke(prompt)
            runner = self._structured_runner(DeepEvalJudgeResponse)
            parsed: DeepEvalJudgeResponse = await runner.ainvoke(prompt)
            return parsed.response

        result = await chat_model.ainvoke(prompt)
        return self._extract_text(result)

    def get_model_name(self) -> str:
        return "CoolPrompt DeepEval LangChain Model"
