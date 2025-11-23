from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage


class DeepEvalLangChainModel(DeepEvalBaseLLM):
    """DeepEval LLM wrapper for LangChain BaseLanguageModel."""

    def __init__(self, model: BaseLanguageModel):
        self.model = model

    def load_model(self) -> BaseLanguageModel:
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        result = chat_model.invoke(prompt)
        if isinstance(result, AIMessage):
            return (
                result.content
                if isinstance(result.content, str)
                else str(result.content)
            )
        return str(result)

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        result = await chat_model.ainvoke(prompt)
        if isinstance(result, AIMessage):
            return (
                result.content
                if isinstance(result.content, str)
                else str(result.content)
            )
        return str(result)

    def get_model_name(self) -> str:
        return "CoolPrompt DeepEval LangChain Model"