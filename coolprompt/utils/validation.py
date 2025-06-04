from langchain_core.language_models.base import BaseLanguageModel


def validate_model(model) -> None:
    if not isinstance(model, BaseLanguageModel):
        raise TypeError(
            "Model should be instance of LangChain BaseLanguageModel"
        )
