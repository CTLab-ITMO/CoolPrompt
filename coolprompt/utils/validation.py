from langchain_core.language_models.base import BaseLanguageModel


def validate_model(model) -> None:
    """
    Validating the type of provided model

    Raises:
        TypeError: if model is not an instance of BaseLanguageModel
    """
    if not isinstance(model, BaseLanguageModel):
        raise TypeError(
            "Model should be instance of LangChain BaseLanguageModel"
        )
