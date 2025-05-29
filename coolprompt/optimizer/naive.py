from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.utils.prompt_template import NAIVE_PROMPT_TEMPLATE


def naive_optimizer(model: BaseLanguageModel, prompt: str) -> str:
    """Rewrites prompt by injecting it into predefined template and querying LLM.

    Args:
        model: Any LangChain BaseLanguageModel instance.
        prompt: Input prompt to optimize.

    Returns:
        LLM-generated rewritten prompt.
    """
    template = NAIVE_PROMPT_TEMPLATE
    answer = model.invoke(template.replace("<PROMPT>", prompt)).strip()
    return answer[answer.find("<START>") + len("<START>") : answer.find("<END>")]
