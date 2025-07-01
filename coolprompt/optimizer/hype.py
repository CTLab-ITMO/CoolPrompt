from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.utils.logging_config import logger
from coolprompt.utils.prompt_templates.hype_templates import (
    HYPE_PROMPT_TEMPLATE,
)


def hype_optimizer(model: BaseLanguageModel, prompt: str) -> str:
    """Rewrites prompt by injecting it
    into predefined template and querying LLM.

    Args:
        model (BaseLanguageModel): Any LangChain BaseLanguageModel instance.
        prompt (str): Input prompt to optimize.
    Returns:
        LLM-generated rewritten prompt.
    """
    logger.info('Running HyPE optimization...')
    logger.debug(f'Start prompt:\n{prompt}')
    template = HYPE_PROMPT_TEMPLATE
    start_tag, end_tag = "[PROMPT_START]", "[PROMPT_END]"
    answer = model.invoke(template.replace("<QUERY>", prompt)).strip()
    logger.info('HyPE optimization completed')
    return answer[
        answer.rfind(start_tag) + len(start_tag):answer.rfind(end_tag)
    ]
