from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage

from coolprompt.utils.logging_config import logger
from coolprompt.utils.prompt_templates.hype_templates import (
    HYPE_PROMPT_TEMPLATE,
)
from coolprompt.utils.correction.corrector import correct
from coolprompt.utils.correction.rule import FormatRule


def hype_optimizer(model: BaseLanguageModel, prompt: str) -> str:
    """Rewrites prompt by injecting it
    into predefined template and querying LLM.

    Args:
        model (BaseLanguageModel): Any LangChain BaseLanguageModel instance.
        prompt (str): Input prompt to optimize.
    Returns:
        LLM-generated rewritten prompt.
    """
    logger.info("Running HyPE optimization...")
    logger.debug(f"Start prompt:\n{prompt}")
    query = HYPE_PROMPT_TEMPLATE.format(QUERY=prompt)
    start_tag, end_tag = "[PROMPT_START]", "[PROMPT_END]"
    answer = model.invoke(query)

    if isinstance(answer, AIMessage):
        answer = answer.content

    answer = correct(
        prompt=answer.strip(),
        rule=FormatRule(model),
        start_tag=start_tag,
        end_tag=end_tag,
        context=query,
    )

    logger.info("HyPE optimization completed")
    return answer[
        answer.rfind(start_tag) + len(start_tag) : answer.rfind(end_tag)
    ]
