from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.utils.logging_config import logger
from coolprompt.utils.prompt_templates.hype_templates import (
    HYPE_PROMPT_TEMPLATE,
)
from coolprompt.utils.parsing import (
    extract_answer,
    get_model_answer_extracted,
    safe_template,
)

INSTRUCTIVE_PROMPT_TAGS = ("[PROMPT_START]", "[PROMPT_END]")


def hype_optimizer(
    model: BaseLanguageModel, prompt: str, problem_description: str
) -> str:
    """Rewrites prompt by injecting it
    into predefined template and querying LLM.

    Args:
        model (BaseLanguageModel): Any LangChain BaseLanguageModel instance.
        prompt (str): Input prompt to optimize.
        problem_description (str): Brief description of the task, explaining
            its domain.
    Returns:
        str: LLM-generated rewritten prompt.
    """

    logger.info("Running HyPE optimization...")
    logger.debug(f"Start prompt:\n{prompt}")

    query = safe_template(
        HYPE_PROMPT_TEMPLATE,
        PROBLEM_DESCRIPTION=problem_description,
        QUERY=prompt,
    )

    answer = get_model_answer_extracted(model, query)

    logger.info("HyPE optimization completed")
    logger.debug(f"Raw HyPE output:\n{answer}")

    return extract_answer(
        answer, INSTRUCTIVE_PROMPT_TAGS, format_mismatch_label=answer
    )
