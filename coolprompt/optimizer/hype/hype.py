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
from coolprompt.utils.prompt_freezer import split_prompt, merge_prompt

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

    optimizable_part, frozen_part = split_prompt(prompt)

    if frozen_part:
        logger.info(
            "Found frozen parts in prompt. Optimizing only optimizable part."
        )
        frozen_context = (
            "### CONTEXT INFO ###\n"
            "The user has hard-coded a constraint that will be appended AFTER your optimized prompt.\n"
            f'Frozen Suffix: "{frozen_part}"\n\n'
            "### IMPORTANT INSTRUCTIONS ###\n"
            "1. Ensure your new prompt flows logically into the Frozen Suffix.\n"
            "2. CRITICAL: The Frozen Suffix is ALREADY attached. If you repeat its content, the final prompt will have duplicates. THIS IS FORBIDDEN.\n"
            "3. Your job is to write the PREFIX that leads up to the suffix, NOT to rewrite the suffix itself.\n"
            "4. DO NOT include the Frozen Suffix in your output.\n"
            "5. Output ONLY the optimized version of the TASK_PROMPT.\n\n"
        )
    else:
        frozen_context = ""

    query = safe_template(
        HYPE_PROMPT_TEMPLATE,
        PROBLEM_DESCRIPTION=problem_description,
        QUERY=optimizable_part if optimizable_part else prompt,
        FROZEN_CONTEXT=frozen_context,
    )

    answer = get_model_answer_extracted(model, query)
    optimized_part = extract_answer(
        answer, INSTRUCTIVE_PROMPT_TAGS, format_mismatch_label=answer
    )

    final_prompt = merge_prompt(optimized_part, frozen_part)

    logger.info("HyPE optimization completed")
    logger.debug(f"Raw HyPE output:\n{answer}")
    if frozen_part:
        logger.debug(f"Final prompt with frozen parts:\n{final_prompt}")

    return final_prompt
