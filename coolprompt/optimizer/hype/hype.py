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
from coolprompt.utils.prompt_freezer import (
    split_prompt, merge_prompt, has_freeze_tags, remove_freeze_tags, extract_frozen_parts
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

    has_frozen = has_freeze_tags(prompt)
    
    if has_frozen:
        frozen_parts = extract_frozen_parts(prompt)
        logger.info(
            f"Found {len(frozen_parts)} frozen part(s) in prompt"
        )
        frozen_list = "\n".join([f"- \"{part}\"" for part in frozen_parts])
        frozen_context = (
            "### CONTEXT INFO ###\n"
            "The user has marked parts of the original query with <freeze>...</freeze> tags.\n"
            "The text between these tags represents hard constraints that MUST be preserved verbatim.\n"
            f"Frozen fragments:\n{frozen_list}\n\n"
            "### IMPORTANT INSTRUCTIONS ###\n"
            "1. Optimize the prompt while preserving ALL content between <freeze>...</freeze> tags EXACTLY as written.\n"
            "2. Remove the <freeze> and </freeze> tags from your output, but keep ALL frozen text fragments themselves.\n"
            "3. Maintain the brevity and structure of the original prompt - do NOT make it unnecessarily verbose.\n"
            "4. Include ALL frozen fragments - do NOT skip any of them.\n"
            "5. Do NOT duplicate any frozen fragment - include each one only once.\n"
            "6. Do NOT rewrite or paraphrase the frozen text - use it verbatim.\n"
            "7. Keep your output concise and focused - avoid adding excessive elaboration or redundant phrases.\n\n"
        )
    else:
        frozen_context = ""

    query = safe_template(
        HYPE_PROMPT_TEMPLATE,
        PROBLEM_DESCRIPTION=problem_description,
        QUERY=prompt,
        FROZEN_CONTEXT=frozen_context,
    )

    answer = get_model_answer_extracted(model, query)
    final_prompt = extract_answer(
        answer, INSTRUCTIVE_PROMPT_TAGS, format_mismatch_label=answer
    )
    
    final_prompt = remove_freeze_tags(final_prompt)

    logger.info("HyPE optimization completed")
    logger.debug(f"Raw HyPE output:\n{answer}")
    if has_frozen:
        logger.debug(f"Final prompt with frozen parts integrated:\n{final_prompt}")

    return final_prompt
