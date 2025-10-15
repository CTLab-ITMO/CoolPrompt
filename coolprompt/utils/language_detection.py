import langdetect
from coolprompt.utils.prompt_templates.correction_templates import (
    LANGUAGE_DETECTION_TEMPLATE,
)
from coolprompt.utils.parsing import (
    extract_json,
    get_model_answer_extracted,
    safe_template,
)
from langchain_core.language_models.base import BaseLanguageModel


def detect_language(text: str, llm: BaseLanguageModel) -> str:
    """Detects the provided text's language using the LangChain language
    model.

    Args:
        text (str): text for language detection.
        llm (BaseLanguageModel): LangChain language model.
    Returns:
        str: `text`'s language code in ISO 639-1 format.
    """

    prompt = safe_template(LANGUAGE_DETECTION_TEMPLATE, text=text)

    answer = get_model_answer_extracted(llm, prompt)

    result = extract_json(answer)

    return result["language_code"] if result else langdetect.detect(text)
