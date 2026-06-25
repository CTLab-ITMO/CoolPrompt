import langdetect
from coolprompt.utils.prompt_templates.correction_templates import (
    LANGUAGE_DETECTION_TEMPLATE,
)
from coolprompt.utils.parsing import (
    extract_json,
    get_model_answer_extracted,
    safe_template,
)
from coolprompt.utils.structured_schemas.correction import LanguageDetectionResponse
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage


def detect_language(
    text: str, llm: BaseLanguageModel, use_structured_output: bool = False
) -> str:
    """Detects the provided text's language using the LangChain language
    model.

    Args:
        text (str): text for language detection.
        llm (BaseLanguageModel): LangChain language model.
        use_structured_output (bool): if True, the LLM is queried via
            ``llm.with_structured_output(...)`` using
            :class:`~coolprompt.utils.structured_schemas.correction.LanguageDetectionResponse`;
            otherwise a plain ``invoke()`` is performed and the JSON payload
            is parsed from the raw text response.
    Returns:
        str: `text`'s language code in ISO 639-1 format.
    """

    prompt = safe_template(LANGUAGE_DETECTION_TEMPLATE, text=text)

    if use_structured_output:
        structured_model = llm.with_structured_output(
            schema=LanguageDetectionResponse, method="json_schema"
        )
        output = structured_model.invoke(prompt)
        if isinstance(output, AIMessage):
            output = output.content
        try:
            return output.language_code
        except Exception:
            return output["language_code"]

    answer = get_model_answer_extracted(llm, prompt)

    result = extract_json(answer)

    return result["language_code"] if result else langdetect.detect(text)
