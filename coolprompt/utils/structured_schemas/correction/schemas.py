from pydantic import BaseModel, Field


class LanguageDetectionResponse(BaseModel):
    """Response schema for language detection.

    Used by :func:`coolprompt.utils.language_detection.detect_language`
    to obtain a strict, single-field structured output from the LLM
    when ``use_structured_output=True``.

    Mirrors the JSON contract from ``LANGUAGE_DETECTION_TEMPLATE``:
    ``{"language_code": "XX"}`` or ``{"language_code": "XX-YY"}``.
    """

    language_code: str = Field(
        description=(
            "ISO 639-1 language code of the detected text "
            "(e.g. 'en', 'ru', 'zh-CN', 'pt-BR'). "
            "Use 5-character regional codes when the region is clearly "
            "specified or culturally important; otherwise use 2-character codes."
        )
    )


class TranslationResponse(BaseModel):
    """Response schema for prompt translation.

    Used by :class:`coolprompt.utils.correction.rule.LanguageRule`
    to obtain a strict, single-field structured output from the LLM
    when ``use_structured_output=True``.

    Mirrors the JSON contract from ``TRANSLATION_TEMPLATE``:
    ``{"translated_text": "<translated text>"}``.
    """

    translated_text: str = Field(
        description=(
            "The full translated text in the target language. "
            "All original formatting, spacing, punctuation, and line breaks "
            "must be preserved. Code blocks, variables, function names, URLs, "
            "technical terms, proper names, and any text already in the target "
            "language must not be translated."
        )
    )
