LANGUAGE_DETECTION_TEMPLATE = (
    "You are a multilingual language detection expert. "
    "Analyze ONLY the natural language query (ignore code blocks/comments) "
    "and detect its primary language. Use ISO 639-1 format. Rules:\n"
    "1. Skip code snippets, markdown and special formatting entirely\n"
    "2. For mixed languages: choose dominant language by word count\n"
    "3. For equal distribution: use language of the first meaningful phrase\n"
    "4. Output the answer in JSON format\n\n"
    "EXAMPLES\n"
    "Example 1:\n"
    "Query: Hola, ¿cómo estás?\n"
    'Answer: {{"language_code": "es"}}\n\n'
    "Example 2:\n"
    "Query: def calculate():\n"
    "    # Эта функция вычисляет что-то важное\n"
    "    return result\n"
    'Answer: {{"language_code": "ru"}}\n\n'
    "Example 3:\n"
    "Query: Bonjour! Help to translate it to English - помогите разобраться.\n"
    'Answer: {{"language_code": "en"}}\n\n'
    "Example 4:\n"
    'Query: print("안녕하세요") # Korean greeting\n'
    'Answer: {{"language_code": "en"}}\n\n'
    "Example 5:\n"
    "Query: こんにちは！Это тестовое сообщение на двух языках.\n"
    'Answer: {{"language_code": "ru"}}\n\n'
    "--- END OF EXAMPLES ---\n"
    "CURRENT TASK\n"
    "Query: {text}\n"
    "Answer: "
)

TRANSLATION_TEMPLATE = (
    "You are an expert multilingual translator and programmer. "
    "Analyze the prompt and translate ONLY the relevant parts to "
    "the specified target language (ISO 639-1). Strict rules:\n"
    "1. Detect target language (to which you have to translate) "
    "from <to_lang=XX> prefix\n"
    "2. Identify and translate ONLY phrases that:\n"
    "   a) Explicitly request translation\n"
    "   b) Are in a different language than the target\n"
    "3. Preserve ALL code blocks, variables, technical terms, "
    "markdown and special formatting verbatim, "
    "do NOT remove them from the text\n"
    "4. Maintain original formatting including spaces, "
    "line breaks, and punctuation\n"
    "5. For mixed-language segments: translate only the "
    "non-target language portions\n"
    "6. Output the answer in valid JSON format."
    "7. Instead of escaping curly braces, use DOUBLE curly braces: "
    "{{{{example}}}} instead of {{example}}"
    "EXAMPLES:\n"
    "Example 1:\n"
    'Prompt: <to_lang=de> print("Этот {{text}} не трогаем") '
    "# Please translate this comment\n/**/\n"
    'Output: {{"translated_text": '
    '"print(\\"Этот {{{{text}}}} не трогаем\\")"}}'
    "# Bitte übersetzen Sie diesen Kommentar\\n/**/\n\n"
    "Example 2:\n"
    "Prompt: <to_lang=fr> 1. Russian: Привет! 2. "
    'English: Hello! 3. Code: print("こんにちは")\n'
    'Output: {{"translated_text": "1. Russian: Bonjour! 2. English: '
    'Bonjour! 3. Code: print(\\"こんにちは")\\"}}\n\n'
    "Example 3:\n"
    "Prompt: <to_lang=es> Explain the concept of 'polymorphism' "
    "in programming\n\n"
    'Output: {{"translated_text": "Explica el concepto de \\\'polymorphism\\\''
    ' en programación"}}\n\n'
    "--- END OF EXAMPLES ---\n\n"
    "CURRENT TASK\n"
    "Prompt: [to_lang={to_lang}] {user_prompt}\n\n"
    "Output: "
)
