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

ANSWER_EXTRACTION_TEMPLATE = (
    "You are an output extraction expert. Your task is to complete "
    "the assistant's response by strictly wrapping the final answer "
    "in the specified tags. Rules\n:"
    "1. Analyze the original context and assistant's response\n"
    "2. Identify the core answer that should be placed between "
    "start tag and end tag\n"
    "3. If the answer is already in the tags - extract it exactly\n"
    "4. If partially formed - complete it appropriately\n"
    "5. If missing - generate based on response and context\n"
    "6. Output ONLY the final answer wrapped in start tag and end tag "
    "without any explanations\n\n"
    "EXAMPLES:\n"
    "Example 1:\n"
    "Context: Translate 'Hello' to Spanish. Use <tr_start>, <tr_end> tags\n"
    "Response: Hola means Hello in Spanish. The translation should be Hola\n"
    "Start Tag: '<tr_start>'\n"
    "End Tag: '<tr_end>'\n"
    "Extracted answer: <tr_start>Hola<tr_end>\n\n"
    "Example 2:\n"
    "Context: Count from 1 to 3 in <count_start>, <count_end> tags\n"
    "Response: I should count: 1, 2, 3. <count_start>1,2\n"
    "Start Tag: '<count_start>'\n"
    "End Tag: '<count_end>'\n"
    "Extracted answer: <count_start>1, 2, 3</count_end>\n\n"
    "CURRENT TASK:\n"
    "Context:\n"
    "{context}\n\n"
    "Response:\n"
    "{response}\n\n"
    "Start Tag: '{start_tag}'\n"
    "End Tag: '{end_tag}'\n\n"
    "{format_instructions}\n"
)
