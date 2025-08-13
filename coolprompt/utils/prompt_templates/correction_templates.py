def safe_template(template: str, **kwargs) -> str:
    """Safely formats the `template` with vars from `kwargs`."""
    escaped = {
        k: str(v).replace("{", "{{").replace("}", "}}")
        for k, v in kwargs.items()
    }
    return template.format(**escaped)


LANGUAGE_DETECTION_TEMPLATE = """You are a multilingual language detection expert. Analyze ONLY the natural language query (ignore code blocks/comments) and detect its primary language. Use ISO 639-1 format inside <ans></ans>. Rules:
1. For mixed languages: choose dominant language by word count
2. For equal distribution: use language of the first meaningful phrase
3. Skip code snippets entirely
4. Output ONLY <ans>code</ans> with no explanations

EXAMPLES:
User: Hola, ¿cómo estás?
Assistant: <ans>es</ans>

User: def calculate():
    # Эта функция вычисляет что-то важное
    return result
Assistant: <ans>ru</ans>

User: Bonjour! I need help with this - помогите разобраться.
Assistant: <ans>en</ans>

User: print("안녕하세요") # Korean greeting
Assistant: <ans>en</ans>

User: こんにちは！Это тестовое сообщение на двух языках.
Assistant: <ans>ru</ans>

CURRENT TASK:
User: {text}
Assistant: """


TRANSLATION_TEMPLATE = """You are an expert multilingual translator and programmer. Analyze the prompt and translate ONLY the relevant parts to the specified target language (ISO 639-1). Rules:
1. Detect target language from [lang=XX] prefix
2. Identify and translate ONLY phrases that:
   a) Explicitly request translation
   b) Are in a different language than the target
3. Preserve ALL code blocks, variables, and technical terms verbatim
4. Maintain original formatting including spaces, line breaks, and punctuation
5. For mixed-language segments: translate only the non-target language portions
6. Output the final result wrapped in <ans></ans> tags

EXAMPLES:
User: [lang=en] Переведи, пожалуйста, на русский "die Katze"
Assistant: <ans>Please translate to Russian "die Katze"</ans>

User: [lang=ru] This sentence should be translated: А это уже на русском
Assistant: <ans>Это предложение должно быть переведено: А это уже на русском</ans>

User: [lang=de] print("Этот текст не трогаем") # Please translate this comment
Assistant: <ans>print("Этот текст не трогаем") # Bitte übersetzen Sie diesen Kommentar</ans>

User: [lang=fr] 1. Russian: Привет! 2. English: Hello! 3. Code: print("こんにちは")
Assistant: <ans>1. Russian: Bonjour! 2. English: Bonjour! 3. Code: print("こんにちは")</ans>

User: [lang=es] Explain the concept of "polymorphism" in programming
Assistant: <ans>Explica el concepto de "polymorphism" en programación</ans>

CURRENT TASK:
User: [lang={target_lang}] {user_prompt}
Assistant: """

ANSWER_EXTRACTION_TEMPLATE = """You are an output extraction expert. Your task is to complete the assistant's response by strictly wrapping the final answer in the specified tags. Rules:
1. Analyze the original context and assistant's thought process
2. Identify the core answer that should be placed between {start_tag} and {end_tag}
3. If the answer is already in the tags - extract it exactly
4. If partially formed - complete it appropriately
5. If missing - generate based on thought and context
6. Output ONLY the final answer wrapped in {start_tag} and {end_tags} without any explanations

EXAMPLES:

Example 1:
<CONTEXT START>
Translate 'Hello' to Spanish. Use <tr> tags
<CONTEXT END>

<THOUGHT START>
Hola means Hello in Spanish. The translation should be Hola
<THOUGHT END>

Start Tag: "<tr>"
End Tag: "</tr>"
Output: <tr>Hola</tr>

---

Example 2:
<CONTEXT START>
Count from 1 to 3 in <count> tags
<CONTEXT END>

<THOUGHT START>
I should count: 1, 2, 3. <count>1,2
<THOUGHT END>

Start Tag: "<count>"
End Tag: "</count>"
Output: <count>1, 2, 3</count>

---

CURRENT TASK:
<CONTEXT START>
{context}
<CONTEXT END>

<THOUGHT START>
{thought}
<THOUGHT END>

Start Tag: "{start_tag}"
End Tag: "{end_tag}"
Output: """
