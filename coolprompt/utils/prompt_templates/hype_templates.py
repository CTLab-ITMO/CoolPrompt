HYPE_PROMPT_TEMPLATE = """You are an expert prompt engineer. Your only task is to generate ONE hypothetical instructive prompt that helps a large language model answer the given query.

### HARD CONSTRAINTS - VIOLATION WILL CAUSE FAILURE ###
1. LANGUAGE:
   - Output MUST be in the EXACT SAME LANGUAGE as the query.
   - NEVER translate or switch languages.
2. CONTENT:
   - Output ONLY the hypothetical instructive prompt - do NOT answer the question.
   - NEVER include explanations or meta-commentary.
   - NEVER answer the original query directly.
   - If the original query contains any code snippets, you must include it in final prompt.
   - The hypothetical prompt must solve the same task as the original query.
   - DO NOT answer the query, only generate the hypothetical instruction prompt.
3. TECHNICAL PRESERVATION:
   - Code blocks must be preserved with original syntax and formatting.
   - Variables, placeholders ({{var}}), and technical terms kept unchanged.
   - Markdown and special formatting replicated precisely.

### YOUR OUTPUT FORMAT (strictly one prompt wrapped in tags!) ###
[PROMPT_START]<your hypothetical instructive prompt here>[PROMPT_END]

### INPUT ###
Query: {QUERY}

### OUTPUT ###

"""

CLASSIFICATION_TASK_TEMPLATE_HYPE = """{PROMPT}

### HARD CONSTRAINTS ###
This is an automated evaluation. Your output will be parsed by a script. Any deviation from the required format will result in failure.

1. OUTPUT FORMAT:
   - Output ONLY the final answer in the format: `<ans>LABEL</ans>`
   - LABEL MUST be EXACTLY one item from the list: [{LABELS}]
    - DO NOT include any explanation, reasoning, or extra text.
    - DO NOT include any meta-level commentary (e.g., "Sure", "Here is your answer", "Let's tackle this", "To answer this question", etc).
    - DO NOT modify the tag or the label format.
    - DO NOT repeat the answer or output multiple <ans> tags.

2. STOP CONDITION:
    - IMMEDIATELY stop generating after `</ans>`.
    - DO NOT output anything after `</ans>`.

3. FAILURE CONDITION:
    - If you break any of the above constraints, the output will be considered INVALID and REJECTED.

4. FORMAT EXAMPLE:
    1. Labels are [(A), (B), (C)] and you chose first answer  
       Output will be: <ans>(A)</ans>
    2. Labels are [A, B, C] and you chose the first answer  
       Output will be: <ans>A</ans>

### INPUT ###
{INPUT}

### RESPONSE ###
"""
GENERATION_TASK_TEMPLATE_HYPE = """{PROMPT}
### HARD CONSTRAINTS ###
This is an automated evaluation. Your output will be parsed by a script. Any deviation from the required format will result in failure.

1. OUTPUT FORMAT:
    - Output ONLY the answer to the question or task specified in the prompt.
    - Output ONLY the generated content inside `<ans>` tags: `<ans>GENERATED_TEXT</ans>`
    - NO redundant explanations or meta-commentary.
    - DO NOT include any meta-level commentary (e.g., "Sure", "Here is your answer", "Let's tackle this", "To answer this question", etc).
    - DO NOT explain your reasoning, unless explicitly required by the prompt itself.
    - DO NOT add extra introductory or concluding sentences unless they are part of the intended output.

2. STOP CONDITION:
    - Stop IMMEDIATELY after `</ans>`.
    - DO NOT generate any additional output or comments after the completion.
    - DO NOT write any examples of the original task after answering it.

3. FAILURE CONDITION:
    - If any of the above constraints are violated, the output will be considered INVALID and REJECTED.

### INPUT ###
{INPUT}

### RESPONSE ###
"""
