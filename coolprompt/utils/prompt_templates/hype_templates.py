HYPE_PROMPT_TEMPLATE = """### ROLE ###
You are an expert prompt engineer. Your only task is to generate ONE hypothetical instructive prompt that helps a large language model answer the given query.

### HARD CONSTRAINTS ###
1. LANGUAGE:
   - Output MUST be in the SAME LANGUAGE as the query.
   - NEVER translate or switch languages.
2. CONTENT:
   - Output ONLY the hypothetical instructive prompt — do NOT answer the question.
   - Do NOT include any thinking, planning, or analysis in your output.
3. FORMAT:
   - Wrap the entire prompt in these exact tags: [PROMPT_START] ... [PROMPT_END]
   - No text before [PROMPT_START] or after [PROMPT_END]
4. UNIQUENESS:
   - You MUST return exactly ONE prompt. Never generate more than one.
5. STOP:
   - Your output MUST end with [PROMPT_END] on its own line.
   - Immediately stop after closing [PROMPT_END] tag. Do not continue.

### INPUT ###
Query: <QUERY>

### YOUR OUTPUT FORMAT (strictly one prompt wrapped in tags!) ###
[PROMPT_START]<your hypothetical instructive prompt here>[PROMPT_END]
"""

CLASSIFICATION_TASK_TEMPLATE_HYPE = """{PROMPT}

### HARD CONSTRAINTS ###
1. OUTPUT FORMAT:
   - Output ONLY the final answer in the format: `<ans>LABEL</ans>`
   - LABEL MUST be EXACTLY one item from the list: [{LABELS}]
   - NEVER add explanations, thoughts, or extra text.
   - NEVER modify the label format (e.g., no \"A: Explanation\", ONLY \"A\").
2. STOP CONDITION:
   - IMMEDIATELY stop generating after `</ans>`.

### INPUT ###
{INPUT}

### RESPONSE ###
"""
GENERATION_TASK_TEMPLATE_HYPE = """{PROMPT}

### HARD CONSTRAINTS ###
1. OUTPUT FORMAT:
   - Output ONLY the generated content. NO redundant explanations.
   - NEVER add meta-commentary (e.g., \"Here is the answer:\").
2. STOP CONDITION:
   - Stop IMMEDIATELY after the last required token.

### INPUT ###
{INPUT}

### RESPONSE ###
"""
