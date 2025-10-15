## HyPE optimizer

Provides the prompt refactoring workflow: the initial prompt is injected into special predetermined by our researches query template.
This optimizer requires only **one** query to the LLM, so it surely can be used as a <ins>fast</ins> and <ins>simple</ins> tool to make your prompt better.

##### HyPE template:

```
### ROLE ###
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
```