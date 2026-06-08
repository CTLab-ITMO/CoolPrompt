# Prompt Compressor

## LLM-Based Prompt Compression Method

Provides a prompt compression workflow: the initial verbose prompt is analyzed and condensed into a concise, two-sentence format while preserving the essential task and context.  
This optimizer requires only one query to the LLM, so it can be used as a fast and lightweight tool to make your prompts more efficient.

### Workflow

1. **Input** – A verbose user prompt is provided.
2. **Structured Extraction** – The LLM extracts:
   - The core input context (≤10 words).
   - The primary task/question (≤10 words + "answer briefly").
3. **Output** – The two components are combined into a single compressed prompt.

---

### Prompt Compressor Template

The following prompts are used by default. You can pass custom prompts to PromptCompressor.

#### System Prompt

```
## Role
You are an expert prompt engineer specializing in prompt optimization.

## Task
You will receive an original user prompt containing task context and a question.
Condense it into a new prompt following this structure:
1. Task context in one concise sentence.
2. Target question in one concise sentence.

Execute step-by-step:
- Analyze the text to isolate the core question.
- Condense the task context to ≤10 words.
- Condense the target question to ≤10 words and append "answer briefly".
- Merge both sentences into a single final prompt.

## Response Format
- Output must be strictly valid JSON.
- Use ONLY information from the original prompt.
- You will be penalized for hallucinating or inventing content.
```

#### User Prompt

```
## Original prompt: {prompt}
```

#### Expected JSON Response Schema

```json
{
  "reasoning": "Analysis of the task and question in the original prompt",
  "prompt_input_context": "Extracted input context in one sentence",
  "prompt_task": "Extracted task sentence",
  "final_prompt": "Final compressed prompt"
}
```

---

### Usage

#### Through `PromptTuner` (Recommended)

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from coolprompt import PromptTuner

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini", 
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),  
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

tuner = PromptTuner(target_model=llm, system_model=llm)

original_prompt = """INPUT YOUR PROMPT HERE"""

compressed = tuner.run(
    start_prompt=original_prompt,
    method="compress",
    verbose=1,
)
```

### Customization

You can provide your own templates when instantiating `PromptCompressor`:

```python
compressor = PromptCompressor(
    model=llm,
    system_prompt="Your custom system prompt",
    user_prompt="Your custom template with {prompt}"
)
```

---

### Notes

- The compressor does **not** require a dataset or evaluation metric – it is a zero-shot method.
- The output prompt is designed to be concise while preserving the original task intent.