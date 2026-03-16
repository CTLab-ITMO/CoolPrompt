PE2_SGR_TEMPLATE = """You are a prompt engineer \
optimizing a prompt by analyzing failure examples.

Current prompt:
{prompt}

Full template used:
{full_template}

Below are {batch_size} failure examples where the prompt \
produced incorrect outputs:

{examples}

Analyze each failure example, assess the current prompt, \
decide what changes are needed, and produce an improved \
prompt. The improved prompt must be at most \
{max_tokens} tokens long.

Follow these steps:
1. For each failure, identify what went wrong and categorize \
the root cause.
2. Assess whether the prompt correctly describes the task, \
noting any missing or misleading elements.
3. Decide whether editing is necessary and justify your \
decision.
4. List specific changes to the prompt.
5. Write the full improved prompt incorporating all changes.
6. Summarize the main improvement in one sentence."""
