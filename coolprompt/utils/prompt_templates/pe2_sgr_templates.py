"""PE2+SGR v3 prompt templates.

Phase 1 — free-form diagnosis (below threshold) or
           structured diagnosis (above threshold).
Phase 2 — generation templates (incremental / structural / reimagine).
"""

# ------------------------------------------------------------------
# Phase 1: Diagnosis templates
# ------------------------------------------------------------------

PE2_SGR_FREEFORM_DIAGNOSIS_TEMPLATE = """\
A prompt is a text paragraph that outlines the expected actions \
and instructs the model to generate a specific output. This \
prompt is concatenated with the input text, and the model then \
creates the required output.

In our collaboration, we'll work together to refine a prompt. \
The process consists of two main steps:

## Step 1
I will provide you with the current prompt, how the prompt is \
concatenated with the input text (i.e., "full template"), along \
with {batch_size} example(s) that are associated with this \
prompt. Each example contains the input, the final answer \
produced by the model, and the ground-truth label to the input. \
Your task is to analyze the examples, determining whether the \
existing prompt is describing the task reflected by these \
examples precisely, and suggest changes to the prompt.

## Step 2
Next, you will carefully review your reasoning in step 1, \
integrate the insights to craft a new, optimized prompt.

## Prompt
{prompt}

## Full Template
This describes how the prompt of interest is concatenated with \
the input text. The prompt may appear before the input text, or \
after the input text. Optionally the full template may contain \
other template information.
```
{full_template}
```

## Examples
{examples}

## Instructions
For some of these examples, the output does not match with the \
label. This may be due to the prompt being misleading or not \
describing the task precisely.

Please examine the example(s) carefully. Note that the \
ground-truth labels are __absolutely correct__, but the prompts \
(task descriptions) may be incorrect and need modification.

As you analyze, pay special attention to:
1. What common pattern explains ALL these failures?
2. Is the problem surface-level (formatting), structural \
(missing key instructions), or fundamental (entire approach \
is wrong)?
3. Are the errors homogeneous (all same root cause) or diverse?
4. If the entire prompt approach seems wrong, consider what a \
completely different prompt would look like.

For each example, provide reasoning according to the following \
template:

### Example <id>
Input: <input>
Output: <output>
Label: <label>
Is the output correct compared to the label: \
<yes or no, and your reasoning>
Is the output correctly following the given prompt: \
<yes or no, and your reasoning>
Is the prompt correctly describing the task shown by the \
input-label pair: <yes or no, and your reasoning>
To output the correct label, is it necessary to edit the \
prompt: <yes or no, and your reasoning>
If yes, provide detailed analysis and actionable suggestions \
to edit the prompt: <analysis and suggestions>"""


PE2_SGR_FULL_DIAGNOSIS_TEMPLATE = """\
You are a prompt engineer analyzing why a prompt is failing.

Current prompt:
{prompt}

Full template used:
{full_template}

Below are {batch_size} failure examples where the prompt \
produced incorrect outputs:

{examples}

Analyze each failure individually AND as a whole:
1. For each failure, identify the root cause and categorize \
it (task unclear, missing constraints, wrong format, \
incomplete guidance, overspecification, or other).
2. What common pattern explains these failures across all \
examples?
3. How similar are the root causes — are the errors \
homogeneous (all same cause) or diverse?
4. Does the prompt correctly describe the task? Note any \
missing or misleading elements.
5. Is editing necessary? How confident are you?
6. What strategy should be used — incremental edits, \
structural rewrite, or complete reimagination?
7. What is the single most important insight for \
improvement?"""


# ------------------------------------------------------------------
# Phase 2: Generation templates
# ------------------------------------------------------------------

PE2_SGR_GEN_INCREMENTAL = """\
Review the diagnosis below and make targeted improvements \
to the current prompt.

## Current Prompt
{prompt}

## Diagnosis
{formatted_diagnosis}

## Instructions
* Make specific, targeted changes based on the diagnosis.
* Preserve the overall structure and approach of the \
current prompt.
* The total length of the prompt should be less than \
{max_tokens} words.
* Reply with the prompt only. Do not include other text.

Output the improved prompt between <prompt> and </prompt> \
tags.

<prompt>your improved prompt here</prompt>"""


PE2_SGR_GEN_STRUCTURAL = """\
The diagnosis identified structural issues requiring \
significant changes to the prompt.

## Current Prompt (for reference)
{prompt}

## Diagnosis
{formatted_diagnosis}

## Instructions
* Significantly reorganize the prompt to address the \
structural issues identified in the diagnosis.
* You may change the order, add sections, or reframe \
instructions entirely.
* Preserve the core task intent but not necessarily the \
current structure.
* The total length of the prompt should be less than \
{max_tokens} words.
* Reply with the prompt only. Do not include other text.

Output the improved prompt between <prompt> and </prompt> \
tags.

<prompt>your improved prompt here</prompt>"""


PE2_SGR_GEN_REIMAGINE = """\
The diagnosis found fundamental issues with the current \
prompt's approach.

## Diagnosis
{formatted_diagnosis}

## Instructions
* Design a completely new prompt for this task from scratch.
* Do NOT constrain yourself to the current prompt's \
structure or wording — start fresh.
* Focus on clearly communicating what the task requires \
based on the patterns identified in the diagnosis.
* The total length of the prompt should be less than \
{max_tokens} words.
* Reply with the prompt only. Do not include other text.

Output the improved prompt between <prompt> and </prompt> \
tags.

<prompt>your improved prompt here</prompt>"""
