PE2_REASONING_TEMPLATE = """\
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
(task descriptions) may be incorrect and need modification. For \
each example, provide reasoning according to the following \
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


PE2_REFINEMENT_TEMPLATE = """\
Now please carefully review your reasoning in Step 1 and help \
with Step 2: refining the prompt.

## Current Prompt
{prompt}

## Step 1 Reasoning
{reasoning}

## Instructions
* The total length of the prompt should be less than \
{max_tokens} words.
* Please help edit the prompt so that the updated prompt will \
not fail on these examples anymore.
* Reply with the prompt only. Do not include other text.

Output the improved prompt between <prompt> and </prompt> tags.

<prompt>your improved prompt here</prompt>"""
