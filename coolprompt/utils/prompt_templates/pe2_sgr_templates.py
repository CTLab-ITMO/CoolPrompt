"""PE2+SGR v2 prompt templates.

Phase 1 — diagnosis templates (light / full).
Phase 2 — generation templates (incremental / structural / reimagine).
"""

# ------------------------------------------------------------------
# Phase 1: Diagnosis templates
# ------------------------------------------------------------------

PE2_SGR_LIGHT_DIAGNOSIS_TEMPLATE = """\
You are a prompt engineer analyzing why a prompt is failing.

Current prompt:
{prompt}

Full template used:
{full_template}

Below are {batch_size} failure examples where the prompt \
produced incorrect outputs:

{examples}

Analyze the failures as a whole. Focus on the big picture:
1. What common pattern explains these failures?
2. How severe is the problem — surface-level, structural, \
or fundamental?
3. Does the prompt correctly describe the task? Note any \
missing or misleading elements.
4. What strategy should be used to fix it — incremental \
edits, structural rewrite, or complete reimagination?
5. What is the single most important insight for \
improvement?"""


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
