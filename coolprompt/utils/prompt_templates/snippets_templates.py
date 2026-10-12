"""Prompt guidance for distribution-aware synthetic data generation.

Defines shared quality, diversity, tagging, verification, coverage, and targeted
generation instructions, plus the tagged JSON output contract.
"""

from __future__ import annotations

TAGGED_GENERATION_OUTPUT_CONTRACT = (
    "Return only:\n"
    '{"examples": [{"input": "string", "output": "string", '
    '"axis_tags": {"axis_name": "value_id"}}]}\n'
)

_ROLE = """
Role:
You are a senior data engineer building high-quality supervised training data.
Every example must be correct, realistic, and suitable for model training.
"""

_ROLE_TARGETED = """
Role:
You are a senior data engineer building high-quality supervised training data.
Every example must be correct, realistic, and suitable for model training.
You are now closing specific gaps in the dataset's coverage.

Task axes (dimensions of variation):
<axes>
{axes}
</axes>
"""

_COVERAGE = """
Coverage:
Use the task axes below as dimensions of variation for this batch.

<axes>
{axes}
</axes>
"""

_TARGETS = """
Targets for this batch (counts apply to this request only).

<targets>
{targets}
</targets>

Each target describes one group of examples and may constrain several axes.
Generate exactly the requested number of examples for every target group, and
assign each generated example to one target group only. The sum of target
counts is the total number of examples to return.

Meet every target through the actual content and structure of the example, not
merely through its tags. For axes not constrained by the example's group,
report the values that the example actually exhibits.

Overrepresented values to deprioritize when compatible with the targets
(ignore if empty):
<avoid>
{avoid}
</avoid>
"""

_QUALITY = """
Quality:
- Write realistic examples with enough meaningful context to exercise the
  skills required by the task.
- Match the natural complexity implied by the task specification (reference
  examples, if any, are only a rough guide). Avoid toy-like minimal examples
  unless the task is inherently minimal, and avoid deliberately convoluted or
  adversarial ones.
- Satisfy axis values through the actual content of the example, not through
  simplified templates.
"""

_DIVERSITY = """
Diversity:
<accepted_examples>
{accepted_examples}
</accepted_examples>
(Ignore if empty.)

While preserving all required axis values and targets, avoid semantic
repetition: do not reuse the scenario, topic, problem structure, or reasoning
pattern of accepted examples or of other examples in this batch.

Superficial changes such as different wording, entities, variable names,
numbers, or formatting do not make an example meaningfully different.
"""

_TAGS = """
Tags:
For every example, report axis_tags using only the exact axis names and value
ids listed above, exactly one value id per axis. Each tag must describe a
property actually observable in the generated input-output pair. For a
classification label axis, the tag must agree with the example's output.
"""

_VERIFICATION = """
Verification:
Before returning an example, independently derive the correct output from the
input alone (without looking at your intended output), then check that it
matches and is consistent with the task specification.

If the task involves calculations, verify every computation.

Discard examples that are inconsistent, ambiguous, under-specified, or whose
output is not clearly correct under the task specification, and generate
replacements.
"""

_COMMON = _QUALITY + _DIVERSITY + _TAGS + _VERIFICATION

DISTRIBUTION_AWARE_GUIDANCE = _ROLE + _COVERAGE + _COMMON
TARGETED_GUIDANCE = _ROLE_TARGETED + _TARGETS + _COMMON
