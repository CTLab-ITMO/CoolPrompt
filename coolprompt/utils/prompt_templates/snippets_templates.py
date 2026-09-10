"""Guidance snippets injected into generation prompts."""

from __future__ import annotations

DISTRIBUTION_AWARE_GUIDANCE = """
Coverage guidance:
Use the task axes below to create meaningful variation. For TARGET_PROPORTIONS axes,
keep the batch direction consistent with the shown empirical source proportions; exact
per-batch ratios are not required because feedback corrects them across batches.

Task-distribution axes:
{axes}

Source-distribution reference examples:
{reference_examples}

Use the source examples only to match broad properties such as input cardinality,
concreteness, semantic regime, relation types, and output style. Do NOT copy their exact
concept combinations, scenarios, or wording. Do not drift into abstract/philosophical
examples unless that regime is actually represented in the source references or TaskSpec.

Previously accepted synthetic examples:
{accepted_examples}

Generate examples substantially different from already accepted synthetic examples.
Avoid repeating semantic scenarios, concept combinations, and sentence structures with
only small lexical changes.

For every generated example, report axis_tags using only the exact axis names and value
ids listed above. For each axis, report exactly one value id from that axis.
"""

TARGETED_GUIDANCE = """
Task-distribution axes:
{axes}

Target this batch according to:
{targets}

Overrepresented values to avoid unless required for correctness:
{avoid}

Source-distribution reference examples:
{reference_examples}

Stay in the broad source-data regime shown above. Match its kinds of inputs, semantic
concreteness, relations/actions, and output style without copying exact examples.

Previously accepted synthetic examples:
{accepted_examples}

The new examples must not be simple paraphrases of accepted examples. Vary semantic
scenario, concept combinations, relation structure, and sentence structure before merely
varying wording.

For every generated example, report axis_tags using only exact axis names and value ids
from the task-distribution axes. For each axis, report exactly one value id from that axis.
"""