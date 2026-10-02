"""Guidance snippets injected into generation prompts."""

from __future__ import annotations

DISTRIBUTION_AWARE_GUIDANCE = """
Coverage guidance:
Use the task axes below as explicit dimensions of variation when generating the batch.

Task-distribution axes:
{axes}

Previously accepted synthetic examples:
{accepted_examples}

Generate examples substantially different from already accepted synthetic examples.
Avoid examples that differ from accepted examples only through small lexical or surface
changes. Vary examples meaningfully within this batch too.

For every generated example, report axis_tags using only the exact axis names and value
ids listed above. For each axis, report exactly one value id from that axis.
Each reported tag must describe an observable property actually present in the generated
input-output pair.
"""

TARGETED_GUIDANCE = """
Task-distribution axes:
{axes}

Target this batch according to:
{targets}

Overrepresented values to deprioritize when compatible with the targets:
{avoid}

Previously accepted synthetic examples:
{accepted_examples}

The new examples must not be simple paraphrases or surface-level variants of accepted
examples. Introduce substantive variation while preserving the task constraints and
requested target properties.

The target counts apply to this request only. Meet each target's observable conditions,
not merely its tag. Vary constructions within the batch while preserving task constraints.

For every generated example, report axis_tags using only the exact axis names and value
ids listed above. For each axis, report exactly one value id from that axis.
Each reported tag must match the observable properties actually present in the generated
input-output pair.
"""
