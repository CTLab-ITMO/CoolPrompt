"""Render synthetic-data generation prompts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

from coolprompt.spec_generator.distribution import TaskDistribution
from coolprompt.spec_generator.models import Example, GenerationContext
from coolprompt.utils.enums import Task
from coolprompt.utils.prompt_templates.spec_generator_templates import (
    SPEC_CORNER_CLASSIFICATION_TEMPLATE,
    SPEC_CORNER_GENERATION_TEMPLATE,
    SPEC_REGULAR_CLASSIFICATION_TEMPLATE,
    SPEC_REGULAR_GENERATION_TEMPLATE,
)

_REGULAR_TEMPLATES: Mapping[Task, str] = {
    Task.CLASSIFICATION: SPEC_REGULAR_CLASSIFICATION_TEMPLATE,
    Task.GENERATION: SPEC_REGULAR_GENERATION_TEMPLATE,
}

_CORNER_TEMPLATES: Mapping[Task, str] = {
    Task.CLASSIFICATION: SPEC_CORNER_CLASSIFICATION_TEMPLATE,
    Task.GENERATION: SPEC_CORNER_GENERATION_TEMPLATE,
}


def _bullets(items: Sequence[str]) -> str:
    values = [item.strip() for item in items if item.strip()]
    return "\n".join(f"- {item}" for item in values) or "None"


def _distribution_axes(distribution: TaskDistribution) -> str:
    blocks: list[str] = []
    for axis in distribution.axes:
        values = "\n".join(
            f"  - {value.id}: {value.description}"
            + (f" (target≈{value.target_ratio:.1%})" if value.target_ratio is not None else "")
            for value in axis.values
        )
        blocks.append(f"- {axis.name}: {axis.description}\n{values}")
    return "\n".join(blocks) or "None"


def _target_lines(targets: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for target in targets:
        count = int(target.get("count", 0))
        constraints = target.get("constraints", [])
        if not constraints:
            lines.append(f"- {count} exploratory examples with broad variation")
            continue
        rendered = ", ".join(
            f"{item['axis']}={item['value_id']} ({item['description']})"
            for item in constraints
        )
        lines.append(f"- {count} examples targeting: {rendered}")
    return "\n".join(lines) or "None"


def _avoid_lines(avoid: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        f"- avoid overusing {item['axis']}={item['value_id']}: {item['description']}"
        for item in avoid
    ) or "None"


def _examples(examples: Sequence[Example]) -> str:
    if not examples:
        return "None"

    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        refs = ""
        if example.references:
            rendered = "\n".join(
                f"<reference>{escape(ref)}</reference>" for ref in example.references
            )
            refs = (
                "\n<alternative_valid_outputs>\n"
                f"{rendered}\n"
                "</alternative_valid_outputs>"
            )
        blocks.append(
            f'<example index="{index}">\n'
            f"<input>{escape(example.input)}</input>\n"
            f"<output>{escape(example.output)}</output>"
            f"{refs}\n"
            "</example>"
        )
    return "\n".join(blocks)


def _accepted_examples(examples: Sequence[Example], *, limit: int = 10) -> str:
    if not examples:
        return "None"
    return _examples(examples[-limit:])


def _distribution_reference_examples(
    examples: Sequence[Example],
    *,
    limit: int = 8,
) -> str:
    """Render a small source-distribution sample as style/structure grounding.

    These examples are not extra training targets. They are only broad distribution
    evidence and must not be copied.
    """

    if not examples:
        return "None"
    return _examples(examples[:limit])



def _multi_reference_guidance(context: GenerationContext, valid_outputs_per_example: int) -> str:
    """Ask generation tasks for several genuinely different valid outputs per input."""
    if context.spec.task != Task.GENERATION or valid_outputs_per_example <= 1:
        return ""
    alternatives = valid_outputs_per_example - 1
    return f"""
Multi-reference requirement:
For each generated input, produce exactly {valid_outputs_per_example} valid outputs for the
same input: one primary `output` plus exactly {alternatives} strings in `references`.
All outputs must satisfy the same task requirements and use the same input concepts.
The references must be meaningfully different realizations, not trivial lexical paraphrases:
vary syntax, event framing/subject choice, and reasonable contextual detail while preserving
correctness. Do not introduce a contradictory event or omit required input concepts.
"""

def _inject_before_return(base: str, guidance: str) -> str:
    marker = "\nReturn only:"
    if marker not in base:
        return f"{base.rstrip()}\n\n{guidance.strip()}\n"
    return base.replace(marker, f"\n\n{guidance.strip()}\n{marker}", 1)


class GenerationPromptBuilder:
    """Build regular, targeted, and corner-case generation prompts."""

    def regular(
        self,
        context: GenerationContext,
        n: int,
        *,
        valid_outputs_per_example: int = 1,
    ) -> str:
        base = self._render(context=context, n=n, templates=_REGULAR_TEMPLATES)
        guidance = _multi_reference_guidance(context, valid_outputs_per_example)
        return _inject_before_return(base, guidance) if guidance else base

    def corner(
        self,
        context: GenerationContext,
        n: int,
        *,
        corner_cases: Sequence[str] | None = None,
        valid_outputs_per_example: int = 1,
    ) -> str:
        selected = tuple(context.spec.corner_cases if corner_cases is None else corner_cases)
        if not selected:
            raise ValueError("Corner-case generation requires at least one corner case.")
        base = self._render(
            context=context,
            n=n,
            templates=_CORNER_TEMPLATES,
            corner_cases=_bullets(selected),
        )
        guidance = _multi_reference_guidance(context, valid_outputs_per_example)
        return _inject_before_return(base, guidance) if guidance else base

    def distribution_aware(
        self,
        context: GenerationContext,
        n: int,
        distribution: TaskDistribution,
        *,
        accepted_examples: Sequence[Example] = (),
        reference_examples: Sequence[Example] = (),
        valid_outputs_per_example: int = 1,
    ) -> str:
        """Build exploratory generation grounded in desired and source distributions."""

        base = self.regular(
            context, n, valid_outputs_per_example=valid_outputs_per_example
        )
        guidance = f"""
Coverage guidance:
Use the task axes below to create meaningful variation. For TARGET_PROPORTIONS axes,
keep the batch direction consistent with the shown empirical source proportions; exact
per-batch ratios are not required because feedback corrects them across batches.

Task-distribution axes:
{_distribution_axes(distribution)}

Source-distribution reference examples:
{_distribution_reference_examples(reference_examples)}

Use the source examples only to match broad properties such as input cardinality,
concreteness, semantic regime, relation types, and output style. Do NOT copy their exact
concept combinations, scenarios, or wording. Do not drift into abstract/philosophical
examples unless that regime is actually represented in the source references or TaskSpec.

Previously accepted synthetic examples:
{_accepted_examples(accepted_examples)}

Generate examples substantially different from already accepted synthetic examples.
Avoid repeating semantic scenarios, concept combinations, and sentence structures with
only small lexical changes.

For every generated example, report axis_tags using only the exact axis names and value
ids listed above. For each axis, report exactly one value id from that axis.
"""
        return _inject_before_return(base, guidance)

    def targeted(
        self,
        context: GenerationContext,
        n: int,
        distribution: TaskDistribution,
        *,
        targets: Sequence[dict[str, Any]],
        avoid: Sequence[dict[str, Any]] = (),
        accepted_examples: Sequence[Example] = (),
        reference_examples: Sequence[Example] = (),
        valid_outputs_per_example: int = 1,
    ) -> str:
        """Build a gap-targeted batch grounded in source-distribution examples."""

        base = self.regular(
            context, n, valid_outputs_per_example=valid_outputs_per_example
        )
        guidance = f"""
Task-distribution axes:
{_distribution_axes(distribution)}

Target this batch according to:
{_target_lines(targets)}

Overrepresented values to avoid unless required for correctness:
{_avoid_lines(avoid)}

Source-distribution reference examples:
{_distribution_reference_examples(reference_examples)}

Stay in the broad source-data regime shown above. Match its kinds of inputs, semantic
concreteness, relations/actions, and output style without copying exact examples.

Previously accepted synthetic examples:
{_accepted_examples(accepted_examples)}

The new examples must not be simple paraphrases of accepted examples. Vary semantic
scenario, concept combinations, relation structure, and sentence structure before merely
varying wording.

For every generated example, report axis_tags using only exact axis names and value ids
from the task-distribution axes. For each axis, report exactly one value id from that axis.
"""
        return _inject_before_return(base, guidance)

    def corner_cover(
        self,
        context: GenerationContext,
        corner_cases: Sequence[str],
        *,
        distribution: TaskDistribution | None = None,
        accepted_examples: Sequence[Example] = (),
        reference_examples: Sequence[Example] = (),
        valid_outputs_per_example: int = 1,
    ) -> str:
        if not corner_cases:
            raise ValueError("corner_cases must not be empty")

        base = self.corner(
            context,
            len(corner_cases),
            corner_cases=corner_cases,
            valid_outputs_per_example=valid_outputs_per_example,
        )
        mapping = "\n".join(
            f"- Example {index}: {case}"
            for index, case in enumerate(corner_cases, start=1)
        )
        guidance = f"""
Coverage requirement:
Generate exactly one example for each listed corner case, in the same order:
{mapping}

Source-distribution reference examples:
{_distribution_reference_examples(reference_examples)}

Previously accepted synthetic examples:
{_accepted_examples(accepted_examples)}

Keep corner cases valid for the same source-data regime and avoid semantic/structural
repetition of accepted examples.
"""
        if distribution is not None:
            guidance += f"""

Task-distribution axes:
{_distribution_axes(distribution)}

Also report axis_tags using exact axis names/value ids. For each axis, report exactly one
value id from that axis.
"""
        return _inject_before_return(base, guidance)

    def _render(
        self,
        *,
        context: GenerationContext,
        n: int,
        templates: Mapping[Task, str],
        **extra: str,
    ) -> str:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")

        try:
            template = templates[context.spec.task]
        except KeyError as exc:
            raise ValueError(f"Unsupported task: {context.spec.task!r}.") from exc

        return template.format(
            **self._args(context),
            **extra,
            reference_examples=_examples(context.seed_examples),
            num_samples=n,
        )

    @staticmethod
    def _args(context: GenerationContext) -> dict[str, str]:
        spec = context.spec
        return {
            "description": spec.description,
            "input_format": spec.input_format,
            "output_format": spec.output_format,
            "requirements": _bullets(spec.requirements),
            "labels": _bullets(spec.labels or ()),
            "language": spec.language,
        }
