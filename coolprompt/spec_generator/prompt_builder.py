"""Render synthetic-data generation prompts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from coolprompt.spec_generator.distribution import TaskDistribution
from coolprompt.spec_generator.models import Example, GenerationContext
from coolprompt.utils.prompt_templates.snippets_templates import (
    DISTRIBUTION_AWARE_GUIDANCE,
    TARGETED_GUIDANCE,
)
from coolprompt.utils.enums import Task
from coolprompt.utils.prompt_templates.spec_generator_templates import (
    SPEC_REGULAR_CLASSIFICATION_TEMPLATE,
    SPEC_REGULAR_GENERATION_TEMPLATE,
)


_REGULAR_TEMPLATES: Mapping[Task, str] = {
    Task.CLASSIFICATION: SPEC_REGULAR_CLASSIFICATION_TEMPLATE,
    Task.GENERATION: SPEC_REGULAR_GENERATION_TEMPLATE,
}

_RETURN_MARKER = "\nReturn only:"


def _bullets(items: Sequence[str]) -> str:
    """Render non-empty strings as a Markdown bullet list."""

    return "\n".join(f"- {item.strip()}" for item in items if item.strip()) or "None"


def _distribution_axes(distribution: TaskDistribution) -> str:
    """Render distribution axes and values for a generation prompt."""

    def render_value(value) -> str:
        """Render one axis value with its optional target proportion."""

        target = f" (target≈{value.target_ratio:.1%})" if value.target_ratio is not None else ""
        return f"  - {value.id}: {value.description}{target}"

    return "\n".join(
        f"- {axis.name}: {axis.description}\n"
        + "\n".join(render_value(value) for value in axis.values)
        for axis in distribution.axes
    ) or "None"


def _target_lines(targets: Sequence[dict[str, Any]]) -> str:
    """Render targeted generation quotas as readable instructions."""

    def render_target(target: dict[str, Any]) -> str:
        """Render one targeted or exploratory generation quota."""

        count = int(target.get("count", 0))
        constraints = target.get("constraints", [])

        if not constraints:
            return f"- {count} exploratory examples with broad variation"

        values = ", ".join(
            f"{item['axis']}={item['value_id']} ({item['description']})"
            for item in constraints
        )
        return f"- {count} examples targeting: {values}"

    return "\n".join(render_target(target) for target in targets) or "None"


def _avoid_lines(avoid: Sequence[dict[str, Any]]) -> str:
    """Render axis values that should not be overproduced."""

    return "\n".join(
        f"- avoid overusing {item['axis']}={item['value_id']}: {item['description']}"
        for item in avoid
    ) or "None"


def _examples(examples: Sequence[Example]) -> str:
    """Render examples as JSON for inclusion in a prompt."""

    if not examples:
        return "None"

    return json.dumps(
        [
            {"input": example.input, "output": example.output}
            for example in examples
        ],
        ensure_ascii=False,
        indent=2,
    )


def _limited_examples(
    examples: Sequence[Example],
    limit: int,
    *,
    latest: bool = False,
) -> str:
    """Render a bounded prefix or suffix of an example sequence."""

    selected = examples[-limit:] if latest else examples[:limit]
    return _examples(selected)


def _insert_guidance(base: str, guidance: str) -> str:
    """Insert additional guidance immediately before the output contract."""

    if not guidance:
        return base

    guidance = guidance.strip()
    insert = f"\n\n{guidance}\n"

    return (
        base.replace(_RETURN_MARKER, insert + _RETURN_MARKER, 1)
        if _RETURN_MARKER in base
        else f"{base.rstrip()}{insert}"
    )


class GenerationPromptBuilder:
    """Build regular, distribution-aware, and targeted prompts."""

    def regular(self, context: GenerationContext, n: int) -> str:
        """Build a standard generation prompt for the requested batch size."""

        return self._render(context, n)

    def distribution_aware(
        self,
        context: GenerationContext,
        n: int,
        distribution: TaskDistribution,
        *,
        accepted_examples: Sequence[Example] = (),
        reference_examples: Sequence[Example] = (),
    ) -> str:
        """Build exploratory distribution-aware generation."""
        guidance = DISTRIBUTION_AWARE_GUIDANCE.format(
            axes=_distribution_axes(distribution),
            reference_examples=_limited_examples(reference_examples, 8),
            accepted_examples=_limited_examples(accepted_examples, 10, latest=True),
        )
        return _insert_guidance(self.regular(context, n), guidance)

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
    ) -> str:
        """Build coverage-gap-targeted generation."""
        guidance = TARGETED_GUIDANCE.format(
            axes=_distribution_axes(distribution),
            targets=_target_lines(targets),
            avoid=_avoid_lines(avoid),
            reference_examples=_limited_examples(reference_examples, 8),
            accepted_examples=_limited_examples(accepted_examples, 10, latest=True),
        )
        return _insert_guidance(self.regular(context, n), guidance)

    def _render(self, context: GenerationContext, n: int) -> str:
        """Render the task-specific base template from a generation context."""

        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")

        task = context.spec.task
        template = _REGULAR_TEMPLATES.get(task)

        if template is None:
            raise ValueError(f"Unsupported task: {task!r}.")

        return template.format(
            **self._args(context),
            reference_examples=_examples(context.seed_examples),
            num_samples=n,
        )

    @staticmethod
    def _args(context: GenerationContext) -> dict[str, str]:
        """Convert TaskSpec fields into template-ready strings."""

        spec = context.spec
        return {
            "description": spec.description,
            "input_format": spec.input_format,
            "output_format": spec.output_format,
            "requirements": _bullets(spec.requirements),
            "labels": _bullets(spec.labels or ()),
            "language": spec.language,
        }
