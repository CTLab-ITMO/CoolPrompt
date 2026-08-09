"""Render synthetic-data generation prompts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

from coolprompt.spec_generator.models import GenerationContext, Example
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
    """Render non-empty strings as a bullet list."""

    values = [item.strip() for item in items if item.strip()]
    return "\n".join(f"- {item}" for item in values) or "None"


def _examples(examples: Sequence[Example]) -> str:
    """Render trusted examples as escaped XML."""

    if not examples:
        return "None"

    return "\n".join(
        f'<example index="{index}">\n'
        f"<input>{escape(example.input)}</input>\n"
        f"<output>{escape(example.output)}</output>\n"
        "</example>"
        for index, example in enumerate(examples, start=1)
    )


class GenerationPromptBuilder:
    """Build regular and corner-case generation prompts."""

    def regular(self, context: GenerationContext, n: int) -> str:
        """Build a prompt for regular examples."""

        return self._render(
            context=context,
            n=n,
            templates=_REGULAR_TEMPLATES,
        )

    def corner(self, context: GenerationContext, n: int,
               *, corner_cases: Sequence[str] | None = None) -> str:
        """Build a prompt for difficult but valid examples."""

        selected = tuple(
            context.spec.corner_cases
            if corner_cases is None
            else corner_cases
        )

        if not selected:
            raise ValueError("Corner-case generation requires at least one corner case.")

        return self._render(
            context=context,
            n=n,
            templates=_CORNER_TEMPLATES,
            corner_cases=_bullets(selected),
        )

    def _render(
            self,
            *,
            context: GenerationContext,
            n: int,
            templates: Mapping[Task, str],
            **extra: str) -> str:
        """Render one prompt from the selected task template."""

        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")

        try:
            template = templates[context.spec.task]
        except KeyError as exc:
            raise ValueError(f"Unsupported task: {context.spec.task!r}.") from exc

        return template.format(
            **self._args(context),
            **extra,
            reference_examples=_examples(
                context.seed_examples
            ),
            num_samples=n,
        )

    @staticmethod
    def _args(context: GenerationContext) -> dict[str, str]:
        """Return common template arguments."""

        spec = context.spec

        return {
            "description": spec.description,
            "input_format": spec.input_format,
            "output_format": spec.output_format,
            "requirements": _bullets(spec.requirements),
            "labels": _bullets(spec.labels or ()),
            "language": spec.language,
        }
