from __future__ import annotations

from coolprompt.spec_generator.schema import CornerCase, TaskSpec
from coolprompt.utils.enums import Task
from coolprompt.utils.prompt_templates.spec_generator_templates import (
    SPEC_CORNER_CLASSIFICATION_TEMPLATE,
    SPEC_CORNER_GENERATION_TEMPLATE,
    SPEC_REGULAR_CLASSIFICATION_TEMPLATE,
    SPEC_REGULAR_GENERATION_TEMPLATE,
)
from coolprompt.utils.prompt_templates.data_generator_templates import (
    get_corner_case_rules,
    get_standard_rules,
)

_REGULAR_TEMPLATES: dict[Task, str] = {
    Task.CLASSIFICATION: SPEC_REGULAR_CLASSIFICATION_TEMPLATE,
    Task.GENERATION: SPEC_REGULAR_GENERATION_TEMPLATE,
}

_CORNER_TEMPLATES: dict[Task, str] = {
    Task.CLASSIFICATION: SPEC_CORNER_CLASSIFICATION_TEMPLATE,
    Task.GENERATION: SPEC_CORNER_GENERATION_TEMPLATE,
}


def _join(items: list[str]) -> str:
    return ", ".join(items)


def _corner_cases_block(cases: list[CornerCase]) -> str:
    return "\n".join(
        f"{c.name}: {c.description} (hint: {c.example_hint})" for c in cases
    )


class RequestBuilder:
    def regular(self, spec: TaskSpec, task: Task, n: int) -> str:
        return _REGULAR_TEMPLATES[task].format(
            **self._base(spec),
            **self._classification_extra(task, spec),
            key_skills=_join(spec.key_skills),
            focused_skills=_join(spec.key_skills),
            additional_notes=spec.additional_notes or "None",
            num_samples=n,
        )

    def corner(self, spec: TaskSpec, task: Task, cases: list[CornerCase], n: int) -> str:
        return _CORNER_TEMPLATES[task].format(
            **self._base(spec),
            **self._classification_extra(task, spec),
            typical_errors=_join(spec.typical_errors),
            corner_name="Mixed corner cases",
            corner_description=(
                    "Generate examples covering the following corner-case patterns diversely:\n"
                    + _corner_cases_block(cases)
            ),
            corner_hint=(
                "Cover different patterns across examples. "
                "Do not make all examples the same type."
            ),
            num_samples=n,
        )

    def dataset_regular(self, spec: TaskSpec, dataset_name: str, n: int) -> str | None:
        template = get_standard_rules(dataset_name)

        if template is None:
            return None

        return template.format(**self._dataset_format_args(spec, n))

    def dataset_corner(self, spec: TaskSpec, dataset_name: str, n: int) -> str | None:
        template = get_corner_case_rules(dataset_name)

        if template is None:
            return None

        return template.format(**self._dataset_format_args(spec, n))

    def _base(self, spec: TaskSpec) -> dict[str, str]:
        return {
            "domain": spec.domain,
            "task_summary": spec.task_summary,
            "input_description": spec.io_format.input_description,
            "output_description": spec.io_format.output_description,
            "constraints": _join(spec.constraints),
            "language": spec.language or "English",
        }

    def _classification_extra(self, task: Task, spec: TaskSpec) -> dict[str, str]:
        return {"label_set": _join(spec.label_set or [])} if task == Task.CLASSIFICATION else {}

    def _dataset_format_args(
            self,
            spec: TaskSpec,
            n: int,
    ) -> dict[str, str | int]:
        return {
            "problem_description": spec.task_summary,
            "input_description": spec.io_format.input_description,
            "output_description": spec.io_format.output_description,
            "input_constraints": _join(spec.io_format.input_constraints),
            "output_constraints": _join(spec.io_format.output_constraints),
            "constraints": _join(spec.constraints),
            "language": spec.language or "English",
            "label_set": _join(spec.label_set or []),
            "key_skills": _join(spec.key_skills),
            "typical_errors": _join(spec.typical_errors),
            "corner_cases": (
                _corner_cases_block(spec.corner_cases)
                if spec.corner_cases
                else "No explicit corner cases provided."
            ),
            "num_samples": n,
        }
