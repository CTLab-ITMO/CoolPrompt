from __future__ import annotations

import re

from pydantic import BaseModel, Field, create_model, field_validator

from coolprompt.spec_generator.schema import IOFormat, TaskSpec
from coolprompt.utils.enums import Task

_DEFAULT_MIN_LEN = 1
_DEFAULT_MAX_LEN = 4000

_LEN_HINT_RE = re.compile(
    r"(\d+)\s*(?:-|–|to)\s*(\d+)\s*(chars?|characters?|words?|symbols?)",
    re.IGNORECASE,
)


def _extract_length_bounds(
        constraints: list[str] | None,
) -> tuple[int, int, str] | None:
    text = " ".join(constraints or [])
    match = _LEN_HINT_RE.search(text)

    if not match:
        return None

    min_len = int(match.group(1))
    max_len = int(match.group(2))
    raw_unit = match.group(3).lower()

    unit = "words" if raw_unit.startswith("word") else "chars"

    return min_len, max_len, unit


class ExampleBase(BaseModel):
    input: str = Field(min_length=1)
    output: str = Field(min_length=1)


def build_example_model(
        spec: TaskSpec,
        _: Task,
) -> type[ExampleBase]:
    io_format: IOFormat = spec.io_format

    input_bounds = (
            _extract_length_bounds(io_format.input_constraints)
            or (_DEFAULT_MIN_LEN, _DEFAULT_MAX_LEN, "chars")
    )
    in_min_len, in_max_len, in_length_unit = input_bounds

    output_bounds = _extract_length_bounds(
        io_format.output_constraints
    )

    canonical_labels: dict[str, str] = {
        label.casefold(): label
        for label in spec.label_set or []
    }

    def _validate_input(cls, value: str) -> str:  # noqa: N805
        stripped = value.strip()

        if not stripped:
            raise ValueError(
                "input is empty after stripping whitespace"
            )

        actual_length = (
            len(stripped.split())
            if in_length_unit == "words"
            else len(stripped)
        )

        if not in_min_len <= actual_length <= in_max_len:
            raise ValueError(
                f"input length {actual_length} {in_length_unit} is outside "
                f"allowed bounds [{in_min_len}, {in_max_len}]"
            )

        return stripped

    def _validate_output(cls, value: str) -> str:  # noqa: N805
        stripped = value.strip()

        if not stripped:
            raise ValueError(
                "output is empty after stripping whitespace"
            )

        if output_bounds is not None:
            out_min_len, out_max_len, out_length_unit = output_bounds

            actual_length = (
                len(stripped.split())
                if out_length_unit == "words"
                else len(stripped)
            )

            if not out_min_len <= actual_length <= out_max_len:
                raise ValueError(
                    f"output length {actual_length} {out_length_unit} is "
                    f"outside allowed bounds "
                    f"[{out_min_len}, {out_max_len}]"
                )

        if canonical_labels:
            normalized = stripped.casefold()

            if normalized not in canonical_labels:
                raise ValueError(
                    f"output {stripped!r} is not one of the allowed labels "
                    f"{sorted(canonical_labels.values())}"
                )

            stripped = canonical_labels[normalized]

        return stripped

    validators = {
        "_validate_input": field_validator("input")(
            _validate_input
        ),
        "_validate_output": field_validator("output")(
            _validate_output
        ),
    }

    return create_model(
        "ValidatedExample",
        __base__=ExampleBase,
        __validators__=validators,
    )
