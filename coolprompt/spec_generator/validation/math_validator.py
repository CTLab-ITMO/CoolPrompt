"""Validate arithmetic word problems with exact calculations and unit conversions."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

import pint
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, ConfigDict, Field

from coolprompt.spec_generator.utils import parse_structured
from coolprompt.spec_generator.validation.math_constants import (
    MathValidationConstants as C,
)
from coolprompt.spec_generator.validation.math_parsing import (
    Evaluation,
    calculate,
    evaluate_expression,
    format_number,
    is_terminating_decimal,
    parse_answer,
    parse_answer_with_unit,
)
from coolprompt.spec_generator.validation.math_units import (
    UnitError,
    _alias_key,
    _define,
    _same_unit,
    convert_units,
    normalize_unit,
    unit_dimension,
)
from coolprompt.utils.prompt_templates.math_validation_template import (
    MATH_VALIDATION_TEMPLATE,
)

__all__ = [
    "Evaluation",
    "Example",
    "MathValidator",
    "Solution",
    "SolutionBatch",
    "UnitError",
    "ValidationResult",
    "calculate",
    "convert_units",
    "create_math_validator",
    "evaluate_expression",
    "format_number",
    "is_terminating_decimal",
    "normalize_unit",
    "parse_answer",
    "parse_answer_with_unit",
    "unit_dimension",
]


class Example(BaseModel):
    """Represent a math problem and its original answer."""

    input: str
    output: str


class Solution(BaseModel):
    """Represent a model-generated mathematical solution."""

    model_config = ConfigDict(extra="forbid")
    id: int
    status: Literal["solved", "invalid", "uncertain"]
    expression: str | None
    computed_unit: str | None
    requested_unit: str | None
    must_be_integer: bool
    must_be_nonnegative: bool
    reason: str = Field(description="Empty when solved; otherwise a short reason.")
    rounding_reason: str = Field(
        default="", description="Why floor/ceil is required, or empty."
    )


class SolutionBatch(BaseModel):
    """Batch of solutions returned by a solver model."""

    model_config = ConfigDict(extra="forbid")
    solutions: list[Solution]


class ValidationResult(BaseModel):
    """Validation outcome for a single example."""

    original: Example
    corrected: Example | None = None
    status: Literal["accepted", "corrected", "rejected", "uncertain"]
    reason: str = ""
    expression: str | None = None
    answer_unit: str | None = None
    expected_answer: str | None = None
    reason_code: str = ""
    retryable: bool = False
    verified_output: str | None = None


SolveBatch = Callable[[str], SolutionBatch]


@dataclass(frozen=True)
class _Outcome:
    """Internal result of assessing a single solution."""

    status: Literal["ok", "rejected", "uncertain"]
    reason: str = ""
    expected: Fraction | None = None
    unit: str | None = None
    reason_code: str = ""
    retryable: bool = False


def _uncertain(code: str, reason: str) -> _Outcome:
    """Build an uncertain outcome, marking retryable codes."""
    return _Outcome(
        "uncertain", reason, reason_code=code, retryable=code in C.RETRYABLE
    )


class MathValidator:
    """Validate math problems with exact arithmetic and optional independent models."""

    def __init__(
        self,
        solve: SolveBatch,
        batch_size: int = 10,
        cross_check: SolveBatch | None = None,
        *,
        cross_check_batch_size: int | None = None,
        require_cross_check: bool = False,
        audit_fraction: float = 0.0,
        seed: int = 0,
        unit_registry: pint.UnitRegistry | None = None,
        unit_aliases: Mapping[str, str] | None = None,
        unit_context: str = "",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not 0 <= audit_fraction <= 1:
            raise ValueError("audit_fraction must be between 0 and 1.")
        if audit_fraction and cross_check is None:
            raise ValueError("Accepted-answer auditing requires cross_check.")
        if cross_check_batch_size is not None and (
            type(cross_check_batch_size) is not int or cross_check_batch_size < 1
        ):
            raise ValueError("cross_check_batch_size must be a positive integer.")
        self._solve = solve
        self._cross_check = cross_check
        self._batch_size = batch_size
        self._cross_check_batch_size = (
            cross_check_batch_size if cross_check_batch_size is not None else batch_size
        )
        self._require_cross_check = require_cross_check
        self._audit_fraction = audit_fraction
        self._seed = seed
        registry = (
            unit_registry
            if unit_registry is not None
            else pint.UnitRegistry(non_int_type=Fraction)
        )
        if registry.non_int_type is not Fraction:
            raise ValueError("unit_registry must use non_int_type=Fraction.")
        self._units = dict(registry=registry, aliases=dict(unit_aliases or {}))
        self._unit_context = unit_context

    def validate(self, examples: Sequence[Example]) -> list[ValidationResult]:
        """Validate examples and return one result per example, in input order."""
        examples = [Example(input=item.input, output=item.output) for item in examples]
        results: list[ValidationResult | None] = [None] * len(examples)
        rng = random.Random(self._seed)
        pending_cross_check: list[int] = []
        primary_solutions: dict[int, Solution] = {}
        primary_outcomes: dict[int, _Outcome] = {}
        units_by_index: dict[int, dict] = {}

        for start in range(0, len(examples), self._batch_size):
            batch = examples[start : start + self._batch_size]
            try:
                primary = self._collect(
                    self._solve, self._build_request(batch), len(batch), "Verifier"
                )
            except Exception as exc:
                for local_index, example in enumerate(batch):
                    global_index = start + local_index
                    results[global_index] = ValidationResult(
                        original=example,
                        status="uncertain",
                        reason=f"Verifier batch failed: {exc}",
                        reason_code="verifier_batch_failure",
                        retryable=True,
                    )
                continue

            for local_index, example in enumerate(batch):
                global_index = start + local_index
                solution = primary[local_index]
                units = self._units_for(example)
                outcome = self._assess(solution, units=units)
                decision = self._to_result(example, solution, outcome, units=units)
                primary_solutions[global_index] = solution
                primary_outcomes[global_index] = outcome
                units_by_index[global_index] = units
                needs_cross_check = self._cross_check is not None and (
                    decision.status == "corrected"
                    or (
                        decision.status == "accepted"
                        and rng.random() < self._audit_fraction
                    )
                )
                if needs_cross_check:
                    pending_cross_check.append(global_index)
                elif decision.status == "corrected" and self._require_cross_check:
                    results[global_index] = ValidationResult(
                        original=example,
                        status="uncertain",
                        expression=solution.expression,
                        answer_unit=solution.requested_unit,
                        expected_answer=decision.expected_answer,
                        reason="Correction requires an independent cross-check.",
                        reason_code="cross_check_required",
                        retryable=False,
                    )
                else:
                    results[global_index] = decision

        if self._cross_check is not None:
            for start in range(
                0, len(pending_cross_check), self._cross_check_batch_size
            ):
                indices = pending_cross_check[
                    start : start + self._cross_check_batch_size
                ]
                batch = [examples[index] for index in indices]
                try:
                    secondary = self._collect(
                        self._cross_check,
                        self._build_request(batch),
                        len(batch),
                        "Cross-check",
                    )
                except Exception as exc:
                    for index in indices:
                        results[index] = ValidationResult(
                            original=examples[index],
                            status="uncertain",
                            reason=f"Cross-check failed: {exc}",
                            reason_code="cross_check_batch_failure",
                            retryable=True,
                        )
                    continue
                for local_index, global_index in enumerate(indices):
                    units = units_by_index[global_index]
                    secondary_outcome = self._assess(
                        secondary[local_index], units=units
                    )
                    final_outcome = self._reconcile(
                        primary_outcomes[global_index], secondary_outcome, units=units
                    )
                    results[global_index] = self._to_result(
                        examples[global_index],
                        primary_solutions[global_index],
                        final_outcome,
                        units=units,
                    )
        if any(result is None for result in results):
            raise RuntimeError("MathValidator failed to produce all results.")
        return [result for result in results if result is not None]

    @staticmethod
    def _collect(
        solver: SolveBatch, request: str, size: int, label: str
    ) -> dict[int, Solution]:
        """Call a solver and check that it returned exactly one solution per problem."""
        response = solver(request)
        ids = [item.id for item in response.solutions]
        if sorted(ids) != list(range(size)):
            raise ValueError(f"{label} returned missing, duplicate or unexpected IDs.")
        return {item.id: item for item in response.solutions}

    def _build_request(self, examples: Sequence[Example]) -> str:
        """Build the solver prompt for a batch of problems."""
        problems = [
            {"id": index, "problem": example.input}
            for index, example in enumerate(examples)
        ]
        return (
            f"{MATH_VALIDATION_TEMPLATE.strip()}\n\n"
            f"Unit context: {self._unit_context}\n"
            f"Explicit unit aliases: {json.dumps(self._units['aliases'])}\n\n"
            f"Problems:\n{json.dumps(problems, ensure_ascii=False)}"
        )

    def _units_for(self, example: Example) -> dict:
        """Derive contextual unit aliases from the problem text."""

        aliases = dict(self._units["aliases"])
        existing = {_alias_key(key) for key in aliases}
        problem = example.input
        registry = self._units["registry"]

        detected_currencies = {
            code
            for code, pattern in C.CURRENCY_CONTEXTS
            if re.search(pattern, problem, re.I)
        }

        if not detected_currencies and "$" in problem:
            detected_currencies.add("usd")

        if len(detected_currencies) == 1:
            currency = next(iter(detected_currencies))

            for name in C.CURRENCY_ALIASES[currency]:
                if _alias_key(name) not in existing:
                    aliases[name] = (
                        "us_cent"
                        if currency == "usd" and name in {"cent", "cents"}
                        else currency
                    )

        for canonical, spellings in C.OPAQUE_UNITS:
            if any(
                re.search(rf"(?<!\w){re.escape(name)}(?!\w)", problem, re.I)
                for name in spellings
            ):
                definition = f"unspecified_{canonical}"
                _define(registry, definition, f"[{definition}]")

                for name in spellings:
                    if _alias_key(name) not in existing:
                        aliases[name] = definition

        if C.PAGE_PATTERN.search(problem):
            for name in C.PAGE_ALIASES:
                if _alias_key(name) not in existing:
                    aliases[name] = "count:page"

        return dict(registry=registry, aliases=aliases)

    def _assess(self, solution: Solution, *, units: Mapping | None = None) -> _Outcome:
        """Check a solution and compute the expected answer."""
        if solution.status != "solved":
            if solution.status == "invalid":
                return _Outcome(
                    "rejected", solution.reason, reason_code="invalid_problem"
                )
            return _uncertain("uncertain_interpretation", solution.reason)
        if not solution.expression:
            return _uncertain("missing_expression", "Verifier returned no expression.")
        try:
            evaluation = evaluate_expression(solution.expression)
        except (
            SyntaxError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
            RecursionError,
        ) as exc:
            return _uncertain("invalid_expression", f"Cannot verify arithmetic: {exc}")
        try:
            expected = convert_units(
                evaluation.value,
                solution.computed_unit,
                solution.requested_unit,
                **(units if units is not None else self._units),
            )
            if max(abs(expected.numerator), expected.denominator) > C.LIMIT:
                raise ValueError("Numerical limits exceeded.")
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return _uncertain(
                getattr(exc, "code", "unsupported_conversion"),
                f"Cannot verify units or converted value: {exc}",
            )
        if evaluation.rounded and not solution.rounding_reason.strip():
            return _uncertain("unjustified_rounding", "Rounding has no justification.")
        if solution.must_be_integer and expected.denominator != 1:
            return _Outcome(
                "rejected",
                "Fractional answer for indivisible count.",
                reason_code="noninteger_count",
            )
        if solution.must_be_nonnegative and expected < 0:
            return _Outcome(
                "rejected", "Negative final quantity.", reason_code="negative_quantity"
            )
        return _Outcome("ok", expected=expected, unit=solution.requested_unit)

    def _reconcile(
        self, primary: _Outcome, secondary: _Outcome, *, units: Mapping | None = None
    ) -> _Outcome:
        """Combine primary and cross-check outcomes into one outcome."""
        if primary.status != "ok":
            return _uncertain(
                "verifier_disagreement",
                f"Primary verifier status: {primary.status}; cross-check status: {secondary.status}.",
            )
        if secondary.status != "ok":
            if secondary.retryable:
                return replace(secondary, reason=f"Cross-check: {secondary.reason}")
            return _uncertain(
                "verifier_disagreement",
                f"Cross-check status: {secondary.status}. {secondary.reason}",
            )
        if any(
            value is None
            for value in (
                primary.expected,
                primary.unit,
                secondary.expected,
                secondary.unit,
            )
        ):
            return _uncertain(
                "verifier_disagreement", "Verifier returned an incomplete result."
            )
        try:
            value = convert_units(
                secondary.expected,
                secondary.unit,
                primary.unit,
                **(units if units is not None else self._units),
            )
        except ValueError as exc:
            return _uncertain(
                "verifier_disagreement", f"Incompatible cross-check units: {exc}"
            )
        if value != primary.expected:
            return _uncertain(
                "verifier_disagreement",
                f"Values differ: {primary.expected} vs {value} [{primary.unit}]",
            )
        return primary

    def _to_result(
        self,
        example: Example,
        solution: Solution,
        outcome: _Outcome,
        *,
        units: Mapping | None = None,
    ) -> ValidationResult:
        """Convert an outcome into a validation result."""
        expected = outcome.expected
        details = dict(
            original=example,
            expression=solution.expression,
            answer_unit=solution.requested_unit,
            expected_answer=format_number(expected) if expected is not None else None,
        )
        if outcome.status != "ok":
            return ValidationResult(
                **details,
                status=outcome.status,
                reason=outcome.reason,
                reason_code=outcome.reason_code,
                retryable=outcome.retryable,
            )
        assert (
            expected is not None
            and solution.expression is not None
            and solution.requested_unit is not None
        )
        try:
            actual = parse_answer_with_unit(
                example.output,
                solution.requested_unit,
                **(units if units is not None else self._units),
            )
        except (ValueError, ZeroDivisionError) as exc:
            return ValidationResult(
                **details,
                status="uncertain",
                reason=f"Cannot verify original answer: {exc}",
                reason_code=(
                    exc.code
                    if isinstance(exc, UnitError)
                    else "unsupported_target_format"
                ),
            )
        if actual == expected:
            return ValidationResult(
                **details, status="accepted", verified_output=example.output
            )
        answer = format_number(expected)
        output = answer
        if "####" in example.output:
            value = format_number(calculate(solution.expression))
            lines = [f"{solution.expression} = {value} [{solution.computed_unit}]"]
            if not _same_unit(
                solution.computed_unit,
                solution.requested_unit,
                **(units if units is not None else self._units),
            ):
                lines.append(f"Converted to {solution.requested_unit}: {answer}")
            output = "\n".join([*lines, f"#### {answer}"])
        return ValidationResult(
            **details,
            status="corrected",
            corrected=example.model_copy(update={"output": output}),
            verified_output=output,
            reason_code="answer_mismatch",
            reason="Target replaced with independently computed answer.",
        )


def create_math_validator(
    model: BaseLanguageModel,
    *,
    cross_check_model: BaseLanguageModel | None = None,
    require_cross_check: bool = False,
    **kwargs,
) -> MathValidator:
    """Create a math validator from language models."""

    def solver(llm: BaseLanguageModel) -> SolveBatch:
        """Wrap a language model as a structured batch solver."""
        return lambda request: parse_structured(
            llm, request, SolutionBatch, label="Math validator"
        )

    return MathValidator(
        solve=solver(model),
        cross_check=(
            solver(cross_check_model) if cross_check_model is not None else None
        ),
        require_cross_check=require_cross_check,
        **kwargs,
    )
