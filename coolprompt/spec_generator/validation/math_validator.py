"""Validate arithmetic word problems with exact calculations and unit conversions."""

from __future__ import annotations

import ast
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
from coolprompt.utils.prompt_templates.math_validation_template import (
    MATH_VALIDATION_TEMPLATE,
)

_UREG = pint.UnitRegistry(non_int_type=Fraction)


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
    model_config = ConfigDict(extra="forbid")
    solutions: list[Solution]


class ValidationResult(BaseModel):
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
class Evaluation:
    value: Fraction
    rounded: bool


@dataclass(frozen=True)
class _Outcome:
    status: Literal["ok", "rejected", "uncertain"]
    reason: str = ""
    expected: Fraction | None = None
    unit: str | None = None
    reason_code: str = ""
    retryable: bool = False


class UnitError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _alias_key(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _normalize(text: str) -> str:
    text = re.sub(r"\s+per\s+|_per_", "/", text, flags=re.I)
    text = C.QUALIFIED.sub(
        lambda m: ("US_liquid" if m[1].lower() == "us" else m[1].lower())
        + "_"
        + m[2].lower(),
        text,
    )
    text = re.sub(
        r"\b(square|sq\.?|cubic|cu\.?)[ _]+([A-Za-z]+)\b",
        lambda m: f"({m[2]}**{3 if m[1].lower().startswith('cu') else 2})",
        text,
    )
    return re.sub(r"\b(mm|cm|km|m)([23])\b", r"\1**\2", text)


def normalize_unit(unit: str, aliases: Mapping[str, str] | None = None) -> str:
    """Resolve aliases and normalize unit notation."""
    if not isinstance(unit, str) or not unit.strip() or len(unit) > 200:
        raise UnitError("unsupported_unit", "Missing or oversized unit.")
    explicit: dict[str, str] = {}
    for key, value in (aliases or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise UnitError("unsupported_unit", "Unit aliases must be strings.")
        normalized_key, target = _alias_key(key), value.strip()
        if not normalized_key or not target or len(target) > 200:
            raise UnitError("unsupported_unit", "Empty or oversized unit alias.")
        if normalized_key in explicit and explicit[normalized_key] != target:
            raise UnitError("unsupported_unit", f"Conflicting unit aliases: {key!r}")
        explicit[normalized_key] = target

    source = " ".join(unit.split())
    synonyms = {**C.UNIT_SYNONYMS, **explicit}
    if _alias_key(source) in synonyms:
        return _normalize(synonyms[_alias_key(source)])

    multiword = sorted((key for key in synonyms if " " in key), key=len, reverse=True)
    if multiword:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_:])(?:"
            + "|".join(re.escape(key).replace(r"\ ", r"\s+") for key in multiword)
            + r")(?![A-Za-z0-9_])",
            re.I,
        )
        source = pattern.sub(lambda m: synonyms[_alias_key(m.group(0))], source)

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        return synonyms.get(_alias_key(token), token)

    return _normalize(C.TOKEN.sub(replace_token, source))


def _define(registry: pint.UnitRegistry, name: str, definition: str) -> None:
    if name not in registry:
        registry.define(f"{name} = {definition}")


def _parse_unit(
    unit: str,
    registry: pint.UnitRegistry = _UREG,
    aliases: Mapping[str, str] | None = None,
) -> pint.Unit:
    """Parse and validate a unit using Pint."""
    try:
        for name, definition in C.DEFINITIONS:
            _define(registry, name, definition)
        text = normalize_unit(unit, aliases)

        def normalize_count(match: re.Match[str]) -> str:
            entity = match.group(1).strip().casefold()
            entity = re.sub(r"[^a-z0-9_]+", "_", entity).strip("_")
            if not C.COUNT_ENTITY.fullmatch(entity):
                raise UnitError(
                    "unsupported_unit", f"Unsupported count entity: {match.group(0)!r}"
                )
            return f"count:{entity}"

        text = re.sub(r"count:([^/*()^]+)", normalize_count, text, flags=re.I)

        def resolve(match: re.Match[str]) -> str:
            token = match.group(0)
            name = token.lower()
            if name.startswith("count:"):
                entity = name[6:]
                if not C.COUNT_ENTITY.fullmatch(entity):
                    raise UnitError(
                        "unsupported_unit", f"Unsupported count entity: {token!r}"
                    )
                name = f"count_{entity}"
                _define(registry, name, f"[{name}]")
                return name
            if name.removesuffix("s") in C.AMBIGUOUS:
                raise UnitError("ambiguous_unit", f"Ambiguous unit: {token!r}")
            return name if name in C.CURRENCIES else token

        parsed = registry.parse_units(C.TOKEN.sub(resolve, text))
        result = registry.dimensionless
        for name, power in parsed._units.items():
            valid_power = (
                power.denominator == 1
                if isinstance(power, Fraction)
                else float(power).is_integer()
            )
            if not valid_power:
                raise UnitError(
                    "unsupported_unit", "Fractional unit powers are unsupported."
                )
            resolved_name = f"calendar_{name}" if name in C.CALENDAR else name
            result *= registry.Unit(resolved_name) ** int(power)
        return result
    except UnitError:
        raise
    except Exception as exc:
        raise UnitError(
            "unsupported_unit", f"Unsupported unit: {unit!r} ({exc})"
        ) from exc


def unit_dimension(
    unit: str | None,
    *,
    registry: pint.UnitRegistry = _UREG,
    aliases: Mapping[str, str] | None = None,
) -> str:
    if unit is None:
        raise UnitError("missing_unit", "Missing unit.")
    if _temperature_kind(normalize_unit(unit, aliases)) is not None:
        return "[temperature]"
    return str(_parse_unit(unit, registry, aliases).dimensionality)


def _canonical_temperature_unit(unit: str) -> str:
    return C.TEMPERATURE_ALIASES.get(unit, unit)


def _temperature_kind(unit: str) -> str | None:
    """Classify standalone temperature units; reject compound expressions."""
    unit = _canonical_temperature_unit(unit)
    if unit in C.ABSOLUTE_TEMPERATURE_UNITS:
        return "absolute"
    if unit in C.DELTA_TEMPERATURE_UNITS:
        return "delta"
    if C.TEMPERATURE_TOKEN.search(unit):
        raise UnitError(
            "unsupported_unit", f"Compound temperature unit is not supported: {unit!r}"
        )
    return None


def _temperature_to_kelvin(value: Fraction, unit: str) -> Fraction:
    if unit == "kelvin":
        return value
    if unit == "degC":
        return value + Fraction(27315, 100)
    if unit == "degF":
        return (value - 32) * Fraction(5, 9) + Fraction(27315, 100)
    if unit == "degR":
        return value * Fraction(5, 9)
    raise UnitError("unsupported_unit", f"Unsupported absolute temperature: {unit!r}")


def _kelvin_to_temperature(value: Fraction, unit: str) -> Fraction:
    if unit == "kelvin":
        return value
    if unit == "degC":
        return value - Fraction(27315, 100)
    if unit == "degF":
        return (value - Fraction(27315, 100)) * Fraction(9, 5) + 32
    if unit == "degR":
        return value * Fraction(9, 5)
    raise UnitError("unsupported_unit", f"Unsupported absolute temperature: {unit!r}")


def _temperature_delta_to_kelvin(value: Fraction, unit: str) -> Fraction:
    if unit in {"delta_kelvin", "delta_degC"}:
        return value
    if unit in {"delta_degF", "delta_degR"}:
        return value * Fraction(5, 9)
    raise UnitError("unsupported_unit", f"Unsupported temperature difference: {unit!r}")


def _kelvin_to_temperature_delta(value: Fraction, unit: str) -> Fraction:
    if unit in {"delta_kelvin", "delta_degC"}:
        return value
    if unit in {"delta_degF", "delta_degR"}:
        return value * Fraction(9, 5)
    raise UnitError("unsupported_unit", f"Unsupported temperature difference: {unit!r}")


def convert_units(
    value: Fraction,
    computed_unit: str | None,
    requested_unit: str | None,
    *,
    registry: pint.UnitRegistry = _UREG,
    aliases: Mapping[str, str] | None = None,
) -> Fraction:
    """Convert compatible units exactly, keeping temperature deltas separate."""
    if computed_unit is None or requested_unit is None:
        raise UnitError("missing_unit", "Missing unit.")
    if type(value) not in (int, Fraction):
        raise ValueError("Value must be an int or Fraction.")
    value = Fraction(value)
    source_text = _canonical_temperature_unit(normalize_unit(computed_unit, aliases))
    target_text = _canonical_temperature_unit(normalize_unit(requested_unit, aliases))
    source_kind = _temperature_kind(source_text)
    target_kind = _temperature_kind(target_text)

    if source_kind is not None or target_kind is not None:
        if source_kind != target_kind:
            raise UnitError(
                "incompatible_temperature_kind",
                f"Cannot convert {computed_unit!r} to {requested_unit!r}: "
                "absolute temperatures and temperature differences are not interchangeable.",
            )
        if source_kind == "absolute":
            result = _kelvin_to_temperature(
                _temperature_to_kelvin(value, source_text), target_text
            )
        elif source_kind == "delta":
            result = _kelvin_to_temperature_delta(
                _temperature_delta_to_kelvin(value, source_text), target_text
            )
        else:
            raise UnitError("unsupported_unit", "Unsupported temperature conversion.")
        if max(abs(result.numerator), result.denominator) > C.LIMIT:
            raise ValueError("Numerical limits exceeded.")
        return result

    source = _parse_unit(computed_unit, registry, aliases)
    target = _parse_unit(requested_unit, registry, aliases)
    try:
        result = registry.Quantity(value, source).to(target).magnitude
    except Exception as exc:
        calendar = any("[calendar]" in unit.dimensionality for unit in (source, target))
        time = any("[time]" in unit.dimensionality for unit in (source, target))
        code = "ambiguous_calendar" if calendar and time else "incompatible_units"
        raise UnitError(
            code, f"Cannot convert {computed_unit!r} to {requested_unit!r}"
        ) from exc
    if not isinstance(result, (int, Fraction)) or isinstance(result, float):
        raise UnitError("inexact_conversion", "Inexact unit conversion.")
    result = Fraction(result)
    if max(abs(result.numerator), result.denominator) > C.LIMIT:
        raise ValueError("Numerical limits exceeded.")
    return result


def _same_unit(
    first: str | None,
    second: str | None,
    *,
    registry: pint.UnitRegistry = _UREG,
    aliases: Mapping[str, str] | None = None,
) -> bool:
    if first is None or second is None:
        return False
    try:
        first_text = _canonical_temperature_unit(normalize_unit(first, aliases))
        second_text = _canonical_temperature_unit(normalize_unit(second, aliases))
        first_kind = _temperature_kind(first_text)
        second_kind = _temperature_kind(second_text)
        if first_kind is not None or second_kind is not None:
            return first_kind == second_kind and first_text == second_text
        return _parse_unit(first, registry, aliases) == _parse_unit(
            second, registry, aliases
        )
    except ValueError:
        return False


def evaluate_expression(expression: str) -> Evaluation:
    """Evaluate a restricted arithmetic AST with exact fractions."""
    expression = expression.strip()
    if not 0 < len(expression) <= 400:
        raise ValueError("Invalid expression.")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Expression is too complex.")
    rounded = False

    def evaluate(node: ast.AST) -> Fraction:
        nonlocal rounded
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            if not literal or not C.NUMBER_LITERAL.fullmatch(literal):
                raise ValueError("Unsupported numeric literal.")
            value = Fraction(literal)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in C.OPERATIONS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.FloorDiv):
                quotient, remainder = divmod(left, right)
                value = Fraction(quotient)
                rounded |= bool(remainder)
            else:
                value = Fraction(C.OPERATIONS[type(node.op)](left, right))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in C.ROUNDING_FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            argument = evaluate(node.args[0])
            value = Fraction(C.ROUNDING_FUNCTIONS[node.func.id](argument))
            rounded |= value != argument
        else:
            raise ValueError("Unsupported expression.")
        if max(abs(value.numerator), value.denominator) > C.LIMIT:
            raise ValueError("Numerical limits exceeded.")
        return value

    return Evaluation(evaluate(tree.body), rounded)


def calculate(expression: str) -> Fraction:
    return evaluate_expression(expression).value


def _decimal_places(value: Fraction) -> int | None:
    n, places = value.denominator, [0, 0]
    for index, factor in enumerate((2, 5)):
        while n % factor == 0:
            n //= factor
            places[index] += 1
    return max(places) if n == 1 else None


def is_terminating_decimal(value: Fraction) -> bool:
    return _decimal_places(value) is not None


def format_number(value: Fraction) -> str:
    places = _decimal_places(value)
    if places is None:
        return str(value)
    if not places:
        return str(value.numerator)
    digits = str(abs(value.numerator) * 10**places // value.denominator).zfill(
        places + 1
    )
    return f"{'-' if value < 0 else ''}{digits[:-places]}.{digits[-places:]}".rstrip(
        "0"
    ).rstrip(".")


def parse_answer(output: str) -> Fraction:
    if output.count("####") > 1:
        raise ValueError("Multiple answer markers.")
    value = output.rsplit("####", 1)[-1].strip()
    if len(value) > 100:
        raise ValueError("Oversized answer.")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", value):
        raise ValueError("Invalid numerical answer.")
    result = Fraction(value)
    if max(abs(result.numerator), result.denominator) > C.LIMIT:
        raise ValueError("Numerical limits exceeded.")
    return result


def _extract_answer_text(output: str) -> str:
    if output.count("####") > 1:
        raise ValueError("Multiple answer markers.")
    return output.rsplit("####", 1)[-1].strip() if "####" in output else output.strip()


def _normalize_answer_unit(unit: str) -> str:
    unit = unit.strip().translate(C.UNICODE_UNIT_TRANSLATION)
    return re.sub(
        r"\b([A-Za-z]+)([23])\b",
        lambda m: (
            f"{m.group(1)}**{m.group(2)}"
            if m.group(1).lower() in {"m", "cm", "mm", "km"}
            else m.group(0)
        ),
        unit,
    )


def _currency_from_context(symbol: str, aliases: Mapping[str, str] | None) -> str:
    if symbol not in C.CURRENCY_SYMBOLS:
        raise UnitError("unsupported_unit", f"Unsupported currency symbol: {symbol!r}")
    currency = C.CURRENCY_SYMBOLS[symbol]
    if currency is not None:
        return currency
    normalized_aliases = {
        _alias_key(key): value for key, value in (aliases or {}).items()
    }
    currency = normalized_aliases.get(symbol)
    if currency is None:
        raise UnitError(
            "ambiguous_unit", f"Currency symbol {symbol!r} requires an explicit alias."
        )
    currency = currency.strip().lower()
    if currency not in C.CURRENCIES:
        raise UnitError(
            "unsupported_unit",
            f"Unsupported currency alias for {symbol!r}: {currency!r}",
        )
    return currency


def parse_answer_with_unit(
    output: str,
    requested_unit: str,
    *,
    registry: pint.UnitRegistry = _UREG,
    aliases: Mapping[str, str] | None = None,
) -> Fraction:
    raw = _extract_answer_text(output)
    if not raw or len(raw) > 200:
        raise ValueError("Missing or oversized final answer.")
    try:
        return parse_answer(output)
    except ValueError:
        pass
    percent_match = re.fullmatch(rf"\s*(?P<number>(?:{C.ANSWER_NUMBER}))\s*%\s*", raw)
    if percent_match:
        return convert_units(
            parse_answer(percent_match.group("number")),
            "percent",
            requested_unit,
            registry=registry,
            aliases=aliases,
        )
    if raw[0] in C.CURRENCY_SYMBOLS:
        symbol = raw[0]
        value = parse_answer(raw[1:].strip())
        source_unit = _currency_from_context(symbol, aliases)
        return convert_units(
            value, source_unit, requested_unit, registry=registry, aliases=aliases
        )
    match = C.ANSWER_PATTERN.fullmatch(raw)
    if match is None:
        raise ValueError("Unsupported numerical answer format.")
    value = parse_answer(match.group("number"))
    source_unit = _normalize_answer_unit(match.group("unit"))
    if not source_unit:
        return value
    if source_unit in C.CURRENCY_SYMBOLS:
        source_unit = _currency_from_context(source_unit, aliases)
    return convert_units(
        value, source_unit, requested_unit, registry=registry, aliases=aliases
    )


def _uncertain(code: str, reason: str) -> _Outcome:
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
        response = solver(request)
        ids = [item.id for item in response.solutions]
        if sorted(ids) != list(range(size)):
            raise ValueError(f"{label} returned missing, duplicate or unexpected IDs.")
        return {item.id: item for item in response.solutions}

    def _build_request(self, examples: Sequence[Example]) -> str:
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
