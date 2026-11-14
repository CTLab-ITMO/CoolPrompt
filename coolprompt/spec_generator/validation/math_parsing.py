"""Arithmetic expression evaluation and answer parsing for math validation."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import pint

from coolprompt.spec_generator.validation.math_constants import (
    MathValidationConstants as C,
)
from coolprompt.spec_generator.validation.math_units import (
    _UREG,
    UnitError,
    _alias_key,
    convert_units,
)


@dataclass(frozen=True)
class Evaluation:
    """Exact value of an expression and whether rounding was applied."""

    value: Fraction
    rounded: bool


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
        """Recursively evaluate a supported AST node."""
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
    """Evaluate an expression and return its exact value."""
    return evaluate_expression(expression).value


def _decimal_places(value: Fraction) -> int | None:
    """Return the number of decimal places, or None if non-terminating."""
    n, places = value.denominator, [0, 0]
    for index, factor in enumerate((2, 5)):
        while n % factor == 0:
            n //= factor
            places[index] += 1
    return max(places) if n == 1 else None


def is_terminating_decimal(value: Fraction) -> bool:
    """Return True if the value has a finite decimal form."""
    return _decimal_places(value) is not None


def format_number(value: Fraction) -> str:
    """Format a fraction as a decimal string, or as a fraction if non-terminating."""
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
    """Parse the final numeric answer after the "####" marker."""
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
    """Return the text after the "####" marker, or the whole output."""
    if output.count("####") > 1:
        raise ValueError("Multiple answer markers.")
    return output.rsplit("####", 1)[-1].strip() if "####" in output else output.strip()


def _normalize_answer_unit(unit: str) -> str:
    """Normalize Unicode symbols and metric powers in an answer unit."""
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
    """Resolve a currency symbol to a currency code."""
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
    """Parse a numeric answer with an optional unit and convert it to the requested unit."""
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
