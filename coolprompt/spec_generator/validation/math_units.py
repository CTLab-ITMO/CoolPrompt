"""Unit normalization and exact unit conversion for math validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from fractions import Fraction

import pint

from coolprompt.spec_generator.validation.math_constants import (
    MathValidationConstants as C,
)

_UREG = pint.UnitRegistry(non_int_type=Fraction)


class UnitError(ValueError):
    """Unit parsing or conversion error with a machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _alias_key(text: str) -> str:
    """Return a case- and whitespace-insensitive key for alias lookup."""
    return " ".join(text.strip().casefold().split())


def _normalize(text: str) -> str:
    """Rewrite "per", qualified, square, and cubic notation into Pint syntax."""
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
        """Replace a single unit token with its synonym, if any."""
        token = match.group(0)
        return synonyms.get(_alias_key(token), token)

    return _normalize(C.TOKEN.sub(replace_token, source))


def _define(registry: pint.UnitRegistry, name: str, definition: str) -> None:
    """Define a unit in the registry unless it already exists."""
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
            """Normalize a "count:<entity>" unit to a safe identifier."""
            entity = match.group(1).strip().casefold()
            entity = re.sub(r"[^a-z0-9_]+", "_", entity).strip("_")
            if not C.COUNT_ENTITY.fullmatch(entity):
                raise UnitError(
                    "unsupported_unit", f"Unsupported count entity: {match.group(0)!r}"
                )
            return f"count:{entity}"

        text = re.sub(r"count:([^/*()^]+)", normalize_count, text, flags=re.I)

        def resolve(match: re.Match[str]) -> str:
            """Map a token to a registry unit, rejecting ambiguous units."""
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
    """Return the dimensionality of a unit as a string."""
    if unit is None:
        raise UnitError("missing_unit", "Missing unit.")
    if _temperature_kind(normalize_unit(unit, aliases)) is not None:
        return "[temperature]"
    return str(_parse_unit(unit, registry, aliases).dimensionality)


def _canonical_temperature_unit(unit: str) -> str:
    """Map a temperature alias to its canonical name."""
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
    """Convert an absolute temperature to kelvin."""
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
    """Convert an absolute temperature from kelvin."""
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
    """Convert a temperature difference to kelvin."""
    if unit in {"delta_kelvin", "delta_degC"}:
        return value
    if unit in {"delta_degF", "delta_degR"}:
        return value * Fraction(5, 9)
    raise UnitError("unsupported_unit", f"Unsupported temperature difference: {unit!r}")


def _kelvin_to_temperature_delta(value: Fraction, unit: str) -> Fraction:
    """Convert a temperature difference from kelvin."""
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
    """Return True if both units resolve to the same unit."""
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
