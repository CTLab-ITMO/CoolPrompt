"""Constants used by the arithmetic word-problem validator."""

from __future__ import annotations

import math
import operator
import re

from collections.abc import Callable
from fractions import Fraction
from typing import ClassVar

import ast


class MathValidationConstants:
    """Configuration constants for mathematical validation."""

    LIMIT: ClassVar[int] = 10**30

    CALENDAR: ClassVar[frozenset[str]] = frozenset(
        {
            "month",
            "year",
            "decade",
            "century",
            "millennium",
        }
    )

    CURRENCIES: ClassVar[tuple[str, ...]] = (
        "rub",
        "usd",
        "eur",
        "gbp",
        "jpy",
        "cad",
        "aud",
        "chf",
        "cny",
        "inr",
        "php",
    )

    DEFINITIONS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("calendar_month", "[calendar]"),
        ("calendar_year", "12 * calendar_month"),
        ("calendar_decade", "10 * calendar_year"),
        ("calendar_century", "10 * calendar_decade"),
        ("calendar_millennium", "10 * calendar_century"),
        ("percentage_point", "[percentage_change]"),
        ("us_cent", "1 / 100 * usd"),
        *((code, f"[currency_{code}]") for code in CURRENCIES),
    )

    AMBIGUOUS: ClassVar[frozenset[str]] = frozenset(
        {
            "ton",
            "oz",
            "ounce",
            "gal",
            "gallon",
            "qt",
            "quart",
            "pt",
            "pint",
            "cup",
        }
    )

    RETRYABLE: ClassVar[frozenset[str]] = frozenset(
        {
            "missing_expression",
            "invalid_expression",
            "missing_unit",
            "unsupported_unit",
            "incompatible_units",
            "unjustified_rounding",
            "inexact_conversion",
        }
    )

    OPERATIONS: ClassVar[
        dict[type[ast.AST], Callable[[Fraction, Fraction], Fraction | int]]
    ] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
    }

    ROUNDING_FUNCTIONS: ClassVar[dict[str, Callable]] = {
        "floor": math.floor,
        "ceil": math.ceil,
    }

    QUALIFIED: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(short|long|metric|fluid|troy|avoirdupois|imperial|us)[ _]+"
        r"(ton|ounce|gallon|quart|pint|cup)s?\b",
        re.I,
    )

    TOKEN: ClassVar[re.Pattern[str]] = re.compile(r"[^\s/*()^]+")

    COUNT_ENTITY: ClassVar[re.Pattern[str]] = re.compile(r"[a-z0-9_]{1,40}")

    NUMBER_LITERAL: ClassVar[re.Pattern[str]] = re.compile(r"(?:\d+\.?\d*|\.\d+)")

    UNIT_SYNONYMS: ClassVar[dict[str, str]] = {
        "us_dollar": "usd",
        "us_dollars": "usd",
        "u.s._dollar": "usd",
        "u.s._dollars": "usd",
        "us dollar": "usd",
        "us dollars": "usd",
        "u.s. dollar": "usd",
        "u.s. dollars": "usd",
        "united states dollar": "usd",
        "united states dollars": "usd",
        "american dollar": "usd",
        "american dollars": "usd",
        "canadian dollar": "cad",
        "canadian dollars": "cad",
        "australian dollar": "aud",
        "australian dollars": "aud",
        "euro": "eur",
        "euros": "eur",
        "british pound": "gbp",
        "british pounds": "gbp",
        "pound sterling": "gbp",
        "pounds sterling": "gbp",
        "japanese yen": "jpy",
        "chinese yuan": "cny",
        "indian rupee": "inr",
        "swiss franc": "chf",
        "swiss francs": "chf",
        "russian ruble": "rub",
        "russian rubles": "rub",
        "russian rouble": "rub",
        "russian roubles": "rub",
        "philippine peso": "php",
        "philippine pesos": "php",
        "us cent": "us_cent",
        "us cents": "us_cent",
        "american cent": "us_cent",
        "american cents": "us_cent",
        "unitless": "dimensionless",
        "no unit": "dimensionless",
        "percentage": "percent",
        "percentages": "percent",
        "percentage points": "percentage_point",
        "sq m": "meter**2",
        "sq meter": "meter**2",
        "sq meters": "meter**2",
        "square meter": "meter**2",
        "square meters": "meter**2",
        "cu m": "meter**3",
        "cu meter": "meter**3",
        "cu meters": "meter**3",
        "cubic meter": "meter**3",
        "cubic meters": "meter**3",
        "person": "count:person",
        "people": "count:person",
    }

    ABSOLUTE_TEMPERATURE_UNITS: ClassVar[frozenset[str]] = frozenset(
        {
            "kelvin",
            "degC",
            "degF",
            "degR",
        }
    )

    DELTA_TEMPERATURE_UNITS: ClassVar[frozenset[str]] = frozenset(
        {
            "delta_kelvin",
            "delta_degC",
            "delta_degF",
            "delta_degR",
        }
    )

    TEMPERATURE_ALIASES: ClassVar[dict[str, str]] = {
        "degree_Celsius": "degC",
        "degree_Fahrenheit": "degF",
        "degree_Rankine": "degR",
        "degree_Kelvin": "kelvin",
        "delta_degree_Celsius": "delta_degC",
        "delta_degree_Fahrenheit": "delta_degF",
        "delta_degree_Rankine": "delta_degR",
        "delta_degree_Kelvin": "delta_kelvin",
    }

    TEMPERATURE_TOKEN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?:delta_degree_Celsius|delta_degree_Fahrenheit|"
        r"delta_degree_Rankine|delta_degree_Kelvin|"
        r"degree_Celsius|degree_Fahrenheit|degree_Rankine|"
        r"degree_Kelvin|delta_kelvin|delta_degC|delta_degF|"
        r"delta_degR|kelvin|degC|degF|degR)"
        r"(?![A-Za-z0-9_])"
    )

    ANSWER_NUMBER: ClassVar[str] = (
        r"[+-]?"
        r"(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d*)?"
        r"(?:/[+-]?\d+)?"
        r"|[+-]?\.\d+(?:/[+-]?\d+)?"
    )

    ANSWER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        rf"^\s*(?P<number>(?:{ANSWER_NUMBER}))" rf"\s*(?P<unit>.*?)\s*$",
        re.DOTALL,
    )

    CURRENCY_SYMBOLS: ClassVar[dict[str, str | None]] = {
        "€": "eur",
        "£": "gbp",
        "¥": None,
        "$": None,
        "₽": "rub",
        "₹": "inr",
    }

    UNICODE_UNIT_TRANSLATION: ClassVar[dict[int, str]] = str.maketrans(
        {
            "²": "2",
            "³": "3",
            "·": "*",
            "×": "*",
            "−": "-",
        }
    )

    CURRENCY_CONTEXTS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("rub", r"\b(?:Russian\s+(?:rubles?|roubles?)|RUB)\b"),
        ("cad", r"\b(?:Canadian\s+dollars?|CAD)\b"),
        ("aud", r"\b(?:Australian\s+dollars?|AUD)\b"),
        ("usd", r"\b(?:(?:US|U\.S\.|American)\s+dollars?|USD)\b"),
        ("eur", r"\b(?:euros?|EUR)\b"),
        ("gbp", r"\b(?:(?:British\s+)?pounds?\s+sterling|British\s+pounds?|GBP)\b"),
        ("jpy", r"\b(?:Japanese\s+yen|JPY)\b"),
        ("cny", r"\b(?:Chinese\s+yuan|CNY|RMB)\b"),
        ("inr", r"\b(?:Indian\s+rupees?|INR)\b"),
        ("chf", r"\b(?:Swiss\s+francs?|CHF)\b"),
        ("php", r"\b(?:Philippine\s+pesos?|PHP)\b"),
    )

    CURRENCY_ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "rub": ("ruble", "rubles", "rouble", "roubles", "₽"),
        "cad": ("dollar", "dollars", "$"),
        "aud": ("dollar", "dollars", "$"),
        "usd": ("dollar", "dollars", "$", "cent", "cents"),
        "eur": ("euro", "euros", "€"),
        "gbp": ("pound", "pounds", "£"),
        "jpy": ("yen", "¥"),
        "cny": ("yuan", "¥"),
        "inr": ("rupee", "rupees", "₹"),
        "chf": ("franc", "francs"),
        "php": ("peso", "pesos"),
    }

    OPAQUE_UNITS: ClassVar[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("cup", ("cup", "cups")),
        ("ounce", ("ounce", "ounces", "oz")),
    )

    PAGE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\bpages?\b",
        re.I,
    )

    PAGE_ALIASES: ClassVar[tuple[str, ...]] = (
        "page",
        "pages",
    )
