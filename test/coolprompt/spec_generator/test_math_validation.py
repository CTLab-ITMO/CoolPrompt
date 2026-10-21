"""Offline unit, contract, and regression tests for MathValidator (no API calls)."""

import json
from fractions import Fraction
from typing import Literal

import pytest

from coolprompt.spec_generator.validation.math_validator import (
    Example,
    MathValidator,
    Solution,
    SolutionBatch,
    UnitError,
    calculate,
    convert_units,
    evaluate_expression,
    format_number,
    normalize_unit,
    parse_answer,
)


def make_solution(
    expression: str | None = "2 + 3",
    *,
    id: int = 0,
    status: Literal["solved", "invalid", "uncertain"] = "solved",
    computed_unit: str = "dimensionless",
    requested_unit: str = "dimensionless",
    must_be_integer: bool = False,
    must_be_nonnegative: bool = True,
    rounding_reason: str = "",
) -> Solution:
    return Solution(
        id=id,
        status=status,
        expression=expression,
        computed_unit=computed_unit,
        requested_unit=requested_unit,
        must_be_integer=must_be_integer,
        must_be_nonnegative=must_be_nonnegative,
        reason="",
        rounding_reason=rounding_reason,
    )


def make_solver(*solutions: Solution):
    def solve(_request: str) -> SolutionBatch:
        return SolutionBatch(solutions=list(solutions))

    return solve


def validate_one(problem: str, answer: str, solution: Solution, **kwargs):
    return MathValidator(solve=make_solver(solution), **kwargs).validate(
        [Example(input=problem, output=answer)]
    )[0]


def validate_with_expression(
    example: Example,
    expression: str,
    *,
    computed_unit: str = "dimensionless",
    requested_unit: str = "dimensionless",
):
    return MathValidator(
        solve=make_solver(
            make_solution(
                expression,
                computed_unit=computed_unit,
                requested_unit=requested_unit,
            )
        )
    ).validate([example])[0]


# ============================ Arithmetic ============================


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("18 + 7 - 5", 20),
        ("6 * 8 - 11", 37),
        ("120 - 35 - 28", 57),
        ("45 - 3 / 5 * 45", 18),
        ("80 - 25 / 100 * 80", 60),
        ("15 * 8 * 3", 360),
        ("240 / 60", 4),
        ("2 * (14 + 9)", 46),
        ("12 * 7", 84),
        ("3 / 12 * 30", Fraction(15, 2)),
        ("200 - 30 / 100 * 200 - 25 / 100 * 140", 105),
        ("180 + 2 / 3 * 180", 300),
        ("4 * 12 / 6", 8),
        ("90 - 15 * 4", 30),
        ("3 * 42 - 110", 16),
        ("1500 + 200 * 5", 2500),
        ("5 * 3.20 + 2 * 1.50", 19),
        ("(11 * 60 + 45) - (9 * 60 + 15)", 150),
        ("2.5 - 0.75 - 0.6", Fraction(115, 100)),
        ("60 + 20 / 100 * 60", 72),
    ],
)
def test_arithmetic(expression, expected):
    assert calculate(expression) == Fraction(expected)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("999999999999 + 1", Fraction(1000000000000)),
        ("-15 + 8", Fraction(-7)),
        ("0.1 + 0.2", Fraction(3, 10)),
        ("1 / 7 * 7", Fraction(1)),
    ],
)
def test_additional_arithmetic_precision(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected", "rounded"),
    [
        ("10 / 3", Fraction(10, 3), False),
        ("10 // 3", Fraction(3), True),
        ("floor(10 / 3)", Fraction(3), True),
        ("ceil(10 / 3)", Fraction(4), True),
        ("floor(6 / 3)", Fraction(2), False),
        ("ceil(6 / 3)", Fraction(2), False),
    ],
)
def test_rounding(expression, expected, rounded):
    result = evaluate_expression(expression)
    assert result.value == expected
    assert result.rounded == rounded


# ============================ AST safety ============================


@pytest.mark.parametrize(
    "expression",
    [
        "2 ** 10",
        "sqrt(16)",
        "sum([1, 2])",
        "x + 2",
        "1 / 0",
        "__import__('os')",
        "[1, 2, 3]",
        "2 << 3",
    ],
)
def test_invalid_expressions(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__",
        "(1).__class__.__mro__",
        "[x for x in range(10)]",
        "().__getattribute__('__class__')",
    ],
)
def test_additional_unsafe_ast_expressions(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        calculate(expression)


# ============================ Number formatting / parsing ============================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Fraction(10), "10"),
        (Fraction(1, 2), "0.5"),
        (Fraction(1, 8), "0.125"),
        (Fraction(-1, 4), "-0.25"),
        (Fraction(1, 3), "1/3"),
        (Fraction(0), "0"),
        (Fraction(115, 100), "1.15"),
    ],
)
def test_format_number(value, expected):
    assert format_number(value) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", 42),
        ("40,000", 40000),
        ("#### 273", 273),
        ("0.25", Fraction(1, 4)),
        ("-15", -15),
        ("7.5", Fraction(15, 2)),
        ("1/3", Fraction(1, 3)),
    ],
)
def test_parse_answer(text, expected):
    assert parse_answer(text) == Fraction(expected)


@pytest.mark.parametrize("text", ["hello", "42 apples", "#### 10 #### 20", "1/0", ""])
def test_invalid_answers(text):
    with pytest.raises((ValueError, ZeroDivisionError)):
        parse_answer(text)


@pytest.mark.parametrize("answer", ["NaN", "Infinity", "1e100000000"])
def test_additional_invalid_answers(answer):
    with pytest.raises((ValueError, ZeroDivisionError, OverflowError)):
        parse_answer(answer)


# ============================ Unit normalization / conversion ============================


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("US dollars", "usd"),
        ("U.S. dollars", "usd"),
        ("Canadian dollars", "cad"),
        ("Australian dollars", "aud"),
        ("British pounds", "gbp"),
        ("Japanese yen", "jpy"),
        ("Russian rubles", "rub"),
        ("Square Meters", "meter**2"),
        ("cubic meters", "meter**3"),
        ("percentage points", "percentage_point"),
        ("unitless", "dimensionless"),
        ("MHz", "MHz"),
    ],
)
def test_normalize_unit(unit, expected):
    assert normalize_unit(unit) == expected


def test_explicit_alias_priority():
    assert normalize_unit("US dollars", {"us dollars": "cad"}) == "cad"


def test_case_insensitive_alias():
    assert normalize_unit("DOLLARS", {"dollars": "usd"}) == "usd"


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (1, "hour", "minute", 60),
        (3, "yard", "foot", 9),
        (1, "mile", "foot", 5280),
        (1, "kilogram", "gram", 1000),
        (2, "year", "month", 24),
        (1, "square_meter", "meter**2", 1),
        (25, "percent", "dimensionless", Fraction(1, 4)),
        (5, "count:apple", "count:apple", 5),
        (100, "usd", "usd", 100),
        (100, "US dollars", "usd", 100),
        (50, "Canadian dollars", "cad", 50),
    ],
)
def test_unit_conversions(value, source, target, expected):
    assert convert_units(Fraction(value), source, target) == Fraction(expected)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("meter", "second"),
        ("month", "day"),
        ("usd", "eur"),
        ("count:apple", "count:book"),
    ],
)
def test_incompatible_units(source, target):
    with pytest.raises(UnitError):
        convert_units(Fraction(1), source, target)


def test_ambiguous_cup_without_context():
    with pytest.raises(UnitError) as exc:
        convert_units(Fraction(1), "cup", "cup")
    assert exc.value.code == "ambiguous_unit"


# ============================ Validator contracts ============================


def test_accepts_correct_answer():
    result = validate_one("What is 2 + 3?", "5", make_solution())
    assert result.status == "accepted"
    assert result.expected_answer == "5"
    assert result.corrected is None


def test_corrects_wrong_answer():
    result = validate_one("What is 2 + 3?", "6", make_solution())
    assert result.status == "corrected"
    assert result.expected_answer == "5"
    assert result.corrected is not None
    assert result.corrected.output == "5"


def test_cross_check_disabled_by_default():
    result = validate_one("What is 2 + 3?", "6", make_solution())
    assert result.status == "corrected"


def test_cross_check_required():
    result = validate_one(
        "What is 2 + 3?", "6", make_solution(), require_cross_check=True
    )
    assert result.status == "uncertain"
    assert result.reason_code == "cross_check_required"
    assert result.corrected is None


def test_cross_check_agreement():
    solution = make_solution()
    validator = MathValidator(
        solve=make_solver(solution), cross_check=make_solver(solution)
    )
    result = validator.validate([Example(input="What is 2 + 3?", output="6")])[0]
    assert result.status == "corrected"
    assert result.corrected is not None
    assert result.corrected.output == "5"


def test_cross_check_disagreement():
    validator = MathValidator(
        solve=make_solver(make_solution("2 + 3")),
        cross_check=make_solver(make_solution("2 + 4")),
    )
    result = validator.validate([Example(input="What is 2 + 3?", output="7")])[0]
    assert result.status == "uncertain"
    assert result.reason_code == "verifier_disagreement"


def test_invalid_problem():
    result = validate_one(
        "An impossible math problem.", "10", make_solution(status="invalid")
    )
    assert result.status == "rejected"
    assert result.reason_code == "invalid_problem"


def test_missing_expression():
    result = validate_one("What is 2 + 3?", "5", make_solution(None))
    assert result.status == "uncertain"
    assert result.reason_code == "missing_expression"
    assert result.retryable is True


def test_unjustified_rounding():
    result = validate_one("Round 10 / 3 down.", "3", make_solution("floor(10 / 3)"))
    assert result.status == "uncertain"
    assert result.reason_code == "unjustified_rounding"


def test_justified_rounding():
    result = validate_one(
        "Round 10 / 3 down.",
        "4",
        make_solution(
            "floor(10 / 3)",
            rounding_reason="The problem requires rounding down.",
        ),
    )
    assert result.status == "corrected"
    assert result.expected_answer == "3"


def test_noninteger_count():
    result = validate_one(
        "How many whole apples?",
        "2",
        make_solution(
            "5 / 2",
            computed_unit="count:apple",
            requested_unit="count:apple",
            must_be_integer=True,
        ),
    )
    assert result.status == "rejected"
    assert result.reason_code == "noninteger_count"


def test_negative_quantity():
    result = validate_one(
        "How many apples remain?",
        "0",
        make_solution(
            "3 - 5",
            computed_unit="count:apple",
            requested_unit="count:apple",
            must_be_nonnegative=True,
        ),
    )
    assert result.status == "rejected"
    assert result.reason_code == "negative_quantity"


# ============================ Contextual units ============================


def test_dollar_resolved_from_dollar_sign():
    result = validate_one(
        "Olivia earns $15 per hour for 8 hours.",
        "100",
        make_solution("15 * 8", computed_unit="dollar", requested_unit="dollar"),
    )
    assert result.status == "corrected"
    assert result.expected_answer == "120"


def test_dollar_resolved_from_us_dollars():
    result = validate_one(
        "A jacket costs 80 US dollars. It has a 25% discount.",
        "65",
        make_solution(
            "80 * (1 - 25 / 100)",
            computed_unit="dollar",
            requested_unit="dollar",
        ),
    )
    assert result.status == "corrected"
    assert result.expected_answer == "60"


def test_dollar_without_context_remains_unsupported():
    result = validate_one(
        "Olivia earns 15 dollars per hour for 8 hours.",
        "100",
        make_solution("15 * 8", computed_unit="dollar", requested_unit="dollar"),
    )
    assert result.status == "uncertain"
    assert result.reason_code == "unsupported_unit"


def test_cup_same_unit_arithmetic():
    result = validate_one(
        "A bakery needs 3 cups of flour for 12 muffins. How many cups for 30 muffins?",
        "8",
        make_solution("3 / 12 * 30", computed_unit="cup", requested_unit="cup"),
    )
    assert result.status == "corrected"
    assert result.expected_answer == "7.5"


def test_ounce_same_unit_arithmetic():
    result = validate_one(
        "A package contains 10 ounces of food. Another package contains 5 ounces. How many ounces in total?",
        "14",
        make_solution("10 + 5", computed_unit="ounce", requested_unit="ounce"),
    )
    assert result.status == "corrected"
    assert result.expected_answer == "15"


def test_opaque_cup_cannot_convert_to_liter():
    result = validate_one(
        "A recipe uses 3 cups of water. How many liters is that?",
        "1",
        make_solution("3", computed_unit="cup", requested_unit="liter"),
    )
    assert result.status == "uncertain"
    assert result.reason_code == "incompatible_units"


def test_page_as_count():
    result = validate_one(
        "A book has 120 pages. Lucas reads 35 pages and then 28 pages. How many pages remain?",
        "58",
        make_solution("120 - 35 - 28", computed_unit="page", requested_unit="page"),
    )
    assert result.status == "corrected"
    assert result.expected_answer == "57"


def test_hyphenated_count_entity():
    result = validate_one(
        "There are 10 red apples and 5 more red apples.",
        "14",
        make_solution(
            "10 + 5",
            computed_unit="count:red-apple",
            requested_unit="count:red-apple",
        ),
    )
    assert result.status == "corrected"
    assert result.expected_answer == "15"


def test_explicit_alias_overrides_context():
    result = validate_one(
        "A toy costs $10. Two toys cost how much?",
        "15",
        make_solution("10 * 2", computed_unit="dollar", requested_unit="dollar"),
        unit_aliases={"dollar": "cad"},
    )
    assert result.status == "corrected"
    assert result.expected_answer == "20"


# ============================ Batching / solver errors ============================


def test_multiple_examples():
    def solver(_request: str) -> SolutionBatch:
        return SolutionBatch(
            solutions=[
                make_solution("2 + 3", id=0),
                make_solution("10 * 2", id=1),
            ]
        )

    validator = MathValidator(solve=solver, batch_size=2)
    results = validator.validate(
        [
            Example(input="What is 2 + 3?", output="5"),
            Example(input="What is 10 * 2?", output="15"),
        ]
    )
    assert len(results) == 2
    assert results[0].status == "accepted"
    assert results[1].status == "corrected"
    assert results[1].expected_answer == "20"


def test_invalid_solution_ids():
    validator = MathValidator(solve=make_solver(make_solution(id=5)))
    result = validator.validate([Example(input="What is 2 + 3?", output="5")])[0]
    assert result.status == "uncertain"
    assert result.reason_code == "verifier_batch_failure"
    assert result.retryable is True


def test_solver_failure():
    def failing_solver(_request: str) -> SolutionBatch:
        raise RuntimeError("LLM unavailable")

    validator = MathValidator(solve=failing_solver)
    result = validator.validate([Example(input="What is 2 + 3?", output="5")])[0]
    assert result.status == "uncertain"
    assert result.reason_code == "verifier_batch_failure"
    assert result.retryable is True


def test_empty_solver_response_is_not_accepted():
    validator = MathValidator(solve=make_solver())
    result = validator.validate([Example(input="What is 2 + 2?", output="4")])[0]
    assert result.status == "uncertain"
    assert result.corrected is None


def test_duplicate_solver_ids_are_not_accepted():
    validator = MathValidator(
        solve=make_solver(make_solution("2 + 2", id=0), make_solution("2 + 2", id=0))
    )
    result = validator.validate([Example(input="What is 2 + 2?", output="4")])[0]
    assert result.status != "accepted"


def test_missing_solver_id_is_not_accepted():
    validator = MathValidator(solve=make_solver(make_solution("2 + 2", id=99)))
    result = validator.validate([Example(input="What is 2 + 2?", output="4")])[0]
    assert result.status == "uncertain"


# ============================ Cross-check failure cases ============================


def test_cross_check_failure_prevents_correction():
    def cross_check(_request: str) -> SolutionBatch:
        raise RuntimeError("Cross-check model unavailable")

    validator = MathValidator(
        solve=make_solver(make_solution("2 + 2")),
        cross_check=cross_check,
        require_cross_check=True,
    )
    result = validator.validate([Example(input="What is 2 + 2?", output="5")])[0]
    assert result.status == "uncertain"
    assert result.corrected is None


def test_cross_check_disagreement_does_not_change_original():
    example = Example(input="What is 10 + 5?", output="16")
    validator = MathValidator(
        solve=make_solver(make_solution("10 + 5")),
        cross_check=make_solver(make_solution("10 + 6")),
        require_cross_check=True,
    )
    result = validator.validate([example])[0]
    assert result.status == "uncertain"
    assert result.corrected is None
    assert example.output == "16"


# ============================ Incorrect LLM interpretations ============================


def test_wrong_percentage_interpretation_is_not_silently_accepted():
    example = Example(
        input="A ticket costs $32 with a 12.5% service fee. What is the final price?",
        output="36",
    )
    result = validate_with_expression(
        example,
        "32 + 12 / 100 * 32",
        computed_unit="usd",
        requested_unit="usd",
    )
    assert result.status != "accepted"


def test_omitted_operations_are_not_silently_accepted():
    example = Example(
        input=(
            "A merchant has 45 kilograms of tea. "
            "It sells 8 kilograms and then 11 kilograms. "
            "How many kilograms remain?"
        ),
        output="26",
    )
    result = validate_with_expression(
        example,
        "45 - 45 / 2",
        computed_unit="kilogram",
        requested_unit="kilogram",
    )
    assert result.status != "accepted"


# ============================ Extra unit / quantity failures ============================


def test_incompatible_units_cannot_be_accepted():
    result = validate_with_expression(
        Example(input="A rope is 10 meters long. How long is it?", output="10"),
        "10",
        computed_unit="meter",
        requested_unit="second",
    )
    assert result.status != "accepted"


def test_negative_physical_quantity_is_rejected():
    result = validate_with_expression(
        Example(input="A box contains 3 apples. Five apples are removed.", output="-2"),
        "3 - 5",
        computed_unit="count:apple",
        requested_unit="count:apple",
    )
    assert result.status != "accepted"


# ============================ Configuration validation ============================


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        MathValidator(solve=make_solver(), batch_size=0)


def test_invalid_audit_fraction():
    with pytest.raises(ValueError):
        MathValidator(solve=make_solver(), audit_fraction=1.5)


def test_audit_requires_cross_check():
    with pytest.raises(ValueError):
        MathValidator(solve=make_solver(), audit_fraction=0.5)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_cross_check_batch_size(batch_size):
    with pytest.raises(ValueError):
        MathValidator(
            solve=make_solver(),
            cross_check_batch_size=batch_size,
        )


# ============================ Prompt isolation / answer leakage ============================


def test_solver_does_not_receive_target():
    requests = []

    def solver(request: str) -> SolutionBatch:
        requests.append(request)
        return SolutionBatch(solutions=[make_solution("2 + 3")])

    validator = MathValidator(solve=solver)
    validator.validate([Example(input="What is 2 + 3?", output="987654321")])

    assert len(requests) == 1
    assert "987654321" not in requests[0]


def test_cross_check_does_not_receive_target_or_primary_solution():
    primary_requests = []
    cross_check_requests = []
    problem = "What is 2 plus 3?"
    secret_target = "987654321"

    def primary_solver(request: str) -> SolutionBatch:
        primary_requests.append(request)
        return SolutionBatch(solutions=[make_solution("2 + 3")])

    def cross_check_solver(request: str) -> SolutionBatch:
        cross_check_requests.append(request)
        return SolutionBatch(solutions=[make_solution("1 + 4")])

    validator = MathValidator(
        solve=primary_solver,
        cross_check=cross_check_solver,
        require_cross_check=True,
        audit_fraction=1.0,
    )
    results = validator.validate([Example(input=problem, output=secret_target)])

    assert len(results) == 1
    assert len(primary_requests) == 1
    assert len(cross_check_requests) == 1

    primary_request = primary_requests[0]
    cross_check_request = cross_check_requests[0]
    assert problem in primary_request
    assert problem in cross_check_request
    assert secret_target not in primary_request
    assert secret_target not in cross_check_request
    assert "2 + 3" not in cross_check_request
    assert primary_request == cross_check_request


# ============================ Cross-check reconciliation / batching ============================


def test_cross_check_disagreement_returns_uncertain():
    def primary_solver(_request: str) -> SolutionBatch:
        return SolutionBatch(solutions=[make_solution("100 - 20 - 15")])

    def cross_check_solver(_request: str) -> SolutionBatch:
        return SolutionBatch(solutions=[make_solution("100 * 0.8 * 0.85")])

    original = Example(
        input=(
            "A person has 100 dollars, spends 20% "
            "and then spends another 15 dollars. "
            "How many dollars remain?"
        ),
        output="65",
    )
    validator = MathValidator(
        solve=primary_solver,
        cross_check=cross_check_solver,
        require_cross_check=True,
        audit_fraction=1.0,
    )
    results = validator.validate([original])

    assert len(results) == 1
    result = results[0]
    assert result.status == "uncertain"
    assert result.corrected is None
    assert result.original.output == "65"
    assert result.reason_code == "verifier_disagreement"


def test_cross_check_accumulates_across_batches():
    primary_calls = 0
    cross_check_calls = 0
    cross_check_sizes = []

    def primary_solver(request: str) -> SolutionBatch:
        nonlocal primary_calls
        primary_calls += 1
        problems = json.loads(request.split("Problems:\n", 1)[1])
        return SolutionBatch(
            solutions=[make_solution("2 + 3", id=item["id"]) for item in problems]
        )

    def secondary_solver(request: str) -> SolutionBatch:
        nonlocal cross_check_calls
        cross_check_calls += 1
        problems = json.loads(request.split("Problems:\n", 1)[1])
        cross_check_sizes.append(len(problems))
        return SolutionBatch(
            solutions=[make_solution("1 + 4", id=item["id"]) for item in problems]
        )

    examples = [
        Example(input=f"What is 2 plus 3? Task {i}", output="999") for i in range(6)
    ]
    validator = MathValidator(
        solve=primary_solver,
        cross_check=secondary_solver,
        require_cross_check=True,
        audit_fraction=0.0,
        batch_size=2,
        cross_check_batch_size=5,
    )
    results = validator.validate(examples)

    assert primary_calls == 3
    assert cross_check_calls == 2
    assert cross_check_sizes == [5, 1]
    assert len(results) == 6
    assert all(result.status == "corrected" for result in results)
    assert all(result.corrected is not None for result in results)
    assert all(result.corrected.output == "5" for result in results)


# ============================ Temperature conversions ============================


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (0, "degC", "kelvin", Fraction(27315, 100)),
        (100, "degC", "degF", 212),
        (32, "degF", "degC", 0),
        (27315 / 100, "kelvin", "degC", 0),
        (0, "degC", "degR", Fraction(49167, 100)),
        (10, "delta_degC", "delta_degF", 18),
        (18, "delta_degF", "delta_degC", 10),
        (10, "delta_kelvin", "delta_degC", 10),
        (18, "delta_degR", "delta_kelvin", 10),
    ],
)
def test_temperature_conversions(value, source, target, expected):
    assert convert_units(
        Fraction(str(value)),
        source,
        target,
    ) == Fraction(expected)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("degC", "delta_degC"),
        ("delta_degF", "degF"),
        ("kelvin", "delta_kelvin"),
        ("delta_kelvin", "kelvin"),
    ],
)
def test_absolute_and_delta_temperatures_are_incompatible(source, target):
    with pytest.raises(UnitError) as exc:
        convert_units(Fraction(10), source, target)

    assert exc.value.code == "incompatible_temperature_kind"


@pytest.mark.parametrize(
    "unit",
    [
        "degC / second",
        "meter * degF",
        "delta_degC / hour",
        "kelvin**2",
    ],
)
def test_compound_temperature_units_are_rejected(unit):
    with pytest.raises(UnitError):
        convert_units(Fraction(10), unit, unit)


# ============================ Exact arithmetic ============================


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("0.1 + 0.2", Fraction(3, 10)),
        ("0.0000001 * 10000000", 1),
        ("1 / 3 + 1 / 6", Fraction(1, 2)),
        ("999999999999999999 + 1", 1000000000000000000),
        ("-10 // 3", -4),
        ("ceil(-10 / 3)", -3),
        ("floor(-10 / 3)", -4),
    ],
)
def test_exact_arithmetic_regressions(expression, expected):
    assert calculate(expression) == Fraction(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "10 ** 100000000",
        "2j",
        "True + 1",
        "1e309",
        "round(2.5)",
    ],
)
def test_unsupported_numeric_expressions(expression):
    with pytest.raises((ValueError, OverflowError)):
        calculate(expression)


# ============================ Answers with units ============================


def test_correct_answer_with_explicit_unit():
    result = validate_one(
        "How many meters are in 2 kilometers?",
        "2000 meters",
        make_solution(
            "2",
            computed_unit="kilometer",
            requested_unit="meter",
        ),
    )

    assert result.status == "accepted"
    assert result.expected_answer == "2000"


def test_correct_answer_with_convertible_unit():
    result = validate_one(
        "How many meters are in 2 kilometers?",
        "2 km",
        make_solution(
            "2",
            computed_unit="kilometer",
            requested_unit="meter",
        ),
    )

    assert result.status == "accepted"


def test_wrong_answer_with_explicit_unit():
    result = validate_one(
        "How many meters are in 2 kilometers?",
        "1500 meters",
        make_solution(
            "2",
            computed_unit="kilometer",
            requested_unit="meter",
        ),
    )

    assert result.status == "corrected"
    assert result.expected_answer == "2000"


def test_answer_with_incompatible_unit_is_not_accepted():
    result = validate_one(
        "How many meters are in 2 kilometers?",
        "2000 seconds",
        make_solution(
            "2",
            computed_unit="kilometer",
            requested_unit="meter",
        ),
    )

    assert result.status == "uncertain"
    assert result.reason_code == "incompatible_units"


# ============================ Currency isolation ============================


def test_different_currencies_are_not_interchangeable():
    with pytest.raises(UnitError):
        convert_units(Fraction(100), "usd", "cad")


def test_currency_context_is_not_shared_between_examples():
    def solver(request: str) -> SolutionBatch:
        problems = json.loads(request.split("Problems:\n", 1)[1])

        return SolutionBatch(
            solutions=[
                make_solution(
                    "10 * 2",
                    id=item["id"],
                    computed_unit="dollar",
                    requested_unit="dollar",
                )
                for item in problems
            ]
        )

    examples = [
        Example(input="Two items cost 10 US dollars each.", output="20"),
        Example(input="Two items cost 10 Canadian dollars each.", output="20"),
    ]

    results = MathValidator(
        solve=solver,
        batch_size=2,
    ).validate(examples)

    assert len(results) == 2
    assert all(result.status == "accepted" for result in results)


# ============================ Validation limits ============================


def test_oversized_expression_is_not_accepted():
    expression = "1+" * 250 + "1"

    result = validate_one(
        "Calculate the sum.",
        "251",
        make_solution(expression),
    )

    assert result.status == "uncertain"
    assert result.reason_code == "invalid_expression"


def test_division_by_zero_is_not_accepted():
    result = validate_one(
        "Calculate 10 divided by zero.",
        "0",
        make_solution("10 / 0"),
    )

    assert result.status == "uncertain"
    assert result.reason_code == "invalid_expression"


def test_invalid_answer_does_not_produce_correction():
    result = validate_one(
        "What is 2 + 3?",
        "not a number",
        make_solution("2 + 3"),
    )

    assert result.status == "uncertain"
    assert result.corrected is None
    assert result.reason_code == "unsupported_target_format"
