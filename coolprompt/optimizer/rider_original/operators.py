"""The nine RIDER operators (Sec. III-C) as meta-prompt builders.

Each operator turns the current search state into one meta-prompt for the
system model and records which population members it used. The optimizer
sends the meta-prompt, extracts the text inside ``<prompt>`` tags and
validates it. Every operator receives the best prompt's failure cases
(error-directed evolution) and the long-term memory as context.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .diversity import drop_near_duplicates
from .population import Candidate, ErrorCase

ANSWER_FORMAT = (
    "Write only the new instruction. It must be a complete, self-contained "
    "instruction that a language model can follow without seeing this message. "
    "Keep any output-format requirements of the task. Do not add commentary.\n"
    "Wrap the instruction in <prompt></prompt> tags."
)

ZERO_ORDER_APPROACHES = (
    "Write it as a precise step-by-step procedure.",
    "Write it as a concise expert instruction focused on the decisive criteria.",
    "Write it as a checklist of rules the answer must satisfy.",
    "Write it from the perspective of a domain specialist explaining what to do.",
    "Write it constraints-first: state the exact output format, then the method.",
    "Write it with explicit handling of ambiguous and edge cases.",
)


@dataclass
class OperatorContext:
    """Search state an operator may use."""

    task_description: str
    population: list[Candidate]
    errors: list[ErrorCase]
    memory: str
    tournament_size: int
    rng: random.Random
    # Input/target pairs from the training split, used when there are no errors.
    train_examples: tuple[tuple[str, str], ...] = ()
    # Embedding similarity, used to filter near-duplicates for EDA.
    similarity: Callable[[Sequence[str]], np.ndarray] | None = None
    duplicate_threshold: float = 0.92

    @property
    def ranked(self) -> list[Candidate]:
        return sorted(self.population, key=lambda c: c.fitness, reverse=True)

    @property
    def best(self) -> Candidate:
        return self.ranked[0]

    def tournament(self, exclude: Candidate | None = None) -> Candidate:
        pool = [c for c in self.population if c is not exclude] or self.population
        contestants = self.rng.sample(pool, min(self.tournament_size, len(pool)))
        return max(contestants, key=lambda c: c.fitness)

    def random_member(self, exclude: tuple[Candidate, ...] = ()) -> Candidate:
        pool = [c for c in self.population if c not in exclude] or self.population
        return self.rng.choice(pool)


@dataclass
class OperatorRequest:
    """A meta-prompt to send and the parents it was built from."""

    meta_prompt: str
    parents: tuple[Candidate, ...]
    # The response mixes reasoning with the prompt, so the tags are mandatory.
    requires_tags: bool = False


def target_text(target) -> str:
    """Readable target; lists of references are joined."""

    if isinstance(target, (list, tuple)):
        return " | ".join(str(t).strip() for t in target)
    return str(target).strip()


def _header(ctx: OperatorContext) -> str:
    return f"You are optimizing an instruction (prompt) for this task:\n{ctx.task_description.strip()}\n"


def _feedback(ctx: OperatorContext) -> str:
    parts = []
    if ctx.errors:
        cases = "\n\n".join(e.render() for e in ctx.errors)
        parts.append(
            "The current best instruction still fails on these validation examples:\n"
            f"{cases}\n"
            "The new instruction should fix these failure modes without breaking "
            "what already works. Do not copy example answers into the instruction."
        )
    if ctx.memory:
        parts.append(ctx.memory)
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


def _numbered(cands: list[Candidate], with_scores: bool) -> str:
    lines = []
    for i, c in enumerate(cands, 1):
        score = f" (score {c.fitness:.3f})" if with_scores else ""
        lines.append(f"{i}.{score} {c.text.strip()}")
    return "\n".join(lines)


def eda_mutation(ctx: OperatorContext) -> OperatorRequest:
    """EDA: synthesize a new instruction from the population distribution."""

    filtered = list(ctx.ranked)
    if ctx.similarity is not None:
        filtered = drop_near_duplicates(
            filtered, ctx.similarity, ctx.duplicate_threshold
        )
    ctx.rng.shuffle(filtered)
    meta = (
        _header(ctx)
        + "\nHere is a list of instructions from the current population:\n"
        + _numbered(filtered, with_scores=False)
        + "\n\nStudy what these instructions have in common and how they differ. "
        "Write one new instruction that combines their strongest common ideas "
        "and is better than any of them." + _feedback(ctx) + "\n\n" + ANSWER_FORMAT
    )
    return OperatorRequest(meta, tuple(filtered))


def eda_rank_index(ctx: OperatorContext, top_k: int = 5) -> OperatorRequest:
    """EDA with fitness-ranked top-K: learn from the ordering by score."""

    top = ctx.ranked[:top_k]
    ordered = list(reversed(top))
    meta = (
        _header(ctx)
        + "\nThese instructions are sorted from lower to higher validation score:\n"
        + _numbered(ordered, with_scores=True)
        + "\n\nInfer which properties make an instruction score higher. Write a "
        "new instruction that is expected to score higher than all of them."
        + _feedback(ctx)
        + "\n\n"
        + ANSWER_FORMAT
    )
    return OperatorRequest(meta, tuple(top))


def de_mutation(ctx: OperatorContext) -> OperatorRequest:
    """DE/current-to-best/1: mutate only the parts where two donors differ."""

    best = ctx.best
    current = ctx.random_member(exclude=(best,))
    donor_a = ctx.random_member(exclude=(best, current))
    donor_b = ctx.random_member(exclude=(best, current, donor_a))
    meta = (
        _header(ctx) + "\nApply a differential-evolution step to instructions.\n"
        f"Current instruction:\n{current.text.strip()}\n\n"
        f"Best instruction so far (score {best.fitness:.3f}):\n{best.text.strip()}\n\n"
        f"Donor A:\n{donor_a.text.strip()}\n\nDonor B:\n{donor_b.text.strip()}\n\n"
        "Steps:\n"
        "1. Identify the parts where Donor A and Donor B differ.\n"
        "2. Mutate only those different parts, producing new variants of them.\n"
        "3. Move the current instruction toward the best one: keep the parts of the "
        "current instruction that agree with the best one, and replace the rest "
        "with the best instruction's parts combined with the mutated variants."
        + _feedback(ctx)
        + "\n\n"
        + ANSWER_FORMAT
    )
    return OperatorRequest(meta, (current, best, donor_a, donor_b))


def zero_order(ctx: OperatorContext) -> OperatorRequest:
    """Generate a fresh instruction from the problem description and a template."""

    approach = ctx.rng.choice(ZERO_ORDER_APPROACHES)
    meta = (
        _header(ctx)
        + "\nWrite a new instruction for this task from scratch, independent of "
        f"any existing instruction. {approach}"
        + _feedback(ctx)
        + "\n\n"
        + ANSWER_FORMAT
    )
    return OperatorRequest(meta, ())


def ga_mutation(ctx: OperatorContext) -> OperatorRequest:
    """Genetic mutation of a tournament-selected parent."""

    parent = ctx.tournament()
    meta = (
        _header(ctx)
        + "\nMutate the instruction below: change its wording, structure or "
        "emphasis in a meaningful way while keeping its intent, so the result "
        f"explores a different but promising variant.\n\nInstruction:\n{parent.text.strip()}"
        + _feedback(ctx)
        + "\n\n"
        + ANSWER_FORMAT
    )
    return OperatorRequest(meta, (parent,))


def ga_crossover(ctx: OperatorContext) -> OperatorRequest:
    """Genetic crossover of two tournament-selected parents."""

    first = ctx.tournament()
    second = ctx.tournament(exclude=first)
    meta = (
        _header(ctx)
        + "\nCross over the two parent instructions below: build one child "
        "instruction that takes the strongest parts of each parent and merges them "
        f"into a coherent whole.\n\nParent 1:\n{first.text.strip()}\n\n"
        f"Parent 2:\n{second.text.strip()}" + _feedback(ctx) + "\n\n" + ANSWER_FORMAT
    )
    return OperatorRequest(meta, (first, second))


def first_order_refinement(ctx: OperatorContext) -> OperatorRequest:
    """Refine a parent by adding edge cases, constraints and tiebreaking rules."""

    parent = ctx.tournament()
    meta = (
        _header(ctx)
        + "\nRefine the instruction below. Keep everything that works and add the "
        "missing details: edge cases, explicit constraints, tiebreaking rules for "
        "ambiguous inputs, and a precise output format.\n\n"
        f"Instruction (score {parent.fitness:.3f}):\n{parent.text.strip()}"
        + _feedback(ctx)
        + "\n\n"
        + ANSWER_FORMAT
    )
    return OperatorRequest(meta, (parent,))


def lamarckian(ctx: OperatorContext) -> OperatorRequest:
    """Back-translate an instruction from the inputs and correct outputs of errors."""

    if ctx.errors:
        pairs = [(e.input, e.expected) for e in ctx.errors]
    else:
        pairs = list(ctx.train_examples[:5])
    examples = "\n\n".join(
        f"Input: {str(x).strip()[:600]}\nCorrect output: {target_text(y)[:600]}"
        for x, y in pairs
    )
    meta = (
        _header(ctx)
        + "\nA language model should map each input below to its correct output. "
        "Infer the instruction that would make a model produce exactly these "
        f"outputs, generalizing beyond these cases.\n\n{examples}"
        + (f"\n\n{ctx.memory}" if ctx.memory else "")
        + "\n\n"
        + ANSWER_FORMAT
    )
    return OperatorRequest(meta, (ctx.best,))


def reflection_crossover(ctx: OperatorContext) -> OperatorRequest:
    """Verbal-gradient guided crossover: reflect on why one parent scores higher."""

    first = ctx.tournament()
    second = ctx.tournament(exclude=first)
    better, worse = sorted((first, second), key=lambda c: c.fitness, reverse=True)
    meta = (
        _header(ctx)
        + f"\nInstruction A (score {better.fitness:.3f}):\n{better.text.strip()}\n\n"
        f"Instruction B (score {worse.fitness:.3f}):\n{worse.text.strip()}\n\n"
        "First reflect briefly on why A scores higher than B (or why they score "
        "the same): which concrete differences explain the gap. Then use that "
        "reflection as a direction of improvement and write a new instruction "
        "that combines both and moves further in that direction. Put the "
        "reflection before the tags; only the new instruction goes inside them."
        + _feedback(ctx)
        + "\n\nWrap the new instruction in <prompt></prompt> tags."
    )
    return OperatorRequest(meta, (better, worse), requires_tags=True)


OPERATOR_BUILDERS: dict[str, Callable[[OperatorContext], OperatorRequest]] = {
    "eda_mutation": eda_mutation,
    "eda_rank_index": eda_rank_index,
    "de_mutation": de_mutation,
    "zero_order": zero_order,
    "ga_mutation": ga_mutation,
    "ga_crossover": ga_crossover,
    "first_order_refinement": first_order_refinement,
    "lamarckian": lamarckian,
    "reflection_crossover": reflection_crossover,
}

DATA_AWARE_TEMPLATE = (
    "{header}\nHere are input-output examples of the task:\n\n{examples}\n\n"
    "Reverse-engineer the instruction that was given to produce these outputs from "
    "these inputs. Describe the task, the method and the exact output format.\n\n"
    + ANSWER_FORMAT
)

ZERO_ORDER_SEED_TEMPLATE = (
    "{header}\nWrite an instruction for this task. {approach}\n\n" + ANSWER_FORMAT
)
