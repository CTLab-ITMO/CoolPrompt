"""The original RIDER evolutionary loop (FRUCT paper, Fig. 1 and Sec. III)."""

from __future__ import annotations

import random
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import TelemetryCallback
from coolprompt.utils.logging_config import logger

from .bandit import WindowedThompsonSelector
from .config import (
    OPEN_ENDED_DISABLED_OPERATORS,
    OPEN_ENDED_METRICS,
    OPERATOR_CLASSES,
    OPERATORS,
    RIDERConfig,
    operator_budget_shares,
)
from .diversity import PromptEmbedder, greedy_kdpp, mu_plus_lambda_select
from .memory import LongTermMemory
from .operators import (
    DATA_AWARE_TEMPLATE,
    OPERATOR_BUILDERS,
    ZERO_ORDER_APPROACHES,
    ZERO_ORDER_SEED_TEMPLATE,
    OperatorContext,
    OperatorRequest,
    target_text,
    zero_order,
)
from .population import Candidate, ErrorCase

_PROMPT_TAG = re.compile(r"<prompt>(.*?)(?:</prompt>|$)", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_META_COMMENTARY = re.compile(
    r"^\s*(here is|here's|sure|certainly|below is|of course|i have|i've|"
    r"the (new|improved|refined|updated) (prompt|instruction)|"
    r"(new|improved|refined|updated) (prompt|instruction)\s*:)",
    re.IGNORECASE,
)
_LABEL_PREFIX = re.compile(
    r"^\s*[*_#\s]*(new |improved |refined )?(instruction|prompt)[*_\s]*:[*_\s]*",
    re.IGNORECASE,
)
_TRAILING_CHATTER = re.compile(
    r"\n\s*(let me know|i hope|feel free|this (prompt|instruction) (will|should))[^\n]*$",
    re.IGNORECASE,
)


@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    temperature: float
    epsilon: float
    stagnation: int
    operators: dict[str, int]
    accepted: dict[str, int]
    restarted: bool = False


@dataclass
class RIDERResult:
    """Outcome of one optimization run."""

    best_prompt: str
    best_fitness: float
    ensemble: list[Candidate]
    population: list[Candidate]
    history: list[GenerationRecord] = field(default_factory=list)
    operator_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    system_calls: int = 0
    evaluations: int = 0


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    norm = sum(weights.values())
    exact = {key: total * w / norm for key, w in weights.items()}
    counts = {key: int(value) for key, value in exact.items()}
    leftover = total - sum(counts.values())
    for key in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)[:leftover]:
        counts[key] += 1
    return counts


def quota_allocation(operators: list[str], total: int) -> list[str]:
    """Split the first generation by the paper's class budget shares.

    The budget goes to classes first (42/31/20/7), then evenly to the enabled
    operators of each class; shares of disabled classes are renormalized.
    """

    enabled = set(operators)
    classes = {
        name: (share, [op for op in members if op in enabled])
        for name, (share, members) in OPERATOR_CLASSES.items()
    }
    classes = {name: value for name, value in classes.items() if value[1]}
    per_class = _largest_remainder({n: share for n, (share, _) in classes.items()}, total)
    chosen: list[str] = []
    for name, (_, members) in classes.items():
        counts = _largest_remainder({op: 1.0 for op in members}, per_class[name])
        chosen += [op for op in members for _ in range(counts[op])]
    return chosen


class RIDEROriginalOptimizer:
    """Evolves a population of prompts with nine LLM operators.

    Fitness is the evaluator's metric on the validation split. All offspring
    of a generation are evaluated before retention; operator choice adapts via
    epsilon-greedy Windowed Thompson Sampling; the best prompt's failure cases
    are fed back to the operators.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        config: RIDERConfig | None = None,
        *,
        template: str | None = None,
        embedder: PromptEmbedder | None = None,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.config = config or RIDERConfig()
        self.config.validate()
        self.template = template
        self.embedder = embedder or PromptEmbedder(self.config.embedding_model)
        self.rng = random.Random(self.config.seed)
        self._cache: dict[str, Candidate] = {}
        self._lock = threading.Lock()
        self._temperature_supported: bool | None = None
        self._last_error: Exception | None = None
        self.system_calls = 0
        self.evaluations = 0

    # ------------------------------------------------------------------ LLM

    def _invoke(self, meta_prompt: str, temperature: float) -> str:
        with self._lock:
            self.system_calls += 1
        response = None
        if self._temperature_supported is not False:
            try:
                response = self.model.bind(temperature=temperature).invoke(meta_prompt)
                self._temperature_supported = True
            except Exception as exc:
                if self._temperature_supported:
                    raise  # temperature worked before: this is a real failure
                logger.warning("RIDER: model rejected temperature (%s); using defaults", exc)
                self._temperature_supported = False
        if response is None:
            response = self.model.invoke(meta_prompt)
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def extract_prompt(response: str) -> tuple[str, bool]:
        """Return the instruction inside ``<prompt>`` tags and whether tags were found."""

        text = _THINK.sub("", response or "")
        matches = [m.strip() for m in _PROMPT_TAG.findall(text) if m.strip()]
        tagged = bool(matches)
        text = matches[-1] if matches else text
        text = text.replace("<prompt>", "").replace("</prompt>", "").strip()
        text = _FENCE.sub("", text).strip()
        text = _LABEL_PREFIX.sub("", text).strip()
        text = _TRAILING_CHATTER.sub("", text).strip().strip('"').strip()
        return text, tagged

    def _is_degenerate(self, text: str) -> bool:
        return (
            len(text.split()) < self.config.min_prompt_words
            or bool(_META_COMMENTARY.match(text))
        )

    def _generate(
        self, meta_prompt: str, temperature: float, requires_tags: bool = False
    ) -> str | None:
        """Call the system model, regenerating degenerate output with higher T."""

        temp = temperature
        for _ in range(self.config.regeneration_attempts + 1):
            try:
                text, tagged = self.extract_prompt(self._invoke(meta_prompt, temp))
            except Exception as exc:
                logger.warning("RIDER: generation failed: %s", exc)
                self._last_error = exc
                text, tagged = "", False
            if text and (tagged or not requires_tags) and not self._is_degenerate(text):
                return text
            temp = min(temp + 0.2, self.config.max_temperature)
        return None

    def _generate_many(
        self, jobs: Sequence[tuple[str, float, bool]]
    ) -> list[str | None]:
        workers = max(1, min(self.config.max_workers, len(jobs)))
        if workers == 1:
            return [self._generate(*job) for job in jobs]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda job: self._generate(*job), jobs))

    # ----------------------------------------------------------- evaluation

    def _evaluate(self, cand: Candidate, val_x: list[str], val_y: list[Any]) -> Candidate:
        cached = self._cache.get(cand.text)
        if cached is not None:
            cand.fitness, cand.errors = cached.fitness, list(cached.errors)
            return cand
        try:
            result = self.evaluator.evaluate(
                prompt=cand.text,
                dataset=val_x,
                targets=val_y,
                template=self.template,
                failed_examples=self.config.num_error_examples,
                return_detailed=True,
            )
            score = result.aggregate_score
        except Exception as exc:
            # One failed evaluation must not end the run; retry if the prompt recurs.
            logger.warning("RIDER: evaluation failed (%s); prompt scored as -inf", exc)
            cand.fitness, cand.errors = float("-inf"), []
            return cand
        self.evaluations += 1
        cand.fitness = float(score) if score is not None else float("-inf")
        cand.errors = [
            ErrorCase(
                input=str(f.instance),
                output=str(f.model_answer_parsed or f.assistant_answer),
                expected=target_text(f.ground_truth),
                score=float(f.metric_value),
            )
            for f in (result.failed_examples or [])
        ]
        self._cache[cand.text] = cand
        return cand

    def _evaluate_all(self, cands: list[Candidate], val_x, val_y) -> list[Candidate]:
        return [self._evaluate(c, val_x, val_y) for c in cands]

    # -------------------------------------------------------- initialization

    def _initial_population(
        self,
        initial_prompt: str,
        task_description: str,
        train: list[tuple[str, Any]],
        val_x: list[str],
        val_y: list[Any],
    ) -> list[Candidate]:
        cfg = self.config
        header = (
            "You are writing an instruction (prompt) for a language model.\n"
            f"Task: {task_description.strip()}\n"
        )
        jobs: list[tuple[str, float, bool]] = []
        kinds: list[str] = []
        lo, hi = cfg.data_aware_temperature
        if train:
            for temp in np.linspace(lo, hi, cfg.data_aware_seeds):
                sample = self.rng.sample(train, min(cfg.data_aware_examples, len(train)))
                examples = "\n\n".join(
                    f"Input: {str(x).strip()[:700]}\nOutput: {target_text(y)[:700]}"
                    for x, y in sample
                )
                jobs.append(
                    (DATA_AWARE_TEMPLATE.format(header=header, examples=examples), float(temp), False)
                )
                kinds.append("data_aware")
        lo, hi = cfg.zero_order_temperature
        for i, temp in enumerate(np.linspace(lo, hi, cfg.zero_order_seeds)):
            approach = ZERO_ORDER_APPROACHES[i % len(ZERO_ORDER_APPROACHES)]
            jobs.append(
                (ZERO_ORDER_SEED_TEMPLATE.format(header=header, approach=approach), float(temp), False)
            )
            kinds.append("zero_order")

        texts = self._generate_many(jobs)
        if jobs and not any(texts):
            raise RuntimeError(
                "RIDER: the system model produced no usable prompt during initialization"
            ) from self._last_error
        population = [
            Candidate(text=text, operator=f"init_{kind}")
            for text, kind in zip(texts, kinds)
            if text
        ]

        fallbacks = [initial_prompt.strip()]
        if task_description.strip() and task_description.strip() != initial_prompt.strip():
            fallbacks.append(
                f"{initial_prompt.strip()}\n\nTask details: {task_description.strip()}"
            )
        else:
            fallbacks.append(
                f"{initial_prompt.strip()}\n\nRead the input carefully, follow the task "
                "requirements exactly and answer in the required output format."
            )
        for text in fallbacks[: max(cfg.fallback_seeds, 1)]:
            population.append(Candidate(text=text, operator="init_fallback"))

        unique: dict[str, Candidate] = {}
        for cand in population:
            unique.setdefault(cand.text, cand)
        population = list(unique.values())[: cfg.population_size]
        population = self._evaluate_all(population, val_x, val_y)
        population = [c for c in population if c.fitness != float("-inf")]
        if len(population) < 2:
            raise RuntimeError("RIDER could not evaluate an initial population")
        return sorted(population, key=lambda c: c.fitness, reverse=True)

    # ------------------------------------------------------------- the loop

    def _enabled_operators(self) -> list[str]:
        disabled = set(self.config.disabled_operators)
        metric_name = type(getattr(self.evaluator, "metric", None)).__name__.lower()
        if any(part in metric_name for part in OPEN_ENDED_METRICS):
            disabled |= OPEN_ENDED_DISABLED_OPERATORS
        return [op for op in OPERATORS if op not in disabled]

    def _selector(self, operators: list[str]) -> WindowedThompsonSelector:
        shares = operator_budget_shares()
        top = max(shares[op] for op in operators)
        priors = {op: 0.2 + 0.6 * shares[op] / top for op in operators}
        cfg = self.config
        return WindowedThompsonSelector(
            operators,
            priors,
            prior_strength=cfg.prior_strength,
            window=cfg.thompson_window,
            decay=cfg.thompson_decay,
            penalty_threshold=cfg.penalty_reward_threshold,
            penalty_min_trials=cfg.penalty_min_trials,
            rng=self.rng,
        )

    def _retain(self, pool: list[Candidate]) -> list[Candidate]:
        return mu_plus_lambda_select(
            [c for c in pool if c.fitness != float("-inf")],
            self.config.population_size,
            self.embedder.similarity,
            threshold=self.config.diversity_threshold,
            tolerance=self.config.fitness_tie_tolerance,
            elite=self.config.elite_size,
        )

    def _context(self, population, best, task_description, memory, train) -> OperatorContext:
        return OperatorContext(
            task_description=task_description,
            population=population,
            errors=best.errors,
            memory=memory.render(),
            tournament_size=self.config.tournament_size,
            rng=self.rng,
            train_examples=tuple((str(x), target_text(y)) for x, y in train[:20]),
            similarity=self.embedder.similarity,
            duplicate_threshold=self.config.diversity_threshold,
        )

    def optimize(
        self,
        initial_prompt: str,
        *,
        train_inputs: list[str],
        train_targets: list[Any],
        val_inputs: list[str],
        val_targets: list[Any],
        problem_description: str | None = None,
        telemetry_callback: TelemetryCallback | None = None,
    ) -> RIDERResult:
        cfg = self.config
        if not val_inputs:
            val_inputs, val_targets = train_inputs, train_targets
        if not val_inputs:
            raise ValueError("RIDER needs a validation (or training) split for fitness")
        train = list(zip(train_inputs, train_targets)) or list(zip(val_inputs, val_targets))
        task_description = (problem_description or "").strip() or (
            f"Improve this instruction for its task:\n{initial_prompt.strip()}"
        )

        population = self._initial_population(
            initial_prompt, task_description, train, val_inputs, val_targets
        )
        operators = self._enabled_operators()
        selector = self._selector(operators)
        memory = LongTermMemory(patterns=cfg.memory_patterns)
        temperature, stagnation = cfg.base_temperature, 0
        best = population[0]
        history: list[GenerationRecord] = []
        logger.info("RIDER: generation 0 best fitness %.4f", best.fitness)
        if telemetry_callback:
            telemetry_callback(0, best.fitness, best.text)

        for generation in range(1, cfg.num_generations + 1):
            if cfg.stop_at_perfect and best.fitness >= cfg.perfect_score:
                logger.info("RIDER: best prompt is already perfect; stopping")
                break
            stagnated = stagnation >= cfg.stagnation_generations
            epsilon = cfg.stagnation_epsilon if stagnated else cfg.epsilon
            ctx = self._context(population, best, task_description, memory, train)
            if generation == 1:
                # The initial budget follows the paper's class shares (42/31/20/7).
                chosen = quota_allocation(operators, cfg.offspring_per_generation)
                self.rng.shuffle(chosen)
            else:
                chosen = [
                    selector.select(generation, epsilon)
                    for _ in range(cfg.offspring_per_generation)
                ]
            requests: list[OperatorRequest] = [OPERATOR_BUILDERS[op](ctx) for op in chosen]
            texts = self._generate_many(
                [(r.meta_prompt, temperature, r.requires_tags) for r in requests]
            )
            if not any(texts):
                raise RuntimeError(
                    f"RIDER: generation {generation} produced no usable offspring"
                ) from self._last_error

            offspring: list[Candidate] = []
            produced: list[tuple[str, Candidate | None]] = []
            for op, req, text in zip(chosen, requests, texts):
                cand = None
                if text:
                    cand = Candidate(
                        text=text,
                        generation=generation,
                        operator=op,
                        parents=tuple(p.id for p in req.parents),
                    )
                    offspring.append(cand)
                produced.append((op, cand))
            # Evaluate-first: every offspring is scored before any selection.
            self._evaluate_all(offspring, val_inputs, val_targets)
            pool = population + offspring
            # Reward: the offspring beats the median of the parent population.
            median = statistics.median(c.fitness for c in population)
            new_population = self._retain(pool)
            survivors = {id(c) for c in new_population}

            accepted: dict[str, int] = {}
            counts: dict[str, int] = {}
            for op, cand in produced:
                counts[op] = counts.get(op, 0) + 1
                reward = 1.0 if cand is not None and cand.fitness > median else 0.0
                selector.update(op, generation, reward)
                if cand is not None and id(cand) in survivors:
                    accepted[op] = accepted.get(op, 0) + 1

            previous_best = best.fitness
            population = new_population
            best = population[0]
            improvement = (best.fitness - previous_best) / max(abs(previous_best), 1e-9)
            restarted = False
            if improvement < cfg.stagnation_improvement:
                stagnation += 1
                if stagnation == cfg.stagnation_generations:
                    temperature = min(
                        temperature * cfg.stagnation_temperature_factor, cfg.max_temperature
                    )
            else:
                # Cool down only after genuine progress from a non-stagnated state.
                if stagnation < cfg.stagnation_generations:
                    temperature = max(
                        cfg.base_temperature, temperature * cfg.recovery_temperature_factor
                    )
                stagnation = 0
            if stagnation >= cfg.restart_generations:
                restart_ctx = self._context(population, best, task_description, memory, train)
                population = self._soft_restart(
                    population, restart_ctx, temperature, val_inputs, val_targets
                )
                pool = pool + population
                best = population[0]
                stagnation = 0
                restarted = True

            if generation % cfg.memory_update_interval == 0:
                memory.update(population, pool)

            history.append(
                GenerationRecord(
                    generation=generation,
                    best_fitness=best.fitness,
                    mean_fitness=float(np.mean([c.fitness for c in population])),
                    temperature=temperature,
                    epsilon=epsilon,
                    stagnation=stagnation,
                    operators=counts,
                    accepted=accepted,
                    restarted=restarted,
                )
            )
            logger.info(
                "RIDER: generation %d best %.4f mean %.4f T=%.2f eps=%.2f accepted=%s",
                generation,
                best.fitness,
                history[-1].mean_fitness,
                temperature,
                epsilon,
                accepted,
            )
            if telemetry_callback:
                telemetry_callback(generation, best.fitness, best.text)

        ensemble = greedy_kdpp(
            population, cfg.ensemble_size, self.embedder.similarity, cfg.ensemble_quality_weight
        )
        return RIDERResult(
            best_prompt=best.text,
            best_fitness=best.fitness,
            ensemble=ensemble,
            population=population,
            history=history,
            operator_statistics=selector.statistics(),
            system_calls=self.system_calls,
            evaluations=self.evaluations,
        )

    def _soft_restart(
        self,
        population: list[Candidate],
        ctx: OperatorContext,
        temperature: float,
        val_x: list[str],
        val_y: list[Any],
    ) -> list[Candidate]:
        """Replace the bottom half with fresh zero-order prompts (Sec. III-F)."""

        cfg = self.config
        keep = population[: len(population) - int(len(population) * cfg.restart_fraction)]
        needed = cfg.population_size - len(keep)
        requests = [zero_order(ctx) for _ in range(needed)]
        texts = self._generate_many([(r.meta_prompt, temperature, False) for r in requests])
        seen = {c.text for c in keep}
        fresh: list[Candidate] = []
        for text in texts:
            if text and text not in seen:
                seen.add(text)
                fresh.append(Candidate(text=text, operator="restart_zero_order"))
        self._evaluate_all(fresh, val_x, val_y)
        fresh = [c for c in fresh if c.fitness != float("-inf")]
        logger.info("RIDER: soft restart replaced %d prompts", len(fresh))
        merged = keep + fresh
        if len(merged) < cfg.population_size:
            merged += population[len(keep) : len(keep) + cfg.population_size - len(merged)]
        return sorted(merged, key=lambda c: c.fitness, reverse=True)
