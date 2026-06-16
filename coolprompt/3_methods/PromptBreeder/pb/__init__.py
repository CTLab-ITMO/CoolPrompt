import logging
import concurrent.futures
import time
from typing import List, Optional, Dict

from rich import print

from pb.mutation_operators import mutate, set_dataset_examples
from pb import datasets
from pb.types import EvolutionUnit, Population

logger = logging.getLogger(__name__)


# Module-level dataset configuration. ``configure_dataset`` must be called
# before ``init_run`` / ``run_for_n``.
_DATASET_NAME: Optional[str] = None
_EVAL_EXAMPLES: List[Dict] = []
_METRIC_NAME: Optional[str] = None


def configure_dataset(name: str, examples: List[Dict], metric: Optional[str] = None) -> None:
    """Register the active dataset (and optional metric) for fitness evaluation."""
    global _DATASET_NAME, _EVAL_EXAMPLES, _METRIC_NAME
    from pb.metrics import default_metric_for, SUPPORTED_METRICS

    _DATASET_NAME = name
    _EVAL_EXAMPLES = list(examples)
    chosen = metric or default_metric_for(name)
    if chosen not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown metric {chosen!r}. Supported: {SUPPORTED_METRICS}"
        )
    _METRIC_NAME = chosen
    # Mirror the same examples into the mutation operators module so the
    # lamarckian ``working_out_task_prompt`` operator stays in sync.
    set_dataset_examples(name, examples)


def get_active_metric() -> Optional[str]:
    """Return the metric currently configured for fitness evaluation."""
    return _METRIC_NAME


def get_active_dataset() -> Optional[str]:
    """Return the dataset currently configured for fitness evaluation."""
    return _DATASET_NAME


def create_population(tp_set: List, mutator_set: List, problem_description: str) -> Population:
    """Sample mutation_prompts x thinking_styles to build the initial population."""
    data = {
        'size': len(tp_set) * len(mutator_set),
        'age': 0,
        'problem_description': problem_description,
        'elites': [],
        'units': [
            EvolutionUnit(T=t, M=m, P='', fitness=0, history=[])
            for t in tp_set for m in mutator_set
        ],
    }
    return Population(**data)


def init_run(population: Population, model, num_evals: int):
    """First pass: synthesise initial task-prompts from (T, M, problem)."""
    if _DATASET_NAME is None:
        raise RuntimeError("Call pb.configure_dataset(...) before init_run().")

    start_time = time.time()

    prompts = [
        f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        for unit in population.units
    ]

    results = model.batch_generate(prompts)

    logger.info(f"Prompt initialization done. {time.time() - start_time:.2f}s")

    assert len(results) == population.size, "size of model response to population is mismatched"
    for i, item in enumerate(results):
        population.units[i].P = item[0].text

    _evaluate_fitness(population, model, num_evals)
    return population


def run_for_n(n: int, population: Population, model, num_evals: int, on_generation=None):
    """Run the genetic algorithm for n generations."""
    p = population
    for i in range(n):
        print(f"================== Population {i} ================== ")
        mutate(p, model)
        print("done mutation")
        _evaluate_fitness(p, model, num_evals)
        print("done evaluation")
        if on_generation is not None:
            try:
                on_generation(i + 1, p)
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("on_generation callback failed: %s", exc)
    return p


def _evaluate_fitness(population: Population, model, num_evals: int) -> Population:
    """Score each prompt P against the active dataset's first ``num_evals`` examples."""
    if _DATASET_NAME is None or not _EVAL_EXAMPLES:
        raise RuntimeError("Call pb.configure_dataset(...) before evaluating fitness.")

    logger.info("Starting fitness evaluation...")
    start_time = time.time()

    batch = _EVAL_EXAMPLES[:num_evals]
    dataset_name = _DATASET_NAME

    elite_fitness = -1
    current_elite = None
    examples = []
    for unit in population.units:
        unit.fitness = 0
        examples.append([datasets.build_eval_prompt(unit.P, ex) for ex in batch])

    results = [None] * len(examples)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(examples))) as executor:
        future_to_unit = {
            executor.submit(model.batch_generate, example_batch, temperature=1.0): unit_index
            for unit_index, example_batch in enumerate(examples)
        }
        for future in concurrent.futures.as_completed(future_to_unit):
            unit_index = future_to_unit[future]
            try:
                results[unit_index] = future.result()
            except Exception as exc:
                print(f"Exception: {exc}")
                results[unit_index] = []

    metric_name = _METRIC_NAME or "exact_match"
    for unit_index, fitness_results in enumerate(results):
        if not fitness_results:
            continue
        unit = population.units[unit_index]
        for i, x in enumerate(fitness_results):
            value = datasets.score(dataset_name, metric_name, x[0].text, batch[i])
            unit.fitness += value / num_evals

            if unit.fitness > elite_fitness:
                current_elite = unit.model_copy()
                elite_fitness = unit.fitness

    if current_elite is not None:
        population.elites.append(current_elite)
    logger.info(f"Done fitness evaluation. {time.time() - start_time:.2f}s")
    return population
