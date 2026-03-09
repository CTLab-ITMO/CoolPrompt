"""HyPEROptimizer: HyPE with iterative refinement via feedback."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from coolprompt.optimizer.hype.hyper import HyPEOptimizer, Optimizer


# --- Structures ---


@dataclass
class FailedExample:
    """Один неудачный пример для формирования рекомендаций.

    Отдаётся Evaluator при детальной оценке.
    """

    instance: str  # инстанс из датасета
    assistant_answer: str
    metric_value: float  # значение метрики для этого примера
    ground_truth: str | int  # целевой ответ


@dataclass
class EvalResult:
    """Результат оценки кандидата на мини-батче."""

    aggregate_score: float
    failed_examples: List[FailedExample] = field(default_factory=list)


# --- Stubs ---


def sample_mini_batch(
    dataset: Sequence[str],
    targets: Sequence[str | int],
    size: int,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str | int]]:
    """Сэмплирует мини-батч из датасета.

    Returns:
        (samples, targets) — списки длины size (или меньше, если датасет меньше).
    """
    import random

    rng = random.Random(seed)
    n = min(size, len(dataset))
    indices = rng.sample(range(len(dataset)), n)
    return (
        [dataset[i] for i in indices],
        [targets[i] for i in indices],
    )


def _evaluate_candidate_stub(
    prompt: str,
    dataset: List[str],
    targets: List[str | int],
) -> EvalResult:
    """Заглушка Evaluator: оценивает кандидата на мини-батче.

    TODO: подключить coolprompt.evaluator.Evaluator.
    """
    return EvalResult(
        aggregate_score=0.0,
        failed_examples=[
            FailedExample(
                instance=dataset[i],
                assistant_answer="",
                metric_value=0.0,
                ground_truth=targets[i],
            )
            for i in range(min(3, len(dataset)))
        ],
    )


def _feedback_module_stub(
    failed_examples: List[FailedExample],
    k_samples: int,
) -> List[str]:
    """Заглушка FeedbackModule: по неудачным примерам выдаёт рекомендации.

    TODO: реализовать LLM-based feedback.
    """
    return [f"Consider improving based on example: {fe.instance[:50]}..." for fe in failed_examples[:k_samples]]


def filter_recommendations(recommendations: List[str]) -> List[str]:
    """Фильтрует рекомендации (заглушка).

    TODO: убрать дубликаты, нерелевантные и т.д.
    """
    return list(recommendations)


# --- HyPEROptimizer ---


class HyPEROptimizer(Optimizer):
    """HyPE с итеративным уточнением через рекомендации на основе оценки."""

    def __init__(
        self,
        model: Any,
        *,
        n_candidates: int = 3,
        top_n_candidates: int = 2,
        k_samples: int = 3,
        mini_batch_size: int = 16,
        n_iterations: int = 2,
    ) -> None:
        super().__init__(model)
        self.hype = HyPEOptimizer(model)
        self.n_candidates = n_candidates
        self.top_n_candidates = top_n_candidates
        self.k_samples = k_samples
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations

    def optimize(
        self,
        prompt: str,
        dataset: Sequence[str],
        targets: Sequence[str | int],
        meta_info: Optional[dict[str, Any]] = None,
    ) -> str:
        """Генерирует кандидатов, оценивает, обновляет recommendations, повторяет."""
        hype = self.hype
        best_candidate = prompt

        for iteration in range(self.n_iterations):
            # 1. Генерация n_candidates
            candidates: List[str] = []
            for _ in range(self.n_candidates):
                candidate = hype.optimize(prompt, meta_info)
                candidates.append(candidate)

            if not candidates:
                return best_candidate

            # 2. Мини-батч
            samples, sample_targets = sample_mini_batch(
                dataset, targets, self.mini_batch_size
            )
            if not samples:
                best_candidate = candidates[0]
                if iteration == self.n_iterations - 1:
                    return best_candidate
                continue

            # 3. Оценка (заглушка Evaluator)
            scored: List[Tuple[float, str, EvalResult]] = []
            for cand in candidates:
                res = _evaluate_candidate_stub(cand, samples, sample_targets)
                scored.append((res.aggregate_score, cand, res))

            # 4. Top-k кандидатов
            scored.sort(key=lambda x: x[0], reverse=True)
            best_candidate = scored[0][1]

            if iteration == self.n_iterations - 1:
                return best_candidate

            top = scored[: self.top_n_candidates]

            # 5. Собираем k_samples FailedExample для top
            all_failed: List[FailedExample] = []
            for _, _, res in top:
                for fe in res.failed_examples[: self.k_samples]:
                    all_failed.append(fe)

            # 6. FeedbackModule → рекомендации
            recs = _feedback_module_stub(all_failed, self.k_samples)
            recs = filter_recommendations(recs)

            # 7. Обновляем recommendations в мета-промпте hype
            hype.update_section("recommendations", recs)

        return best_candidate
