"""
Task-adaptive приоритеты операторов для разных датасетов.

Этот модуль определяет начальные веса операторов для UCB в зависимости
от типа задачи. Приоритеты основаны на эмпирических наблюдениях:

- GSM8K (математика): reflection + elitism
- AG_News (классификация): простые операторы (GA, EDA)
- SQuAD_2 (QA): reflection-based операторы
- CommonGen (генерация): EDA + zero_order + contrastive
- XSum (суммаризация): EDA + OPRO trajectory + contrastive

Веса используются для инициализации UCB статистики перед эволюцией.
"""

from typing import Dict

# Task-specific приоритеты операторов.
# lineage_based, elitist_mutation, concept_brainstorm заменены на:
#   opro_trajectory_mutation (OPRO, Yang et al. ICLR 2024)
#   contrastive_error_decomposition (оригинальный RIDER)
#   semantic_paraphrase (PhaseEvo 2024)
# EDA операторы доминируют (100% success в PromptBreeder).
TASK_OPERATOR_PRIORITIES: Dict[str, Dict[str, float]] = {
    'GSM8K': {
        # Математика: EDA + zero_order + reflection доминируют.
        # first_order_refinement=0.00 (negative ALL 6 experiments).
        # Вес перераспределён на opro_trajectory (+0.06) и semantic_paraphrase (+0.06).
        'eda_mutation': 0.22,
        'eda_rank_index': 0.16,
        'zero_order': 0.14,
        'first_order_refinement': 0.00,
        'reflection_crossover': 0.08,
        'de_mutation': 0.06,
        'ga_mutation': 0.04,
        'opro_trajectory_mutation': 0.13,
        'contrastive_error_decomposition': 0.06,
        'semantic_paraphrase': 0.11,
        'vortex': 0.00,  # VORTEX: triggered by stagnation, not by UCB selection
    },

    'AG_News': {
        # Классификация: EDA + zero_order.
        # first_order_refinement=0.00.
        'eda_mutation': 0.22,
        'eda_rank_index': 0.16,
        'zero_order': 0.14,
        'first_order_refinement': 0.00,
        'ga_mutation': 0.07,
        'de_mutation': 0.07,
        'reflection_crossover': 0.04,
        'opro_trajectory_mutation': 0.13,
        'contrastive_error_decomposition': 0.06,
        'semantic_paraphrase': 0.11,
        'vortex': 0.00,  # VORTEX: triggered by stagnation, not by UCB selection
    },

    'SQuAD_2': {
        # QA: EDA + opro + contrastive для детальных промптов.
        # first_order_refinement=0.00 (0% success на SQ2-gemini).
        # contrastive_error высокий — 70.6% success на SQ2-gemini (лучший оператор).
        'eda_mutation': 0.21,
        'eda_rank_index': 0.15,
        'zero_order': 0.11,
        'first_order_refinement': 0.00,
        'de_mutation': 0.06,
        'ga_mutation': 0.05,
        'reflection_crossover': 0.05,
        'opro_trajectory_mutation': 0.14,
        'contrastive_error_decomposition': 0.14,
        'semantic_paraphrase': 0.09,
        'vortex': 0.00,  # VORTEX: triggered by stagnation, not by UCB selection
    },

    'HotpotQA': {
        # Multi-hop QA: аналогично SQuAD_2, contrastive важен для multi-hop reasoning.
        'eda_mutation': 0.21,
        'eda_rank_index': 0.15,
        'zero_order': 0.11,
        'first_order_refinement': 0.00,
        'de_mutation': 0.06,
        'ga_mutation': 0.05,
        'reflection_crossover': 0.05,
        'opro_trajectory_mutation': 0.14,
        'contrastive_error_decomposition': 0.14,
        'semantic_paraphrase': 0.09,
        'vortex': 0.00,
    },

    'CommonGen': {
        # Генерация: EDA + zero_order + contrastive (boosted).
        # first_order_refinement=0.00.
        'eda_mutation': 0.20,
        'eda_rank_index': 0.14,
        'zero_order': 0.14,
        'de_mutation': 0.06,
        'first_order_refinement': 0.00,
        'ga_mutation': 0.05,
        'reflection_crossover': 0.02,
        'opro_trajectory_mutation': 0.12,
        'contrastive_error_decomposition': 0.17,
        'semantic_paraphrase': 0.10,
        'vortex': 0.00,  # VORTEX: triggered by stagnation, not by UCB selection
    },

    'XSum': {
        # Summarization: EDA + opro + contrastive (open framing).
        # first_order_refinement=0.00.
        # opro_trajectory boosted — 80% success на XSum-gemini (лучший оператор).
        'eda_mutation': 0.18,
        'eda_rank_index': 0.14,
        'zero_order': 0.13,
        'de_mutation': 0.05,
        'first_order_refinement': 0.00,
        'ga_mutation': 0.04,
        'reflection_crossover': 0.02,
        'opro_trajectory_mutation': 0.15,
        'contrastive_error_decomposition': 0.17,
        'semantic_paraphrase': 0.12,
        'vortex': 0.00,  # VORTEX: triggered by stagnation, not by UCB selection
    },

    'CodeSearchNet': {
        # Code summarization (генеративная задача).
        # first_order_refinement=0.00.
        'eda_mutation': 0.22,
        'eda_rank_index': 0.16,
        'zero_order': 0.15,
        'de_mutation': 0.09,
        'first_order_refinement': 0.00,
        'ga_mutation': 0.06,
        'reflection_crossover': 0.03,
        'opro_trajectory_mutation': 0.13,
        'contrastive_error_decomposition': 0.06,
        'semantic_paraphrase': 0.10,
        'vortex': 0.00,  # VORTEX: triggered by stagnation, not by UCB selection
    }
}

# Default конфигурация для неизвестных датасетов
DEFAULT_OPERATOR_PRIORITIES: Dict[str, float] = {
    'eda_mutation': 0.22,
    'eda_rank_index': 0.16,
    'zero_order': 0.14,
    'first_order_refinement': 0.00,  # excluded
    'de_mutation': 0.08,
    'ga_mutation': 0.06,
    'reflection_crossover': 0.04,
    'opro_trajectory_mutation': 0.13,
    'contrastive_error_decomposition': 0.06,
    'semantic_paraphrase': 0.11,
    'vortex': 0.00,  # VORTEX: triggered by stagnation, not by UCB selection
}


def get_task_operator_weights(dataset_name: str) -> Dict[str, float]:
    """
    Возвращает веса операторов для конкретной задачи.

    Веса используются для инициализации UCB статистики:
    - Операторы с высокими весами получают больше шансов в начале эволюции
    - UCB адаптирует веса во время эволюции на основе реальной эффективности

    Args:
        dataset_name: Название датасета (GSM8K, AG_News, SQuAD_2, CommonGen, XSum)

    Returns:
        Словарь {operator_name: initial_weight}
        Веса суммируются в 1.0 для каждого датасета

    Example:
        >>> weights = get_task_operator_weights('GSM8K')
        >>> print(weights['eda_mutation'])            # 0.22
        >>> print(weights['reflection_crossover'])    # 0.08
    """
    return TASK_OPERATOR_PRIORITIES.get(dataset_name, DEFAULT_OPERATOR_PRIORITIES)


def get_top_operators(dataset_name: str, top_k: int = 5) -> list:
    """
    Возвращает топ-K операторов для датасета по приоритету.

    Args:
        dataset_name: Название датасета
        top_k: Количество топ операторов

    Returns:
        Список tuple (operator_name, weight), отсортированный по убыванию веса

    Example:
        >>> top_ops = get_top_operators('GSM8K', top_k=3)
        >>> for op, weight in top_ops:
        ...     print(f"{op}: {weight:.1%}")
        eda_mutation: 22.0%
        eda_rank_index: 16.0%
        zero_order: 14.0%
    """
    weights = get_task_operator_weights(dataset_name)
    sorted_ops = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    return sorted_ops[:top_k]


def validate_priorities(priorities: Dict[str, float], tolerance: float = 0.01) -> bool:
    """
    Проверяет что веса суммируются в 1.0.

    Args:
        priorities: Словарь {operator: weight}
        tolerance: Допустимое отклонение от 1.0

    Returns:
        True если веса валидны

    Example:
        >>> weights = get_task_operator_weights('GSM8K')
        >>> is_valid = validate_priorities(weights)
        >>> print(is_valid)  # True
    """
    total = sum(priorities.values())
    return abs(total - 1.0) < tolerance


# Валидация конфигурации при импорте
def _validate_all_priorities():
    """Проверяет все конфигурации при импорте модуля."""
    for dataset, priorities in TASK_OPERATOR_PRIORITIES.items():
        if not validate_priorities(priorities):
            total = sum(priorities.values())
            raise ValueError(
                f"Invalid priorities for {dataset}: sum={total:.4f}, expected 1.0"
            )


# Запускаем валидацию
_validate_all_priorities()
