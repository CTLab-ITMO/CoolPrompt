"""
Управление разнообразием популяции и k-DPP отбор ансамбля.

Этот модуль реализует два ключевых механизма:
1. DiversityManager - контроль семантического разнообразия через embeddings
2. kDPPSelector - Determinantal Point Process для выбора diverse ансамбля

Используется для предотвращения преждевременной сходимости к локальным оптимумам.
"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DiversityManager:
    """
    Управление разнообразием в популяции промптов.

    Использует SentenceTransformer embeddings для вычисления семантического
    сходства между промптами и фильтрации слишком похожих кандидатов.

    Args:
        sentence_encoder: SentenceTransformer модель для embeddings
        diversity_threshold: Порог cosine similarity для фильтрации (default: 0.72)
            Промпты с similarity > threshold отклоняются
            ИСПРАВЛЕНО: 0.85→0.72 для лучшего semantic diversity control

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> encoder = SentenceTransformer('all-MiniLM-L6-v2')
        >>> manager = DiversityManager(encoder, diversity_threshold=0.72)
        >>> # Вычислить diversity score
        >>> score = manager.compute_diversity_score(population)
        >>> # Отобрать с diversity tiebreak
        >>> selected = manager.select_with_diversity_tiebreak(candidates, target_size=10)
    """

    def __init__(
        self,
        sentence_encoder,  # SentenceTransformer
        diversity_threshold: float = 0.95,  # Мягкий threshold (было 0.72)
        adaptive_threshold: bool = False,  # Fixed threshold (adaptive не нужен при 0.95)
        min_threshold: float = 0.95,  # Фиксированный (было 0.65)
        max_threshold: float = 0.95   # Фиксированный (было 0.80)
    ):
        """
        Инициализация Diversity Manager.

        Threshold 0.95 — отклоняет только ПОЧТИ ИДЕНТИЧНЫЕ промпты.
        Старый threshold 0.65-0.80 отклонял 75-85% offspring, убивая эволюцию.
        PromptBreeder не имеет diversity filter вообще (acceptance_rate=100%).
        Threshold 0.95 сохраняет фишку RIDER (diversity awareness) без убийства валидных offspring.

        Args:
            sentence_encoder: SentenceTransformer модель (all-MiniLM-L6-v2)
            diversity_threshold: Base порог similarity для отклонения
            adaptive_threshold: Использовать adaptive threshold
            min_threshold: Минимальный threshold
            max_threshold: Максимальный threshold
        """
        self.encoder = sentence_encoder
        self.base_threshold = diversity_threshold
        self.adaptive = adaptive_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Для stagnation detection
        self.fitness_history: List[float] = []

        threshold_type = "adaptive" if adaptive_threshold else "fixed"
        logger.info(
            f"DiversityManager initialized: {threshold_type} threshold "
            f"({min_threshold:.2f}-{max_threshold:.2f}), "
            f"encoder={type(sentence_encoder).__name__}"
        )

    def get_adaptive_threshold(self, generation: int, total_generations: int) -> float:
        """
        Вычислить adaptive threshold для текущего поколения.

        Formula (EvoPrompt++, 2024):
            threshold = min_threshold + (max_threshold - min_threshold) * (generation / total_generations)

        - Начало (gen 0): 0.65 (строгий, высокое diversity для exploration)
        - Конец (gen N): 0.80 (ослабленный, exploitation)

        Также учитывает stagnation: если fitness не растёт, увеличить diversity.

        Args:
            generation: Текущее поколение
            total_generations: Общее количество поколений

        Returns:
            Adaptive threshold в [min_threshold, max_threshold]
        """
        if not self.adaptive:
            return self.base_threshold

        # Linear schedule: от min к max по мере прогресса
        progress = generation / max(total_generations, 1)
        threshold = self.min_threshold + (self.max_threshold - self.min_threshold) * progress

        # Stagnation detection: если fitness не растёт, увеличить diversity
        if len(self.fitness_history) >= 3:
            recent = self.fitness_history[-3:]
            improvement = (recent[-1] - recent[0]) / max(recent[0], 0.001)
            if improvement < 0.01:  # < 1% improvement over 3 generations
                # Stagnation detected: сделать threshold строже для escape
                threshold -= 0.05
                logger.debug(
                    f"Gen {generation}: Stagnation detected, "
                    f"threshold adjusted: {threshold + 0.05:.3f} → {threshold:.3f}"
                )

        # Clip to bounds
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))

        return threshold

    def update_fitness_history(self, best_fitness: float) -> None:
        """
        Обновить историю fitness для stagnation detection.

        Args:
            best_fitness: Лучший fitness текущего поколения
        """
        self.fitness_history.append(best_fitness)
        # Хранить только последние 5 значений
        if len(self.fitness_history) > 5:
            self.fitness_history.pop(0)

    def compute_embeddings(self, prompts: List) -> None:
        """
        Вычислить embeddings для всех промптов.

        Embeddings сохраняются в prompt.diversity_features

        Args:
            prompts: Список Prompt объектов
        """
        if not prompts:
            return

        texts = [p.text for p in prompts]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        for prompt, embedding in zip(prompts, embeddings):
            prompt.diversity_features = embedding

        logger.debug(f"Computed embeddings for {len(prompts)} prompts")

    def compute_similarity_matrix(self, prompts: List) -> np.ndarray:
        """
        Вычислить матрицу попарных косинусных сходств.

        Args:
            prompts: Список Prompt объектов

        Returns:
            Матрица similarity [N x N], где N = len(prompts)
        """
        # Убедиться что embeddings вычислены
        if any(p.diversity_features is None for p in prompts):
            self.compute_embeddings(prompts)

        embeddings = np.array([p.diversity_features for p in prompts])

        # Cosine similarity через normalized dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarity_matrix = normalized @ normalized.T

        return similarity_matrix

    def compute_diversity_score(self, population: List) -> float:
        """
        Вычислить diversity score популяции.

        Diversity = среднее попарное cosine distance.

        Args:
            population: Список Prompt объектов

        Returns:
            Diversity score в [0, 1], где 1 = максимальное разнообразие
        """
        if len(population) < 2:
            return 1.0

        similarity_matrix = self.compute_similarity_matrix(population)

        # Взять верхний треугольник без диагонали
        n = len(population)
        upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]

        # Diversity = 1 - average similarity
        avg_similarity = np.mean(upper_triangle)
        diversity = 1.0 - avg_similarity

        return float(diversity)

    def get_statistics(self, population: List) -> Dict:
        """
        Получить статистику разнообразия популяции.

        Returns:
            Словарь со статистикой:
            - diversity_score: общий diversity score
            - avg_similarity: средняя similarity
            - min_similarity: минимальная similarity (самые различные)
            - max_similarity: максимальная similarity (самые похожие)
        """
        if len(population) < 2:
            return {
                'diversity_score': 1.0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0
            }

        similarity_matrix = self.compute_similarity_matrix(population)
        n = len(population)
        upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]

        return {
            'diversity_score': 1.0 - np.mean(upper_triangle),
            'avg_similarity': float(np.mean(upper_triangle)),
            'min_similarity': float(np.min(upper_triangle)),
            'max_similarity': float(np.max(upper_triangle))
        }


    def select_with_diversity_tiebreak(
        self,
        candidates: list,  # List[Prompt]
        target_size: int,
        similarity_threshold: float = 0.92  # ужесточён с 0.98 (diversity collapse)
    ) -> list:
        """
        Diversity as tiebreaker, not gate.

        Selects target_size prompts from candidates using fitness-first selection
        with diversity as tiebreaker for near-identical prompts only.

        1. Sort by fitness (descending)
        2. For each candidate: add if cosine_sim < threshold with all selected
        3. If blocked AND fitness within 1% of similar prompt -> skip (true duplicate)
        4. If blocked BUT fitness > 1% better than similar prompt -> REPLACE it
        5. Fill remaining from unselected candidates by fitness

        Args:
            candidates: Combined pool of parents + offspring (all evaluated)
            target_size: How many to keep
            similarity_threshold: Cosine similarity threshold (0.98 = near-identical only)

        Returns:
            List of selected prompts
        """
        if len(candidates) <= target_size:
            return list(candidates)

        # Compute embeddings for all candidates
        self.compute_embeddings(candidates)

        # Sort by fitness descending
        sorted_candidates = sorted(candidates, key=lambda p: p.fitness, reverse=True)

        selected = []

        for candidate in sorted_candidates:
            if len(selected) >= target_size:
                break

            if not selected:
                selected.append(candidate)
                continue

            # Check similarity with all selected
            candidate_emb = candidate.diversity_features
            candidate_norm = candidate_emb / (np.linalg.norm(candidate_emb) + 1e-8)

            max_sim = 0.0
            most_similar_idx = -1
            for i, sel in enumerate(selected):
                sel_emb = sel.diversity_features
                sel_norm = sel_emb / (np.linalg.norm(sel_emb) + 1e-8)
                sim = float(np.dot(candidate_norm, sel_norm))
                if sim > max_sim:
                    max_sim = sim
                    most_similar_idx = i

            if max_sim < similarity_threshold:
                # Different enough — add
                selected.append(candidate)
            else:
                # Near-identical to selected[most_similar_idx]
                similar_prompt = selected[most_similar_idx]
                fitness_diff = (candidate.fitness - similar_prompt.fitness) / max(abs(similar_prompt.fitness), 0.001)

                if fitness_diff > 0.01:
                    # Candidate is >1% better -> REPLACE the similar one
                    selected[most_similar_idx] = candidate
                    logger.debug(
                        f"Diversity tiebreak: replaced similar prompt "
                        f"(fitness {similar_prompt.fitness:.4f} -> {candidate.fitness:.4f})"
                    )
                # else: True duplicate or worse -> skip

        # If we still need more, fill from remaining candidates by fitness
        if len(selected) < target_size:
            selected_ids = {id(p) for p in selected}
            for candidate in sorted_candidates:
                if id(candidate) not in selected_ids:
                    selected.append(candidate)
                    if len(selected) >= target_size:
                        break

        logger.debug(
            f"Diversity tiebreak: {len(candidates)} -> {len(selected)} "
            f"(threshold={similarity_threshold})"
        )

        return selected[:target_size]


class kDPPSelector:
    """
    k-DPP (Determinantal Point Process) для выбора diverse ансамбля.

    k-DPP максимизирует det(L_Y), где L - ядро комбинирующее
    quality (fitness) и diversity (embeddings similarity).

    Это обеспечивает выбор k промптов которые одновременно:
    - Высокого качества (high fitness)
    - Максимально различны (low similarity)

    Args:
        quality_weight: Вес quality vs diversity (default: 0.5)

    Example:
        >>> selector = kDPPSelector(quality_weight=0.5)
        >>> ensemble = selector.select_ensemble(population, k=5)
        >>> # ensemble содержит 5 diverse высококачественных промптов
    """

    def __init__(self, quality_weight: float = 0.5):
        """
        Инициализация k-DPP селектора.

        Args:
            quality_weight: Вес качества в ядре [0, 1]
                0.5 = равный баланс quality и diversity
        """
        self.quality_weight = quality_weight

        logger.info(f"kDPPSelector initialized with quality_weight={quality_weight}")

    def select_ensemble(
        self,
        prompts: List,  # List[Prompt]
        k: int = 5
    ) -> List:
        """
        Выбрать k-элементный ансамбль через k-DPP.

        Args:
            prompts: Список промптов для выбора
            k: Размер ансамбля

        Returns:
            Список выбранных k промптов
        """
        if len(prompts) <= k:
            return prompts

        # Quality scores (fitness)
        quality_scores = np.array([p.fitness for p in prompts])

        # Similarity matrix
        embeddings = np.array([p.diversity_features for p in prompts])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarity_matrix = normalized @ normalized.T

        # Greedy MAP inference (approximation для k-DPP)
        selected_indices = self.greedy_map_inference(
            quality_scores,
            similarity_matrix,
            k
        )

        selected_prompts = [prompts[i] for i in selected_indices]

        logger.info(
            f"Selected {len(selected_prompts)} prompts for ensemble via k-DPP. "
            f"Fitness range: [{min(p.fitness for p in selected_prompts):.3f}, "
            f"{max(p.fitness for p in selected_prompts):.3f}]"
        )

        return selected_prompts

    def greedy_map_inference(
        self,
        quality_scores: np.ndarray,
        similarity_matrix: np.ndarray,
        k: int
    ) -> List[int]:
        """
        Greedy алгоритм для приближенного решения k-DPP.

        Args:
            quality_scores: Вектор качества [N]
            similarity_matrix: Матрица similarity [N x N]
            k: Количество элементов для выбора

        Returns:
            Список индексов выбранных элементов
        """
        n = len(quality_scores)
        selected = []
        remaining = list(range(n))

        # Нормализовать quality scores
        if quality_scores.max() > quality_scores.min():
            norm_quality = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min())
        else:
            norm_quality = np.ones_like(quality_scores)

        for _ in range(k):
            if not remaining:
                break

            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                # Quality term
                quality_term = self.quality_weight * norm_quality[idx]

                # Diversity term (distance to selected)
                if selected:
                    # Минимальная similarity с уже выбранными
                    similarities_to_selected = similarity_matrix[idx, selected]
                    avg_similarity = np.mean(similarities_to_selected)
                    diversity_term = (1.0 - self.quality_weight) * (1.0 - avg_similarity)
                else:
                    diversity_term = (1.0 - self.quality_weight) * 1.0

                # Комбинированный score
                score = quality_term + diversity_term

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected
