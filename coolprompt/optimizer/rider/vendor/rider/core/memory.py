"""
Long-term Memory для накопления знаний между поколениями.

Этот модуль реализует механизм долгосрочной памяти, который накапливает
паттерны успешных и неудачных промптов для передачи опыта между поколениями.

Используется в reflection_crossover и других операторах для создания более информированных мутаций.
"""

from typing import List, Dict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Запись в памяти с паттерном и метаданными"""
    text: str
    fitness: float
    generation: int
    is_success: bool  # True для success_patterns, False для failure_patterns
    metadata: Dict = field(default_factory=dict)


class LongTermMemory:
    """
    Long-term Memory для накопления паттернов успеха и провала.

    Собирает топ-N лучших и худших промптов из каждого поколения
    для анализа общих паттернов и генерации insights.

    Args:
        max_patterns: Максимальное количество паттернов каждого типа (default: 20)
        top_k: Количество топ промптов для извлечения из поколения (default: 3)
        bottom_k: Количество худших промптов для извлечения (default: 3)

    Example:
        >>> memory = LongTermMemory(max_patterns=20)
        >>> # Обновить память после поколения
        >>> generation_prompts = [...]  # Список Prompt объектов
        >>> memory.update(generation_prompts, generation=5)
        >>> # Получить контекст для использования в мутациях
        >>> context = memory.get_context()
        >>> print(context['success_patterns'][:3])
        >>> # Генерировать LLM-сводку инсайтов
        >>> insights = memory.generate_insights(llm_client, model="gpt-4")
    """

    def __init__(
        self,
        max_patterns: int = 20,
        top_k: int = 3,
        bottom_k: int = 3
    ):
        """Инициализация Long-term Memory"""
        self.max_patterns = max_patterns
        self.top_k = top_k
        self.bottom_k = bottom_k

        # Хранилище паттернов
        self.success_patterns: List[MemoryEntry] = []
        self.failure_patterns: List[MemoryEntry] = []

        # История обновлений для tracking
        self.update_history: List[Dict] = []

        logger.info(
            f"LongTermMemory initialized: "
            f"max_patterns={max_patterns}, top_k={top_k}, bottom_k={bottom_k}"
        )

    def update(
        self,
        population: List,  # List[Prompt]
        generation: int
    ) -> Dict:
        """
        Обновить память на основе текущей популяции.

        Извлекает топ-K лучших и bottom-K худших промптов из популяции
        и добавляет их в соответствующие хранилища.

        Args:
            population: Список Prompt объектов с fitness
            generation: Номер текущего поколения

        Returns:
            Словарь с информацией об обновлении:
            {
                'generation': int,
                'added_success': int,
                'added_failure': int,
                'total_success': int,
                'total_failure': int
            }
        """
        if not population:
            logger.warning(f"Gen {generation}: Empty population, skipping memory update")
            return {
                'generation': generation,
                'added_success': 0,
                'added_failure': 0,
                'total_success': len(self.success_patterns),
                'total_failure': len(self.failure_patterns)
            }

        # Сортировать по fitness
        sorted_population = sorted(population, key=lambda p: p.fitness, reverse=True)

        # Извлечь топ-K успешных
        top_prompts = sorted_population[:self.top_k]
        for prompt in top_prompts:
            entry = MemoryEntry(
                text=self._extract_pattern(prompt.text),
                fitness=prompt.fitness,
                generation=generation,
                is_success=True,
                metadata={
                    'prompt_id': prompt.id,
                    'mutation_type': prompt.mutation_type,
                    'parent_ids': prompt.parent_ids
                }
            )
            self.success_patterns.append(entry)

        # Извлечь bottom-K провальных
        bottom_prompts = sorted_population[-self.bottom_k:] if len(sorted_population) > self.bottom_k else []
        for prompt in bottom_prompts:
            entry = MemoryEntry(
                text=self._extract_pattern(prompt.text),
                fitness=prompt.fitness,
                generation=generation,
                is_success=False,
                metadata={
                    'prompt_id': prompt.id,
                    'mutation_type': prompt.mutation_type,
                    'parent_ids': prompt.parent_ids
                }
            )
            self.failure_patterns.append(entry)

        # Ограничить размер хранилищ
        if len(self.success_patterns) > self.max_patterns:
            # Удалить самые старые
            self.success_patterns = self.success_patterns[-self.max_patterns:]

        if len(self.failure_patterns) > self.max_patterns:
            self.failure_patterns = self.failure_patterns[-self.max_patterns:]

        # Логирование
        update_info = {
            'generation': generation,
            'added_success': len(top_prompts),
            'added_failure': len(bottom_prompts),
            'total_success': len(self.success_patterns),
            'total_failure': len(self.failure_patterns),
            'best_fitness': top_prompts[0].fitness if top_prompts else 0.0,
            'worst_fitness': bottom_prompts[-1].fitness if bottom_prompts else 0.0
        }

        self.update_history.append(update_info)

        logger.info(
            f"Gen {generation}: Memory updated - "
            f"added {len(top_prompts)} success, {len(bottom_prompts)} failure patterns. "
            f"Total: {len(self.success_patterns)} success, {len(self.failure_patterns)} failure"
        )

        return update_info

    def _extract_pattern(self, text: str, max_length: int = 200) -> str:
        """
        Извлечь ключевые инструктивные паттерны из текста промпта.
        Семантическая экстракция вместо простого truncation.

        Сохраняет первое предложение (главная инструкция) + предложения
        с action verbs (step, classify, analyze и т.д.).
        """
        if not text:
            return ""
        if len(text) <= max_length:
            return text

        # Разбиваем на предложения
        sentences = [s.strip() for s in text.replace('. ', '.\n').split('\n') if s.strip()]

        if len(sentences) <= 2:
            return text[:max_length]

        # Ключевые слова действий для извлечения
        action_words = [
            'step', 'classify', 'answer', 'solve', 'summarize', 'generate',
            'identify', 'explain', 'analyze', 'extract', 'provide', 'ensure',
            'focus', 'consider', 'determine', 'evaluate', 'compare', 'create'
        ]

        # Всегда сохраняем первое предложение + предложения с action verbs
        key_sentences = [sentences[0]]
        for sent in sentences[1:]:
            if any(word in sent.lower() for word in action_words):
                key_sentences.append(sent)

        result = '. '.join(key_sentences)
        return result[:max_length] if len(result) > max_length else result

    def get_context(self, last_n: int = 5) -> Dict[str, List[str]]:
        """
        Получить контекст для использования в mutations.

        Args:
            last_n: Количество последних паттернов каждого типа

        Returns:
            Словарь:
            {
                'success': [список успешных паттернов],
                'failure': [список провальных паттернов]
            }
        """
        # Взять последние N паттернов
        recent_success = self.success_patterns[-last_n:] if len(self.success_patterns) > 0 else []
        recent_failure = self.failure_patterns[-last_n:] if len(self.failure_patterns) > 0 else []

        context = {
            'success': [entry.text for entry in recent_success],
            'failure': [entry.text for entry in recent_failure]
        }

        logger.debug(
            f"Retrieved context: {len(context['success'])} success, "
            f"{len(context['failure'])} failure patterns"
        )

        return context

    def generate_insights(
        self,
        llm_client,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ) -> str:
        """
        Генерировать текстовую сводку insights из накопленных паттернов.

        Использует LLM для анализа success/failure паттернов и
        извлечения общих закономерностей.

        Args:
            llm_client: LLM клиент для генерации
            model: Модель для использования
            temperature: Температура генерации

        Returns:
            Текстовая сводка insights
        """
        if not self.success_patterns and not self.failure_patterns:
            return "No patterns accumulated yet."

        # Подготовить промпт для LLM
        context = self.get_context(last_n=10)

        prompt = f"""Analyze these prompt patterns and provide insights:

SUCCESSFUL PATTERNS (high fitness):
{chr(10).join(f"- {p}" for p in context['success'])}

UNSUCCESSFUL PATTERNS (low fitness):
{chr(10).join(f"- {p}" for p in context['failure'])}

Please identify:
1. Common characteristics of successful prompts
2. Common pitfalls in unsuccessful prompts
3. Key differences between successful and unsuccessful approaches

Provide concise, actionable insights."""

        try:
            insights = llm_client.generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=500
            )

            logger.info(f"Generated insights from {len(context['success'])} success, {len(context['failure'])} failure patterns")

            return insights

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return f"Failed to generate insights: {e}"

    def get_summary(self) -> str:
        """
        Получить текстовую сводку памяти (без LLM).

        Returns:
            Текстовая сводка накопленных паттернов
        """
        if not self.success_patterns and not self.failure_patterns:
            return "No patterns in memory yet."

        summary_parts = []

        if self.success_patterns:
            summary_parts.append("SUCCESSFUL PATTERNS:")
            for entry in self.success_patterns[-5:]:  # Последние 5
                summary_parts.append(f"  [Gen {entry.generation}, Fitness {entry.fitness:.3f}] {entry.text}")

        if self.failure_patterns:
            summary_parts.append("\nUNSUCCESSFUL PATTERNS:")
            for entry in self.failure_patterns[-5:]:  # Последние 5
                summary_parts.append(f"  [Gen {entry.generation}, Fitness {entry.fitness:.3f}] {entry.text}")

        return "\n".join(summary_parts)

    def get_statistics(self) -> Dict:
        """
        Получить статистику памяти для анализа.

        Returns:
            Словарь со статистикой:
            - total_success: количество успешных паттернов
            - total_failure: количество провальных паттернов
            - avg_success_fitness: средний fitness успешных
            - avg_failure_fitness: средний fitness провальных
            - num_updates: количество обновлений памяти
        """
        stats = {
            'total_success': len(self.success_patterns),
            'total_failure': len(self.failure_patterns),
            'num_updates': len(self.update_history)
        }

        if self.success_patterns:
            stats['avg_success_fitness'] = sum(e.fitness for e in self.success_patterns) / len(self.success_patterns)
            stats['best_fitness'] = max(e.fitness for e in self.success_patterns)

        if self.failure_patterns:
            stats['avg_failure_fitness'] = sum(e.fitness for e in self.failure_patterns) / len(self.failure_patterns)
            stats['worst_fitness'] = min(e.fitness for e in self.failure_patterns)

        return stats

    def get_full_history(self) -> List[Dict]:
        """
        Получить полную историю обновлений памяти.

        Returns:
            Список словарей с информацией об обновлениях
        """
        return self.update_history.copy()

    def reset(self) -> None:
        """Очистить всю память"""
        self.success_patterns = []
        self.failure_patterns = []
        self.update_history = []
        logger.info("Long-term memory reset")

    def __repr__(self) -> str:
        """Строковое представление для отладки"""
        return (
            f"LongTermMemory("
            f"success={len(self.success_patterns)}, "
            f"failure={len(self.failure_patterns)}, "
            f"updates={len(self.update_history)})"
        )
