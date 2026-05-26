"""
GENESIS Protocol — Generative Evolutionary Narrative for Strategy Improvement through Self-reflection.

Каждый оператор накапливает "уроки" — объяснения ПОЧЕМУ определённые мутации привели к улучшению.
В отличие от OPERATOR FORGE (который хранит лучшие выходы), GENESIS хранит ПРИЧИНЫ успеха.

Вдохновлён GEPA (ICLR 2026 Oral) trace reflection, адаптирован для single-prompt optimization.

References:
- GEPA: Reflective Prompt Evolution (ICLR 2026 Oral)
- ReEvo: LLMs as Hyper-Heuristics with Reflective Evolution (NeurIPS 2024)
"""

import logging
import threading
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class GenesisMemory:
    """
    Ancestral Lesson Memory для эволюционных операторов.

    После каждого успешного offspring (fitness > parent fitness):
    1. Извлекает "урок" через LLM: ПОЧЕМУ эта мутация улучшила промт
    2. Хранит top-K уроков на оператор, сортированных по fitness delta
    3. Инжектит уроки как контекст при будущих вызовах операторов

    Это позволяет операторам учиться на своих ПРОШЛЫХ УСПЕХАХ — не просто
    повторять хорошие выходы (как FORGE), а понимать ПРИНЦИПЫ что работает.
    """

    def __init__(self, max_lessons: int = 5):
        self.max_lessons = max_lessons
        self._lessons: Dict[str, List[dict]] = {}  # operator -> [{lesson, delta, gen}]
        self._lock = threading.Lock()
        self._total_lessons_extracted = 0
        self._extraction_failures = 0

    def extract_lesson(
        self,
        parent_text: str,
        offspring_text: str,
        operator_name: str,
        fitness_delta: float,
        generation: int,
        task_description: str,
        llm_client: Any,
        temperature: float = 0.3
    ) -> Optional[str]:
        """
        Извлечь урок из успешной мутации через LLM.

        Вызывается когда offspring.fitness > parent.fitness.
        Стоимость: 1 LLM call (~100-200 tokens output).

        Returns:
            str: Extracted lesson, or None on failure
        """
        # Truncate prompts to save tokens
        parent_short = parent_text[:300] + "..." if len(parent_text) > 300 else parent_text
        offspring_short = offspring_text[:300] + "..." if len(offspring_text) > 300 else offspring_text

        extraction_prompt = f"""Analyze this successful prompt mutation and extract a concise lesson.

TASK: {task_description[:200]}

PARENT PROMPT (fitness: lower):
{parent_short}

IMPROVED OFFSPRING (fitness delta: +{fitness_delta:.4f}):
{offspring_short}

OPERATOR USED: {operator_name}

What SPECIFIC change made the offspring better? Extract ONE concise lesson (1-2 sentences) that could guide future mutations. Focus on the PRINCIPLE, not the specific words.

Format: Just the lesson text, nothing else."""

        try:
            response = llm_client.call(
                prompt=extraction_prompt,
                system_prompt="You are an expert at analyzing prompt improvements. Be concise and specific.",
                temperature=temperature,
                max_tokens=150
            )

            lesson_text = (response or "").strip()
            if len(lesson_text) < 10:
                with self._lock:
                    self._extraction_failures += 1
                return None

            # Store the lesson
            with self._lock:
                if operator_name not in self._lessons:
                    self._lessons[operator_name] = []

                self._lessons[operator_name].append({
                    'lesson': lesson_text,
                    'delta': fitness_delta,
                    'generation': generation,
                })

                # Keep top-K by delta
                self._lessons[operator_name].sort(key=lambda x: x['delta'], reverse=True)
                self._lessons[operator_name] = self._lessons[operator_name][:self.max_lessons]
                self._total_lessons_extracted += 1
            logger.info(f"GENESIS: Extracted lesson for {operator_name} (Δ={fitness_delta:.4f}): {lesson_text[:80]}...")
            return lesson_text

        except Exception as e:
            with self._lock:
                self._extraction_failures += 1
            logger.warning(f"GENESIS: Failed to extract lesson for {operator_name}: {e}")
            return None

    def get_context(self, operator_name: str) -> str:
        """
        Получить контекст уроков для оператора.

        Вызывается ПЕРЕД каждым вызовом оператора.
        Возвращает форматированный блок уроков для инъекции в task_desc.
        Стоимость: 0 LLM calls (чтение из памяти).
        """
        with self._lock:
            lessons = self._lessons.get(operator_name, [])

        if not lessons:
            return ""

        lines = ["", "LESSONS FROM PREVIOUS SUCCESSFUL MUTATIONS:"]
        for i, entry in enumerate(lessons, 1):
            lines.append(f"  {i}. [Δ={entry['delta']:.4f}] {entry['lesson']}")
        lines.append("Apply these lessons to create an even better prompt.")
        lines.append("")

        return "\n".join(lines)

    def get_all_context(self) -> str:
        """Получить уроки от ВСЕХ операторов (для VORTEX и stagnation escape)."""
        with self._lock:
            all_lessons = []
            for op, lessons in self._lessons.items():
                for entry in lessons:
                    all_lessons.append((op, entry))

        if not all_lessons:
            return ""

        # Sort by delta across all operators
        all_lessons.sort(key=lambda x: x[1]['delta'], reverse=True)

        lines = ["", "KEY LESSONS FROM EVOLUTION HISTORY:"]
        for op, entry in all_lessons[:7]:  # Top 7 across all operators
            lines.append(f"  [{op}, Δ={entry['delta']:.4f}] {entry['lesson']}")
        lines.append("")

        return "\n".join(lines)

    def has_lessons(self, operator_name: str) -> bool:
        """Check if operator has any lessons."""
        with self._lock:
            return bool(self._lessons.get(operator_name))

    def get_statistics(self) -> dict:
        """Статистика для логирования."""
        with self._lock:
            per_op = {op: len(lessons) for op, lessons in self._lessons.items()}
            total_lessons = self._total_lessons_extracted
            failures = self._extraction_failures
        return {
            'total_lessons': total_lessons,
            'failures': failures,
            'per_operator': per_op,
            'total_stored': sum(per_op.values())
        }

    def to_dict(self) -> dict:
        """Serialize for checkpoint saving."""
        with self._lock:
            return {
                'lessons': {op: list(lessons) for op, lessons in self._lessons.items()},
                'total_extracted': self._total_lessons_extracted,
                'failures': self._extraction_failures
            }

    @classmethod
    def from_dict(cls, data: dict, max_lessons: int = 5) -> 'GenesisMemory':
        """Restore from checkpoint."""
        memory = cls(max_lessons=max_lessons)
        memory._lessons = data.get('lessons', {})
        memory._total_lessons_extracted = data.get('total_extracted', 0)
        memory._extraction_failures = data.get('failures', 0)
        return memory
