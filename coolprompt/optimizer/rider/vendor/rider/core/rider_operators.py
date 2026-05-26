"""
RIDER-специфичные операторы с рефлексией и памятью.

Этот модуль реализует продвинутые эволюционные операторы для RIDER:
1. reflection_crossover - Кроссовер с рефлексией на примерах
2. opro_trajectory_mutation - OPRO-style trajectory (Yang et al. ICLR 2024)
3. contrastive_error_decomposition - Контрастивная декомпозиция ошибок
4. semantic_paraphrase - Семантический парафраз (PhaseEvo 2024)
5. vortex_mutation - Парадигменный сдвиг при стагнации

Эти операторы дополняют базовые из operators.py и реализуют механизмы
рефлексии, долговременной памяти и адаптации характерные для RIDER.
"""

from typing import List, Dict, Optional
import logging

from rider.core.prompts import Prompt
from rider.core.operators import EvolutionaryOperators, extract_prompt_from_response

logger = logging.getLogger(__name__)


class RIDEROperators(EvolutionaryOperators):
    """
    RIDER-специфичные операторы с рефлексией и памятью.

    Наследует базовые операторы (GA, DE, EDA, zero-order) и добавляет
    продвинутые механизмы:
    - Short-term и long-term рефлексия
    - Hypermutation с контекстом
    - OPRO trajectory, contrastive error decomposition и semantic paraphrase
    - Meta-optimization на основе статистики

    Args:
        llm_client: LLM клиент для генерации
        sentence_encoder: SentenceTransformer для embeddings
        long_term_memory: Объект LongTermMemory для накопления паттернов
        model: Модель для генерации
        temperature: Базовая температура

    Example:
        >>> from rider.llm.client import LLMClient
        >>> from rider.core.memory import LongTermMemory
        >>> from sentence_transformers import SentenceTransformer
        >>>
        >>> llm = LLMClient()
        >>> memory = LongTermMemory(max_patterns=20)
        >>> encoder = SentenceTransformer("all-MiniLM-L6-v2")
        >>> ops = RIDEROperators(llm, encoder, memory, model="gpt-4")
        >>>
        >>> # Reflection-guided crossover
        >>> offspring = ops.reflection_crossover(parent1, parent2, task_desc="Solve math")
    """

    def __init__(
        self,
        llm_client,  # LLMClient
        sentence_encoder=None,  # SentenceTransformer
        long_term_memory=None,  # LongTermMemory
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Инициализация RIDER операторов.

        Args:
            llm_client: LLM клиент
            sentence_encoder: Encoder для similarity
            long_term_memory: Долговременная память (опционально)
            model: Модель для генерации
            temperature: Базовая температура
        """
        super().__init__(llm_client, sentence_encoder, model, temperature)

        self.long_term_memory = long_term_memory

        # История рефлексий для анализа
        self.reflection_history: List[str] = []

        # Контекст для hypermutation
        self.hypermutation_context = {
            'evolved_mutations': [],
            'successful_patterns': [],
            'generation_insights': []
        }

        logger.info(
            f"RIDEROperators initialized with model={model}, "
            f"has_memory={long_term_memory is not None}"
        )

    # ========== Reflection Operators ==========

    def reflection_crossover(
        self,
        parent1: Prompt,
        parent2: Prompt,
        task_desc: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Prompt:
        """
        Упрощенный кроссовер с рефлексией.

        Анализирует подходы промптов и создает гибрид.

        Args:
            parent1: Первый родительский промпт
            parent2: Второй родительский промпт
            task_desc: Описание задачи
            temperature: Температура
            top_p: Top-p sampling

        Returns:
            Гибридный промпт
        """
        temp = temperature if temperature is not None else self.temperature

        reflection_prompt = f"""Compare these two prompts by analyzing their approach:

Prompt A (fitness {parent1.fitness:.3f}): "{parent1.text}"
Prompt B (fitness {parent2.fitness:.3f}): "{parent2.text}"

Task: {task_desc}

Analyze their differences and create a hybrid that combines their strengths.
Aim for 50-100 words. Be specific and detailed.

Wrap your answer in <prompt></prompt> tags.

<prompt>"""

        try:
            response = self.llm_client.generate(
                prompt=reflection_prompt,
                model=self.model,
                temperature=temp,
                max_tokens=350,
                top_p=top_p or 1.0
            )

            offspring_text = extract_prompt_from_response(response, self.preserve_cot)

            offspring = Prompt(
                text=offspring_text,
                generation=max(parent1.generation, parent2.generation) + 1,
                parent_ids=[parent1.id, parent2.id],
                mutation_type="reflection_crossover"
            )

            logger.debug(
                f"Reflection crossover: {parent1.id} + {parent2.id} → {offspring.id}"
            )

            return offspring

        except Exception as e:
            logger.error(f"Reflection crossover error: {e}")
            # ИСПРАВЛЕНО: Правильный fallback с копией и метаданными
            return Prompt(
                text=parent1.text,
                generation=max(parent1.generation, parent2.generation) + 1,
                parent_ids=[parent1.id, parent2.id],
                mutation_type="reflection_crossover_fallback",
                fitness=parent1.fitness,
                metadata={'error': str(e)}
            )

    # ========== New Operators ==========

    def opro_trajectory_mutation(
        self,
        population: List[Prompt],
        task_description: str,
        dataset_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Prompt:
        """
        OPRO-style trajectory-based generation (Yang et al. ICLR 2024).

        LLM видит топ-20 промптов с их scores (отсортированных по возрастанию)
        и генерирует новый промпт, который должен набрать больше.
        LLM видит глобальный ландшафт оптимизации, а не 1-2 родителей.

        Args:
            population: Текущая популяция промптов
            task_description: Описание задачи
            dataset_name: Название датасета (опционально)
            temperature: Температура
            top_p: Top-p sampling

        Returns:
            Новый промпт, нацеленный на превышение всех показанных scores
        """
        temp = temperature if temperature is not None else self.temperature

        # Sort by fitness ascending (OPRO shows worst-to-best)
        sorted_pop = sorted(population, key=lambda p: p.fitness)
        top_k = sorted_pop[-20:]  # last 20 = best 20

        trajectory_str = "\n".join([
            f"Instruction: \"{p.text}\"\nScore: {p.fitness:.4f}"
            for p in top_k
        ])

        opro_prompt = (
            f"Task: {task_description}\n\n"
            f"Below are some instructions and their scores. "
            f"The score represents task accuracy (higher is better).\n\n"
            f"{trajectory_str}\n\n"
            f"Generate a new instruction that will score higher than all of the above. "
            f"Aim for 50-100 words. Be specific about output format, edge cases, and constraints.\n\n"
            f"Wrap your answer in <prompt></prompt> tags.\n\n"
            f"<prompt>"
        )

        try:
            response = self.llm_client.generate(
                prompt=opro_prompt,
                model=self.model,
                temperature=temp,
                max_tokens=350,
                top_p=top_p or 1.0
            )

            offspring_text = extract_prompt_from_response(response, self.preserve_cot)

            if not offspring_text or len(offspring_text.split()) < 15:
                logger.warning(
                    f"OPRO trajectory mutation produced short output "
                    f"({len(offspring_text.split()) if offspring_text else 0} words), "
                    f"falling back to best prompt"
                )
                best = sorted_pop[-1]
                return Prompt(
                    text=best.text,
                    generation=best.generation + 1,
                    parent_ids=[best.id],
                    mutation_type="opro_trajectory_mutation_fallback",
                    fitness=best.fitness,
                    metadata={'fallback': True}
                )

            best = sorted_pop[-1]
            offspring = Prompt(
                text=offspring_text,
                generation=best.generation + 1,
                parent_ids=[p.id for p in top_k[-3:]],  # Top-3 as parents
                mutation_type="opro_trajectory_mutation",
                metadata={
                    'parent_fitness': best.fitness,
                    'trajectory_size': len(top_k)
                }
            )

            logger.debug(
                f"OPRO trajectory mutation: {len(top_k)} prompts → {offspring.id} "
                f"(words={len(offspring_text.split())})"
            )

            return offspring

        except Exception as e:
            logger.error(f"OPRO trajectory mutation error: {e}")
            best = sorted_pop[-1] if sorted_pop else population[0]
            return Prompt(
                text=best.text,
                generation=best.generation + 1,
                parent_ids=[best.id],
                mutation_type="opro_trajectory_mutation_fallback",
                fitness=best.fitness,
                metadata={'error': str(e)}
            )

    def contrastive_error_decomposition(
        self,
        elite_prompt: Prompt,
        error_examples: List[Dict],
        population: List[Prompt],
        task_description: str,
        dataset_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        failed_attempts: Optional[List[Dict]] = None,
        memory_context: Optional[Dict] = None
    ) -> Prompt:
        """
        Contrastive Error Decomposition — оригинальный оператор RIDER.

        Формирует контрастивные пары: примеры, которые элита решает ПРАВИЛЬНО
        vs НЕПРАВИЛЬНО. LLM анализирует минимальные различия между парами
        и генерирует правила, покрывающие ошибочные случаи без потери правильных.

        Математическая идея: находим decision boundary промпта через контрастивные
        примеры и строим instruction, которая сдвигает boundary в нужную сторону.

        Args:
            elite_prompt: Лучший промпт
            error_examples: Ошибки (input, prediction, ground_truth)
            population: Текущая популяция (для извлечения успешных примеров)
            task_description: Описание задачи
            dataset_name: Название датасета
            temperature: Температура
            top_p: Top-p sampling

        Returns:
            Промпт с правилами, извлечёнными из контрастивного анализа
        """
        temp = temperature if temperature is not None else self.temperature

        if not error_examples:
            logger.warning("contrastive_error_decomposition: no error examples, falling back to elite")
            return elite_prompt

        # Берём до 3 ошибочных примеров
        errors = error_examples[:3]

        # Извлекаем успешные примеры из метаданных элиты (если есть)
        successes = []
        if hasattr(elite_prompt, 'metadata') and elite_prompt.metadata:
            successes = elite_prompt.metadata.get('correct_examples', [])[:3]

        # Формируем контрастивные пары
        contrastive_pairs = []
        for i, err in enumerate(errors):
            pair = f"FAILURE {i+1}:\n"
            pair += f"  Input: {str(err.get('input', ''))[:200]}\n"
            pair += f"  Expected output: {err.get('ground_truth', '')}\n"
            pair += f"  Model produced: {err.get('prediction', '')}\n"
            contrastive_pairs.append(pair)

        for i, succ in enumerate(successes):
            pair = f"SUCCESS {i+1}:\n"
            pair += f"  Input: {str(succ.get('input', ''))[:200]}\n"
            pair += f"  Expected output: {succ.get('ground_truth', '')}\n"
            pair += f"  Model produced: {succ.get('prediction', '')}\n"
            contrastive_pairs.append(pair)

        pairs_str = "\n".join(contrastive_pairs)

        # Если нет успешных примеров, берём другие промпты из популяции
        # для контраста подходов
        approach_contrast = ""
        if not successes and population:
            sorted_pop = sorted(population, key=lambda p: p.fitness, reverse=True)
            other_approaches = [p for p in sorted_pop[:3] if p.id != elite_prompt.id]
            if other_approaches:
                approach_contrast = "\n\nAlternative instructions that partially work:\n"
                for j, alt in enumerate(other_approaches[:2]):
                    approach_contrast += f"  Alt {j+1} (fitness={alt.fitness:.3f}): \"{alt.text[:150]}\"\n"

        # Trace-enhanced context — show what was already tried
        trace_context = ""
        if failed_attempts:
            trace_context = "\n\nPrevious improvement attempts that did NOT help:\n"
            for i, attempt in enumerate(failed_attempts[-3:]):  # Last 3 attempts
                trace_context += f"  Attempt {i+1} (Gen {attempt.get('generation', '?')}): \"{attempt.get('prompt_snippet', '')}\"\n"
            trace_context += "Do NOT repeat these approaches. Try something fundamentally different.\n"

        # Memory context — accumulated success/failure patterns
        memory_str = ""
        if memory_context:
            if memory_context.get('success'):
                memory_str += "\nPatterns that historically WORK: " + "; ".join(memory_context['success'][:2])
            if memory_context.get('failure'):
                memory_str += "\nPatterns that historically FAIL: " + "; ".join(memory_context['failure'][:2])

        ced_prompt = (
            f"Task: {task_description}\n\n"
            f"Current best instruction:\n\"{elite_prompt.text}\"\n\n"
            f"Contrastive analysis — the instruction succeeds on some inputs but fails on others:\n"
            f"{pairs_str}\n"
            f"{approach_contrast}"
            f"{trace_context}"
            f"{memory_str}\n\n"
            f"Step 1: Identify the MINIMAL DIFFERENTIATING FACTOR between successful and failed cases. "
            f"What specific property of the failed inputs causes the instruction to break?\n\n"
            f"Step 2: Formulate 1-2 PRECISE RULES that would handle the failed cases "
            f"WITHOUT breaking the successful ones.\n\n"
            f"Step 3: Write a complete improved instruction (50-100 words) that incorporates "
            f"these rules naturally into the original instruction.\n\n"
            f"Wrap your final instruction in <prompt></prompt> tags.\n\n"
            f"<prompt>"
        )

        try:
            response = self.llm_client.generate(
                prompt=ced_prompt,
                model=self.model,
                temperature=temp,
                max_tokens=700,  # increased from 500 to prevent truncation
                top_p=top_p or 1.0
            )

            offspring_text = extract_prompt_from_response(response, self.preserve_cot)

            if not offspring_text or len(offspring_text.split()) < 15:
                logger.warning(
                    f"Contrastive error decomposition produced short output "
                    f"({len(offspring_text.split()) if offspring_text else 0} words), "
                    f"falling back to elite"
                )
                return elite_prompt

            offspring = Prompt(
                text=offspring_text,
                generation=elite_prompt.generation + 1,
                parent_ids=[elite_prompt.id],
                mutation_type="contrastive_error_decomposition",
                metadata={
                    'parent_fitness': elite_prompt.fitness,
                    'num_errors_analyzed': len(errors),
                    'num_successes_analyzed': len(successes)
                }
            )

            logger.debug(
                f"Contrastive error decomposition: {elite_prompt.id} "
                f"(fitness={elite_prompt.fitness:.3f}, errors={len(errors)}, "
                f"successes={len(successes)}) → "
                f"{offspring.id} (words={len(offspring_text.split())})"
            )

            return offspring

        except Exception as e:
            logger.error(f"Contrastive error decomposition error: {e}")
            return elite_prompt

    def semantic_paraphrase(
        self,
        prompt_obj: Prompt,
        task_description: str,
        dataset_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Prompt:
        """
        Semantic paraphrase mutation (PhaseEvo 2024).

        Сохраняет точную семантику промпта, меняя только поверхностную форму.
        Эксплуатирует чувствительность LLM к формулировкам (3-8% разброс accuracy).

        Args:
            prompt_obj: Промпт для парафраза
            task_description: Описание задачи
            dataset_name: Название датасета (опционально)
            temperature: Температура (по умолчанию 0.4 для сохранения семантики)
            top_p: Top-p sampling

        Returns:
            Парафраз промпта с сохранённой семантикой
        """
        # Low temperature for semantic preservation
        temp = temperature if temperature is not None else 0.4

        paraphrase_prompt = (
            f"Paraphrase the following instruction for a language model. "
            f"Keep the EXACT same meaning, rules, constraints, and edge cases. "
            f"Only change the wording, sentence structure, and ordering of clauses. "
            f"Do NOT add or remove any rules or information.\n\n"
            f"Original: {prompt_obj.text}\n\n"
            f"Wrap your answer in <prompt></prompt> tags.\n\n"
            f"<prompt>"
        )

        try:
            response = self.llm_client.generate(
                prompt=paraphrase_prompt,
                model=self.model,
                temperature=temp,
                max_tokens=350,
                top_p=top_p or 1.0
            )

            offspring_text = extract_prompt_from_response(response, self.preserve_cot)

            if not offspring_text or len(offspring_text.split()) < 15:
                logger.warning(
                    f"Semantic paraphrase produced short output "
                    f"({len(offspring_text.split()) if offspring_text else 0} words), "
                    f"falling back to parent"
                )
                return prompt_obj

            offspring = Prompt(
                text=offspring_text,
                generation=prompt_obj.generation + 1,
                parent_ids=[prompt_obj.id],
                mutation_type="semantic_paraphrase",
                metadata={'parent_fitness': prompt_obj.fitness}
            )

            logger.debug(
                f"Semantic paraphrase: {prompt_obj.id} "
                f"(fitness={prompt_obj.fitness:.3f}) → {offspring.id} "
                f"(words={len(offspring_text.split())})"
            )

            return offspring

        except Exception as e:
            logger.error(f"Semantic paraphrase error: {e}")
            return prompt_obj

    def vortex_mutation(self, population, task_desc, errors=None, dataset_name=None,
                         forge_context="", temperature=None, top_p=None):
        """VORTEX — Variance-Optimized Radical Transformation Explorer.

        Оператор парадигменного сдвига для выхода из стагнации. Вызывается только
        когда эволюция застряла. Вместо перефразирования существующих промптов,
        просит LLM изобрести ФУНДАМЕНТАЛЬНО ДРУГОЙ подход.

        Ключевые отличия от concept_brainstorm (0% success):
        - Вызывается только при стагнации (не каждое поколение)
        - Получает контекст популяции (знает что уже есть)
        - Получает контекст ошибок (знает что не работает)
        - Получает forge memory (знает что работало раньше)
        - Явная инструкция "paradigm shift" с примерами
        - Высокая температура (1.3) для максимальной креативности
        """
        # Суммаризация существующих подходов
        sorted_pop = sorted(population, key=lambda p: p.fitness, reverse=True)
        existing = []
        for p in sorted_pop[:5]:
            short = p.text[:120].replace('\n', ' ')
            existing.append(f"  [Score: {p.fitness:.3f}] {short}")
        approaches_str = "\n".join(existing)

        # Контекст ошибок
        error_block = ""
        if errors and len(errors) > 0:
            err_strs = []
            for e in errors[:3]:
                inp = str(e.get('input', ''))[:150]
                pred = str(e.get('prediction', ''))[:80]
                truth = str(e.get('ground_truth', ''))[:80]
                err_strs.append(f"Input: {inp}\nModel output: {pred}\nExpected: {truth}")
            error_block = "\n\nThe best prompt FAILS on these examples:\n" + "\n---\n".join(err_strs)

        # Forge context (что работало раньше)
        forge_block = ""
        if forge_context:
            forge_block = f"\n\n{forge_context}"

        prompt = f"""You are a prompt engineering researcher. The current population of prompts has STAGNATED — all variations produce similar results around the same score.

Task: {task_desc}

Current approaches in the population (all stuck at similar scores):
{approaches_str}
{error_block}{forge_block}

Your mission: invent a FUNDAMENTALLY DIFFERENT approach to this task.

NOT a rewording or refinement. A PARADIGM SHIFT in how the task is framed.

Examples of paradigm shifts:
- "Summarize the text" → "What would a news editor write as the headline?"
- "Generate a sentence with words" → "Describe the visual scene these words evoke"
- "Classify this text" → "If this were a newspaper section, which one?"
- "Answer the question" → "You are a search engine. Return the exact matching text span."

Invent a new framing that NONE of the current approaches use.
Write 50-100 words. Output ONLY the prompt text.

<prompt>"""

        t = temperature or max(self.temperature * 1.5, 1.3)
        tp = top_p or 0.99

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                model=self.model,
                temperature=t,
                top_p=tp,
                max_tokens=350
            )
            text = extract_prompt_from_response(response, getattr(self, 'preserve_cot', False))

            if not text or len(text.split()) < 10:
                logger.warning("VORTEX produced short output, falling back to zero_order")
                return self.zero_order_generation(task_desc, temperature=1.2, dataset_name=dataset_name)

            logger.info(f"VORTEX paradigm shift: {text[:80]}...")
            return Prompt(text=text, mutation_type='vortex')

        except Exception as e:
            logger.warning(f"VORTEX failed: {e}")
            return self.zero_order_generation(task_desc, temperature=1.2, dataset_name=dataset_name)

    def __repr__(self) -> str:
        """Строковое представление для отладки."""
        return (
            f"RIDEROperators("
            f"model={self.model}, "
            f"reflections={len(self.reflection_history)}, "
            f"hypermutations={len(self.hypermutation_context['evolved_mutations'])}, "
            f"has_memory={self.long_term_memory is not None})"
        )
