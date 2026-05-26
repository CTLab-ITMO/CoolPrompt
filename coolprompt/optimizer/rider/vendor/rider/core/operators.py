"""
Базовые эволюционные операторы для prompt optimization.

Этот модуль реализует 5 базовых эволюционных операторов:
1. ga_mutation - Генетическая мутация промпта
2. de_mutation - Дифференциальная эволюция (DE/current-to-best/1)
3. zero_order_generation - Генерация промпта с нуля (с валидацией и retry)
4. eda_mutation - EDA мутация на основе распределения популяции
5. eda_rank_index - EDA с ранжированием по fitness

Используется как база для RIDER и других эволюционных алгоритмов.
"""

import numpy as np
import random
import re
from typing import List, Optional
import logging

from rider.core.prompts import Prompt

logger = logging.getLogger(__name__)


def extract_prompt_from_response(response: str, preserve_cot: bool = False) -> str:
    """
    Универсальный парсер для извлечения промпта из ответа LLM.

    Все операторы просят LLM обернуть ответ в <prompt>...</prompt> теги.
    Парсер пытается извлечь текст из тегов; если тегов нет — fallback.

    preserve_cot=True для генеративных задач (CommonGen, XSum).
    На этих задачах CoT-preamble (markdown analysis, step-by-step reasoning)
    выступает как полезный scaffolding, повышая fitness на +3-5%.
    Если XML tags не найдены — возвращаем полный ответ без агрессивной чистки.

    Args:
        response: Полный ответ LLM
        preserve_cot: Если True — при отсутствии XML тегов возвращать полный ответ

    Returns:
        Очищенный текст промпта
    """
    if not response:
        return ""

    text = response.strip()

    # 1. XML tags: <prompt>...</prompt> — всегда приоритет
    match = re.search(r'<prompt>(.*?)</prompt>', text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if len(extracted) >= 10:
            return extracted

    # 2. Markdown-style: ```prompt ... ``` or ```...```
    match = re.search(r'```(?:prompt)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if len(extracted) >= 10:
            return extracted

    # Для генеративных задач — сохраняем CoT scaffolding.
    # Мета-комментарий (# Analysis, ## Step) повышает fitness на haiku.
    # EvoGA/EvoDE побеждают RIDER именно за счёт leaked CoT preamble.
    if preserve_cot:
        return text.strip('"\'').strip()

    # 3. Common markers (только для classification/extraction задач)
    markers = [
        "Improved prompt:", "New prompt:", "Updated prompt:",
        "Prompt:", "Modified prompt:", "Revised prompt:",
        "Improved instruction:", "New instruction:", "Updated instruction:",
        "Instruction:", "Modified instruction:", "Revised instruction:",
        "Combined prompt:", "Hybrid prompt:", "Novel prompt:",
        "Paraphrased prompt:", "Optimized prompt:", "Paraphrased version:",
        "New combined prompt:", "New radical prompt:", "New variant:",
    ]
    text_lower = text.lower()
    for marker in markers:
        idx = text_lower.find(marker.lower())
        if idx >= 0:
            extracted = text[idx + len(marker):].strip().strip('"\'')
            if len(extracted) >= 10:
                return extracted

    # 4. Remove meta-commentary prefixes
    meta_prefixes = [
        "here's the improved", "here is the improved",
        "here's the new", "here is the new",
        "here's my", "here is my",
        "let me think", "i'll improve", "sure,", "certainly,",
    ]
    for prefix in meta_prefixes:
        if text_lower.startswith(prefix):
            idx = text.find('\n')
            if idx > 0:
                text = text[idx:].strip()
            break

    # 5. Fallback: strip quotes
    return text.strip('"\'').strip()


class EvolutionaryOperators:
    """
    Базовые операторы для эволюционных алгоритмов prompt optimization.

    Каждый оператор принимает промпты и создает новый промпт через LLM.

    Args:
        llm_client: LLM клиент для генерации промптов
        sentence_encoder: SentenceTransformer для embeddings (опционально)
        model: Название модели для генерации (default: "gpt-3.5-turbo")
        temperature: Температура для генерации (default: 0.7)

    Example:
        >>> from rider.llm.client import LLMClient
        >>> from sentence_transformers import SentenceTransformer
        >>>
        >>> llm = LLMClient()
        >>> encoder = SentenceTransformer("all-MiniLM-L6-v2")
        >>> ops = EvolutionaryOperators(llm, encoder, model="gpt-4")
        >>>
        >>> # Mutation
        >>> mutant = ops.ga_mutation(parent, "Classify news articles")
        >>>
        >>> # Zero-order generation
        >>> new_prompt = ops.zero_order_generation("Answer questions")
    """

    def __init__(
        self,
        llm_client,  # LLMClient
        sentence_encoder=None,  # SentenceTransformer (optional)
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Инициализация операторов.

        Args:
            llm_client: LLM клиент для генерации
            sentence_encoder: Encoder для similarity (нужен для eda_mutation)
            model: Модель для генерации
            temperature: Температура генерации
        """
        self.llm_client = llm_client
        self.sentence_encoder = sentence_encoder
        self.model = model
        self.temperature = temperature
        # Для генеративных задач (CommonGen, XSum) сохраняем CoT scaffolding.
        # Устанавливается в rider.py на основе dataset_name.
        self.preserve_cot = False

        logger.info(
            f"EvolutionaryOperators initialized with model={model}, T={temperature}"
        )

    # ========== Task Descriptions ==========

    def get_task_description(self, dataset_name: str) -> str:
        """
        Возвращает описание задачи для датасета.

        Args:
            dataset_name: Название датасета

        Returns:
            Текстовое описание задачи
        """
        descriptions = {
            'GSM8K': 'Solve grade school math word problems step by step',
            'AG_News': 'Classify news articles into categories: World, Sports, Business, or Sci/Tech',
            'SQuAD_2': 'Answer questions based on context, or identify if question is impossible to answer',
            'CommonGen': 'Generate a coherent sentence using given concepts',
            'XSum': 'Given a news article, produce a single sentence that captures its essence. You may summarize, extract the lead, or complete the opening',
            'CodeSearchNet': 'Write a concise docstring for a given Python function'
        }
        return descriptions.get(dataset_name, 'Solve the task')

    # ========== GA Operators ==========

    def ga_mutation(
        self,
        parent: Prompt,
        task_desc: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Prompt:
        """
        Генетическая мутация промпта.

        Улучшает существующий промпт через LLM.

        Args:
            parent: Родительский промпт
            task_desc: Описание задачи
            temperature: Температура (если None, используется self.temperature)
            top_p: Top-p sampling (опционально)

        Returns:
            Мутированный промпт
        """
        temp = temperature if temperature is not None else self.temperature

        mutation_prompt = f"""Improve this instruction prompt for the task:

Task: {task_desc}
Current prompt: {parent.text}

Generate an improved version that is clearer and more effective.
Be specific. Add constraints, edge cases, and formatting rules.
Aim for 50-100 words.

Wrap your answer in <prompt></prompt> tags.

<prompt>"""

        try:
            response = self.llm_client.generate(
                prompt=mutation_prompt,
                model=self.model,
                temperature=temp,
                max_tokens=350,
                top_p=top_p or 1.0
            )

            offspring_text = extract_prompt_from_response(response, self.preserve_cot)

            offspring = Prompt(
                text=offspring_text,
                generation=parent.generation + 1,
                parent_ids=[parent.id],
                mutation_type="ga_mutation"
            )

            logger.debug(
                f"GA mutation: {parent.id} → {offspring.id} "
                f"(len={len(offspring_text)})"
            )

            return offspring

        except Exception as e:
            logger.error(f"GA mutation error: {e}")
            return parent  # Fallback

    # ========== DE Operators ==========

    def de_mutation(
        self,
        base: Prompt,
        donor1: Prompt,
        donor2: Prompt,
        best: Prompt,
        task_desc: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Prompt:
        """
        Дифференциальная эволюция: DE/current-to-best/1.

        Использует разность между двумя donor промптами и лучший промпт.

        Args:
            base: Базовый промпт
            donor1: Первый donor промпт
            donor2: Второй donor промпт
            best: Лучший промпт в популяции
            task_desc: Описание задачи
            temperature: Температура
            top_p: Top-p sampling

        Returns:
            Новый промпт
        """
        temp = temperature if temperature is not None else self.temperature

        de_prompt = f"""Analyze these prompts for the task and create an improved version:

Task: {task_desc}

Current best performer: "{best.text}"
Base prompt: "{base.text}"
Donor A: "{donor1.text}"
Donor B: "{donor2.text}"

Instructions:
1. Identify what makes the best performer effective
2. Find key differences between Donor A and B
3. Apply insights to improve the base prompt
4. Incorporate successful patterns from best performer

Create improved prompt (aim for 50-100 words). Include edge cases and output format rules.

Wrap your answer in <prompt></prompt> tags.

<prompt>"""

        try:
            response = self.llm_client.generate(
                prompt=de_prompt,
                model=self.model,
                temperature=temp,
                max_tokens=350,
                top_p=top_p or 1.0
            )

            offspring_text = extract_prompt_from_response(response, self.preserve_cot)

            offspring = Prompt(
                text=offspring_text,
                generation=base.generation + 1,
                parent_ids=[base.id, donor1.id, donor2.id, best.id],
                mutation_type="de_mutation"
            )

            logger.debug(
                f"DE mutation: base={base.id}, donors=({donor1.id},{donor2.id}), "
                f"best={best.id} → {offspring.id}"
            )

            return offspring

        except Exception as e:
            logger.error(f"DE mutation error: {e}")
            return base  # Fallback

    # ========== Zero-Order Generation ==========

    # Task-specific fallback промпты для валидации zero_order output.
    # Используются когда LLM генерирует мусор ("Prompts:", пустые строки и т.д.)
    TASK_FALLBACK_PROMPTS = {
        'SQuAD_2': "Extract the answer from the context. If unanswerable, output nothing.",
        'CommonGen': "Write one short sentence using all given words.",
        'XSum': "Complete the news summary by providing a concise headline or lead sentence that contextualizes the article.",
        'GSM8K': "Solve step by step. Output only the final number.",
        'AG_News': "Classify into: World, Sports, Business, or Sci/Tech.",
        'CodeSearchNet': "Read the Python function and write a concise docstring describing what it does. Include parameter descriptions and return value."
    }

    def _validate_prompt_text(self, text: str) -> bool:
        """
        Проверяет что сгенерированный промпт не дегенеративный.

        На grok zero_order_generation() создавала 12 копий "Prompts:" вместо
        реальных промптов. Начальная популяция = мусор с fitness ~0.05-0.17.

        Returns:
            True если промпт валиден, False если дегенеративный
        """
        if not text or len(text.strip()) < 20:
            return False
        # Grok деgenerate pattern: ответ содержит "Prompts:" вместо реального промпта
        if text.strip().lower() in ('prompts:', 'prompt:', 'prompts', 'prompt'):
            return False
        if text.strip().startswith('Prompts:') and len(text.strip()) < 30:
            return False
        return True

    def zero_order_generation(
        self,
        task_desc: str,
        num_examples: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        dataset_name: Optional[str] = None
    ) -> Prompt:
        """
        Генерация промпта с нуля на основе описания задачи.

        Используется для инициализации популяции.

        Args:
            task_desc: Описание задачи
            num_examples: Количество примеров для генерации
            temperature: Температура
            top_p: Top-p sampling
            dataset_name: Название датасета (для fallback промптов)

        Returns:
            Новый промпт
        """
        temp = temperature if temperature is not None else self.temperature

        # Structured diversity guidance вместо "be creative and diverse"
        approaches = [
            "step-by-step structured reasoning",
            "direct and concise instruction",
            "constraint-based (specify what NOT to do)"
        ]
        # Randomize template selection
        approach = random.choice(approaches)

        # Увеличен лимит с 40 до 100 слов.
        # Победивший промпт PromptBreeder на SQuAD_2 = 72 слова с детальными правилами.
        # RIDER не мог сгенерировать такие детальные промпты при лимите 40 слов.
        zero_order_prompt = f"""Generate {num_examples} fundamentally different instruction prompts for this task:

Task: {task_desc}

Requirements:
- Each prompt should guide a language model to solve this task effectively
- Use different strategies: one with {approach}, others with alternative approaches
- Each prompt maximum 100 words
- Be specific: include edge cases, output format rules, and constraints
- Ensure prompts are structurally different from each other

Prompts:
1."""

        # Retry с валидацией — защита от дегенеративной инициализации на grok.
        # На grok zero_order генерировал "Prompts:" × 12 вместо реальных промптов.
        max_retries = 3
        for retry in range(max_retries):
            try:
                retry_temp = temp + retry * 0.1  # Увеличиваем T при retry
                response = self.llm_client.generate(
                    prompt=zero_order_prompt,
                    model=self.model,
                    temperature=retry_temp,
                    max_tokens=500,
                    top_p=top_p or 1.0
                )

                # Парсим все промпты из нумерованного ответа и выбираем лучший (самый длинный валидный).
                # LLM отвечает форматом "1. ...\n2. ...\n3. ..." — раньше брали только первую строку.
                lines = response.strip().split('\n')
                candidates = []
                for line in lines:
                    cleaned = re.sub(r'^\d+\.\s*', '', line).strip().strip('"\'')
                    if self._validate_prompt_text(cleaned):
                        candidates.append(cleaned)

                if candidates:
                    # Берём самый длинный валидный промпт — он обычно самый детальный
                    prompt_text = max(candidates, key=len)
                    offspring = Prompt(
                        text=prompt_text,
                        generation=0,
                        parent_ids=[],
                        mutation_type="zero_order"
                    )
                    logger.debug(f"Zero-order generation: → {offspring.id} (len={len(prompt_text)}, candidates={len(candidates)})")
                    return offspring
                else:
                    # BUG FIX: avoid UnboundLocalError — prompt_text is undefined
                    # when candidates is empty. Log first raw line instead for diagnostics.
                    first_line = lines[0] if lines else ''
                    logger.warning(
                        f"Zero-order validation failed (retry {retry+1}/{max_retries}): "
                        f"'{first_line[:50]}...'"
                    )

            except Exception as e:
                logger.error(f"Zero-order generation error (retry {retry+1}): {e}")

        # Fallback на task-specific template если все retry провалились
        fallback_text = self.TASK_FALLBACK_PROMPTS.get(
            dataset_name, f"Solve this task: {task_desc}"
        )
        logger.warning(
            f"Zero-order: all {max_retries} retries failed, using fallback: '{fallback_text[:50]}'"
        )
        return Prompt(
            text=fallback_text,
            mutation_type="zero_order_fallback",
            generation=0
        )

    # ========== EDA Mutation ==========

    def eda_mutation(
        self,
        population: List[Prompt],
        task_desc: str,
        similarity_threshold: float = 0.95,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Prompt:
        """
        EDA (Estimation of Distribution Algorithm) мутация.

        Использует распределение успешных промптов в популяции.

        Args:
            population: Текущая популяция
            task_desc: Описание задачи
            similarity_threshold: Порог similarity для фильтрации
            temperature: Температура
            top_p: Top-p sampling

        Returns:
            Новый промпт на основе распределения
        """
        temp = temperature if temperature is not None else self.temperature

        # Фильтруем слишком похожие промпты
        if self.sentence_encoder is not None:
            filtered = self._filter_by_similarity(population, similarity_threshold)
        else:
            filtered = population[:5]  # Fallback без encoder

        if len(filtered) < 2:
            filtered = population[:5]  # Fallback

        prompts_text = '\n'.join([f"{i+1}. {p.text}" for i, p in enumerate(filtered[:5])])

        eda_prompt = f"""Based on these successful prompts for the task, generate a new variant:

Task: {task_desc}

Successful prompts:
{prompts_text}

Generate a new prompt that captures common patterns but with fresh phrasing.
Be specific and detailed. Include edge cases and output format rules.
Aim for 50-100 words.

Wrap your answer in <prompt></prompt> tags.

<prompt>"""

        try:
            response = self.llm_client.generate(
                prompt=eda_prompt,
                model=self.model,
                temperature=temp,
                max_tokens=350,
                top_p=top_p or 1.0
            )

            offspring_text = extract_prompt_from_response(response, self.preserve_cot)

            offspring = Prompt(
                text=offspring_text,
                generation=max(p.generation for p in filtered) + 1,
                parent_ids=[p.id for p in filtered[:3]],
                mutation_type="eda_mutation"
            )

            logger.debug(
                f"EDA mutation: {len(filtered)} prompts → {offspring.id} "
                f"(len={len(offspring_text)})"
            )

            return offspring

        except Exception as e:
            logger.error(f"EDA mutation error: {e}")
            return random.choice(population)  # Fallback

    def eda_rank_index(
        self,
        population: List[Prompt],
        task_desc: str,
        top_k: int = 5,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Prompt:
        """
        EDA с ранжированием по fitness.

        Использует только топ-K промптов по fitness.

        Args:
            population: Текущая популяция
            task_desc: Описание задачи
            top_k: Количество лучших промптов для использования
            temperature: Температура
            top_p: Top-p sampling

        Returns:
            Новый промпт на основе top-K
        """
        temp = temperature if temperature is not None else self.temperature

        # Сортируем по fitness
        sorted_pop = sorted(population, key=lambda p: p.fitness, reverse=True)
        top_prompts = sorted_pop[:top_k]

        prompts_text = '\n'.join([
            f"{i+1}. (fitness={p.fitness:.3f}) {p.text}"
            for i, p in enumerate(top_prompts)
        ])

        eda_rank_prompt = f"""Based on these top-performing prompts for the task, generate a new variant:

Task: {task_desc}

Top prompts (ranked by performance):
{prompts_text}

Generate a new prompt that incorporates patterns from high-performers.
Be specific and detailed. Include edge cases and output format rules.
Aim for 50-100 words.

Wrap your answer in <prompt></prompt> tags.

<prompt>"""

        try:
            response = self.llm_client.generate(
                prompt=eda_rank_prompt,
                model=self.model,
                temperature=temp,
                max_tokens=350,
                top_p=top_p or 1.0
            )

            offspring_text = extract_prompt_from_response(response, self.preserve_cot)

            offspring = Prompt(
                text=offspring_text,
                generation=max(p.generation for p in top_prompts) + 1,
                parent_ids=[p.id for p in top_prompts[:3]],
                mutation_type="eda_rank_index"
            )

            logger.debug(
                f"EDA rank-index: top-{top_k} → {offspring.id} "
                f"(fitness_range=[{top_prompts[-1].fitness:.3f}, {top_prompts[0].fitness:.3f}])"
            )

            return offspring

        except Exception as e:
            logger.error(f"EDA rank-index error: {e}")
            return random.choice(top_prompts)  # Fallback

    # ========== Additional Operators ==========

    # ========== Helper Methods ==========

    def _filter_by_similarity(
        self,
        prompts: List[Prompt],
        threshold: float
    ) -> List[Prompt]:
        """
        Фильтрует промпты по similarity threshold.

        Использует жадный алгоритм для отбора разнообразных промптов.

        Args:
            prompts: Список промптов
            threshold: Порог cosine similarity

        Returns:
            Отфильтрованный список разнообразных промптов
        """
        if not prompts or self.sentence_encoder is None:
            return prompts

        try:
            # Вычисляем embeddings
            texts = [p.text for p in prompts]
            embeddings = self.sentence_encoder.encode(texts, show_progress_bar=False)

            # Жадный отбор непохожих
            selected = [prompts[0]]
            selected_embeddings = [embeddings[0]]

            for i, (prompt, emb) in enumerate(zip(prompts[1:], embeddings[1:]), 1):
                # Проверяем similarity со всеми выбранными
                max_sim = max(
                    np.dot(emb, sel_emb) / (np.linalg.norm(emb) * np.linalg.norm(sel_emb) + 1e-8)
                    for sel_emb in selected_embeddings
                )

                if max_sim < threshold:
                    selected.append(prompt)
                    selected_embeddings.append(emb)

            logger.debug(
                f"Filtered by similarity: {len(prompts)} → {len(selected)} "
                f"(threshold={threshold})"
            )

            return selected if selected else prompts[:1]

        except Exception as e:
            logger.error(f"Similarity filtering error: {e}")
            return prompts  # Fallback

    def __repr__(self) -> str:
        """Строковое представление для отладки."""
        return (
            f"EvolutionaryOperators("
            f"model={self.model}, "
            f"T={self.temperature}, "
            f"has_encoder={self.sentence_encoder is not None})"
        )
