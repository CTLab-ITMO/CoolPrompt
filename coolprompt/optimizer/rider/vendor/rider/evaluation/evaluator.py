"""
PromptEvaluator для оценки промптов на датасетах.

Этот модуль реализует систему оценки промптов:
- Оценка на 5 датасетах (GSM8K, AG_News, SQuAD_2, CommonGen, XSum)
- Кэширование результатов для эффективности
- Greedy decoding (temperature=0.0) для стабильности
- Детальная оценка с возвратом всех метрик
- Batch evaluation для ускорения

Каждый датасет использует свою основную метрику для fitness:
- GSM8K: exact_match (EM)
- AG_News: f1_macro
- SQuAD_2: f1
- CommonGen: bert_score_f1
- XSum: bert_score_f1
"""

from typing import List, Dict, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from rider.core.prompts import Prompt
from rider.evaluation.metrics import MetricsEvaluator

logger = logging.getLogger(__name__)


class PromptEvaluator:
    """
    Evaluator для оценки промптов на датасетах.

    Оценивает промпты, применяя их к данным и вычисляя метрики.
    Кэширует результаты для избежания повторных LLM вызовов.

    Args:
        llm_client: LLM клиент для генерации ответов
        metrics_evaluator: MetricsEvaluator для вычисления метрик
        model: Модель для оценки (default: "gpt-3.5-turbo")
        temperature: Температура для evaluation (default: 0.0 - greedy)
        max_cache_size: Максимальный размер кэша (default: 10000)

    Example:
        >>> from rider.llm.client import LLMClient
        >>> from rider.evaluation.metrics import MetricsEvaluator
        >>>
        >>> llm = LLMClient()
        >>> metrics = MetricsEvaluator()
        >>> evaluator = PromptEvaluator(llm, metrics, model="gpt-4")
        >>>
        >>> prompt = Prompt(text="Solve this math problem step by step:")
        >>> data = [{'question': '2+2', 'answer': '4'}]
        >>> fitness = evaluator.evaluate_prompt(prompt, 'GSM8K', data)
        >>> print(f"Fitness: {fitness}")
    """

    def __init__(
        self,
        llm_client,  # LLMClient
        metrics_evaluator: MetricsEvaluator,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_cache_size: int = 10000,
        max_workers: int = 32,
        use_prompt_repetition: bool = True
    ):
        """
        Инициализация evaluator.

        Args:
            llm_client: LLM клиент
            metrics_evaluator: Метрики для оценки
            model: Модель для генерации
            temperature: Температура (0.0 для greedy decoding)
            max_cache_size: Размер кэша
            max_workers: Количество параллельных потоков для LLM вызовов
            use_prompt_repetition: Повторять инструкцию дважды (Google paper).
                True для RIDER, False для бейзлайнов.
        """
        self.llm_client = llm_client
        self.metrics_evaluator = metrics_evaluator
        self.model = model
        self.temperature = temperature
        self.max_cache_size = max_cache_size
        self.max_workers = max_workers
        self.use_prompt_repetition = use_prompt_repetition

        # Кэш: {(prompt_text, dataset_name, data_hash): fitness}
        self.cache: Dict[str, float] = {}

        # ИСПРАВЛЕНО: Добавлен отдельный кэш для detailed evaluations
        # {(prompt_text, dataset_name, data_hash): detailed_results}
        self.detailed_cache: Dict[str, Dict] = {}

        # Статистика кэша
        self.cache_hits = 0
        self.cache_misses = 0
        self.detailed_cache_hits = 0
        self.detailed_cache_misses = 0

        logger.info(
            f"PromptEvaluator initialized with model={model}, T={temperature}, "
            f"max_cache={max_cache_size}, max_workers={max_workers}, "
            f"prompt_repetition={use_prompt_repetition}"
        )

    # ========== Main Evaluation Methods ==========

    def evaluate_prompt(
        self,
        prompt: Prompt,
        dataset_name: str,
        data: List[Dict],
        use_cache: bool = True,
        show_progress: bool = False
    ) -> float:
        """
        Оценивает промпт на датасете.

        Возвращает fitness score (основную метрику для датасета).

        Args:
            prompt: Промпт для оценки
            dataset_name: Название датасета
            data: Данные для оценки
            use_cache: Использовать кэш
            show_progress: Показывать прогресс

        Returns:
            Fitness score (основная метрика)
        """
        # Проверяем кэш (включая few_shot_examples для CHIMERA)
        cache_key = self._get_cache_key(
            prompt.text, dataset_name, len(data),
            few_shot_examples=getattr(prompt, 'few_shot_examples', None)
        )
        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            fitness = self.cache[cache_key]
            logger.debug(
                f"Cache hit for {dataset_name}: {prompt.id} → fitness={fitness:.4f}"
            )
            # НЕ устанавливаем prompt.fitness — это побочный эффект,
            # который corrupts population fitness при вызове из _nexus_classify_examples.
            # Вызывающий код должен явно установить prompt.fitness = fitness.
            return fitness

        self.cache_misses += 1

        # Выбираем метод оценки
        if dataset_name == 'GSM8K':
            fitness = self._evaluate_gsm8k(prompt, data, show_progress)
        elif dataset_name == 'AG_News':
            fitness = self._evaluate_ag_news(prompt, data, show_progress)
        elif dataset_name == 'SQuAD_2':
            fitness = self._evaluate_squad2(prompt, data, show_progress)
        elif dataset_name == 'CommonGen':
            fitness = self._evaluate_commongen(prompt, data, show_progress)
        elif dataset_name == 'XSum':
            fitness = self._evaluate_xsum(prompt, data, show_progress)
        elif dataset_name == 'CodeSearchNet':
            fitness = self._evaluate_codesearchnet(prompt, data, show_progress)
        elif dataset_name == 'HotpotQA':
            fitness = self._evaluate_hotpotqa(prompt, data, show_progress)
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            fitness = 0.0

        # Сохраняем в кэш
        if use_cache:
            self._add_to_cache(cache_key, fitness)

        # НЕ устанавливаем prompt.fitness здесь — это побочный эффект,
        # который перетирал val fitness при тестовой оценке бейзлайнов.
        # Вызывающий код должен явно установить prompt.fitness = fitness.

        logger.debug(
            f"Evaluated {dataset_name}: {prompt.id} → fitness={fitness:.4f}"
        )

        return fitness

    def evaluate_with_details(
        self,
        prompt: Prompt,
        dataset_name: str,
        data: List[Dict],
        show_progress: bool = False
    ) -> Dict:
        """
        Детальная оценка с возвратом всех метрик и predictions.

        ИСПРАВЛЕНО: Добавлено кэширование для избежания двойной оценки.

        Args:
            prompt: Промпт для оценки
            dataset_name: Название датасета
            data: Данные для оценки
            show_progress: Показывать прогресс

        Returns:
            Словарь с метриками, predictions и ground_truth
        """
        # ИСПРАВЛЕНО: Проверить кэш detailed evaluations (включая few_shot_examples для CHIMERA)
        cache_key = self._get_cache_key(
            prompt.text, dataset_name, len(data),
            few_shot_examples=getattr(prompt, 'few_shot_examples', None)
        )

        if cache_key in self.detailed_cache:
            self.detailed_cache_hits += 1
            logger.debug(
                f"Detailed cache HIT for {dataset_name} (size={len(data)}). "
                f"Hit rate: {self.detailed_cache_hits / (self.detailed_cache_hits + self.detailed_cache_misses):.2%}"
            )
            return self.detailed_cache[cache_key]

        self.detailed_cache_misses += 1

        # Вычислить результаты
        predictions, ground_truth = self._get_predictions(
            prompt, dataset_name, data, show_progress
        )

        # Вычисляем метрики (передаём data для multi-ref BERTScore в CommonGen)
        metrics = self.metrics_evaluator.evaluate(dataset_name, predictions, ground_truth, data=data)

        result = {
            'metrics': metrics,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'dataset_name': dataset_name,
            'prompt_id': prompt.id
        }

        # ИСПРАВЛЕНО: Сохранить в detailed cache
        self._add_to_detailed_cache(cache_key, result)

        return result

    # ========== Parallel LLM Helper ==========

    def _parallel_generate(
        self,
        prompts_and_tokens: List[Tuple[str, int]],
        dataset_name: str,
        show_progress: bool = False
    ) -> List[str]:
        """
        Параллельная генерация ответов LLM для списка промптов.

        Отправляет все запросы одновременно через ThreadPoolExecutor,
        сохраняя порядок результатов.

        Args:
            prompts_and_tokens: Список (full_prompt, max_tokens)
            dataset_name: Название датасета (для логов)
            show_progress: Показывать прогресс

        Returns:
            Список ответов в том же порядке
        """
        results = [""] * len(prompts_and_tokens)

        def _generate_one(idx_prompt_tokens):
            idx, (full_prompt, max_tokens) = idx_prompt_tokens
            try:
                response = self.llm_client.generate(
                    prompt=full_prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                return idx, (response or "").strip()
            except Exception as e:
                logger.error(f"{dataset_name} evaluation error (item {idx}): {e}")
                return idx, ""

        workers = min(self.max_workers, len(prompts_and_tokens))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_generate_one, (i, pt)): i
                for i, pt in enumerate(prompts_and_tokens)
            }
            if show_progress:
                pbar = tqdm(total=len(futures), desc=f"{dataset_name} eval")
            for future in as_completed(futures):
                idx, response = future.result()
                results[idx] = response
                if show_progress:
                    pbar.update(1)
            if show_progress:
                pbar.close()

        return results

    # ========== Prompt Construction Helper ==========

    def _build_instruction_prefix(self, prompt: Prompt, dataset_name: str = None) -> str:
        """Строит префикс инструкции с учётом prompt repetition.

        Prompt Repetition (Google arxiv.org/pdf/2512.14982) —
        повтор инструкции дважды для non-reasoning моделей.
        Для бейзлайнов (use_prompt_repetition=False) инструкция не повторяется.

        ECHO enabled for ALL task types EXCEPT f1_macro (classification).
        bert_score_f1 generative tasks (XSum, CommonGen) now get prompt repetition.
        f1_macro (AG_News) stays disabled — intentional metric-specific decision.
        """
        if not self.use_prompt_repetition:
            return prompt.text
        # disable ECHO for classification (f1_macro) — hurts AG_News
        if dataset_name:
            metric = self.metrics_evaluator.get_primary_metric_name(dataset_name)
            if metric == 'f1_macro':
                return prompt.text  # No repetition for classification
        # ECHO PROTOCOL: повторяем инструкцию для all other tasks (exact_match, f1, bert_score_f1).
        # Google arxiv 2512.14982: 47 wins / 0 losses на non-reasoning моделях.
        return f"{prompt.text}\n\n{prompt.text}"

    def _build_few_shot_block(self, prompt: Prompt, dataset_name: str) -> str:
        """CHIMERA ENGINE: строит блок few-shot примеров из эволюционирующих примеров промпта."""
        if not hasattr(prompt, 'few_shot_examples') or not prompt.few_shot_examples:
            return ""
        if dataset_name == 'CommonGen':
            ex_strs = []
            for ex in prompt.few_shot_examples[:3]:
                concepts = ', '.join(ex.get('concepts', []))
                target = ex.get('target', '')
                ex_strs.append(f"Concepts: {concepts}\nSentence: {target}")
            return "\n\nExamples:\n" + "\n\n".join(ex_strs) + "\n"
        elif dataset_name == 'XSum':
            ex_strs = []
            for ex in prompt.few_shot_examples[:2]:
                doc = ex.get('document', '')[:300]
                summary = ex.get('summary', '')
                ex_strs.append(f"Document: {doc}...\nSummary: {summary}")
            return "\n\nExamples:\n" + "\n\n".join(ex_strs) + "\n"
        elif dataset_name == 'CodeSearchNet':
            ex_strs = []
            for ex in prompt.few_shot_examples[:2]:
                code = ex.get('code', '')[:500]
                docstring = ex.get('docstring', '')
                ex_strs.append(f"Code:\n{code}\nDocstring: {docstring}")
            return "\n\nExamples:\n" + "\n\n".join(ex_strs) + "\n"
        return ""

    # ========== Dataset-Specific Evaluation ==========

    def _evaluate_gsm8k(
        self,
        prompt: Prompt,
        data: List[Dict],
        show_progress: bool = False
    ) -> float:
        """Оценка на GSM8K. МЕТРИКА: EM (Exact Match)"""
        ground_truth = [item['answer'] for item in data]

        instruction = self._build_instruction_prefix(prompt, dataset_name='GSM8K')
        prompts_and_tokens = []
        for item in data:
            full_prompt = f"{instruction}\n\nQuestion: {item['question']}\n\nAnswer:"
            prompts_and_tokens.append((full_prompt, 512))

        predictions = self._parallel_generate(prompts_and_tokens, "GSM8K", show_progress)
        metrics = self.metrics_evaluator.evaluate_gsm8k(predictions, ground_truth)
        return metrics['exact_match']

    def _evaluate_ag_news(
        self,
        prompt: Prompt,
        data: List[Dict],
        show_progress: bool = False
    ) -> float:
        """Оценка на AG News. МЕТРИКА: F1-macro"""
        ground_truth = [item['label'] for item in data]

        instruction = self._build_instruction_prefix(prompt, dataset_name='AG_News')
        # Concise suffix для consistency с _get_predictions
        concise_suffix = "\nRespond with ONLY the answer, nothing else.\nRespond with ONLY the answer, nothing else."
        prompts_and_tokens = []
        for item in data:
            full_prompt = f"{instruction}\n\nText: {item['text']}\n\nCategory:{concise_suffix}"
            prompts_and_tokens.append((full_prompt, 50))

        predictions = self._parallel_generate(prompts_and_tokens, "AG_News", show_progress)
        metrics = self.metrics_evaluator.evaluate_ag_news(predictions, ground_truth)
        return metrics['f1_macro']

    def _evaluate_squad2(
        self,
        prompt: Prompt,
        data: List[Dict],
        show_progress: bool = False
    ) -> float:
        """Оценка на SQuAD 2.0. МЕТРИКА: F1"""
        ground_truth = [
            {'answers': item['answers'], 'is_impossible': item['is_impossible']}
            for item in data
        ]

        instruction = self._build_instruction_prefix(prompt, dataset_name='SQuAD_2')
        concise_suffix = "\nRespond with ONLY the answer, nothing else.\nRespond with ONLY the answer, nothing else."
        prompts_and_tokens = []
        for item in data:
            full_prompt = f"{instruction}\n\nContext: {item['context']}\n\nQuestion: {item['question']}\n\nAnswer:{concise_suffix}"
            prompts_and_tokens.append((full_prompt, 100))

        predictions = self._parallel_generate(prompts_and_tokens, "SQuAD2", show_progress)
        metrics = self.metrics_evaluator.evaluate_squad2(predictions, ground_truth)
        return metrics['f1']

    def _evaluate_hotpotqa(
        self,
        prompt: Prompt,
        data: List[Dict],
        show_progress: bool = False
    ) -> float:
        """Оценка на HotpotQA (Multi-hop QA). МЕТРИКА: F1"""
        ground_truth = [item['answer'] for item in data]

        instruction = self._build_instruction_prefix(prompt, dataset_name='HotpotQA')
        concise_suffix = "\nRespond with ONLY the answer, nothing else.\nRespond with ONLY the answer, nothing else."
        prompts_and_tokens = []
        for item in data:
            full_prompt = f"{instruction}\n\nContext: {item['context'][:2000]}\n\nQuestion: {item['question']}\n\nAnswer:{concise_suffix}"
            prompts_and_tokens.append((full_prompt, 100))

        predictions = self._parallel_generate(prompts_and_tokens, "HotpotQA", show_progress)
        metrics = self.metrics_evaluator.evaluate_hotpotqa(predictions, ground_truth)
        return metrics['f1']

    def _evaluate_commongen(
        self,
        prompt: Prompt,
        data: List[Dict],
        show_progress: bool = False
    ) -> float:
        """Оценка на CommonGen. МЕТРИКА: BERTScore F1

        Multi-reference BERTScore — передаём data с all_targets
        для multi-ref оценки (max score среди всех references).
        """
        ground_truth = [item['target'] for item in data]

        instruction = self._build_instruction_prefix(prompt, dataset_name='CommonGen')
        examples_block = self._build_few_shot_block(prompt, 'CommonGen')
        concise_suffix = "\nRespond with ONLY the answer, nothing else.\nRespond with ONLY the answer, nothing else."
        prompts_and_tokens = []
        for item in data:
            concepts_str = ', '.join(item['concepts'])
            full_prompt = f"{instruction}{examples_block}\n\nConcepts: {concepts_str}\n\nSentence:{concise_suffix}"
            prompts_and_tokens.append((full_prompt, 100))

        predictions = self._parallel_generate(prompts_and_tokens, "CommonGen", show_progress)
        metrics = self.metrics_evaluator.evaluate_commongen(predictions, ground_truth, data=data)
        return metrics['bert_score_f1']

    @staticmethod
    def _clean_generation_output(text: str, dataset_name: str = "") -> str:
        """Post-process generated output: strip noise that hurts BERTScore."""
        t = text.strip()
        # Remove common LLM preambles
        for prefix in ["Summary:", "Here is the summary:", "The summary is:",
                        "Here's the summary:", "Output:", "Answer:",
                        "Docstring:", "Description:", "Here is"]:
            if t.lower().startswith(prefix.lower()):
                t = t[len(prefix):].strip()
        # Remove wrapping quotes
        if len(t) > 2 and t[0] == t[-1] and t[0] in ('"', "'", '\u201c', '\u201d'):
            t = t[1:-1].strip()
        # XSum: keep only first sentence (references are single-sentence)
        if dataset_name == 'XSum' and '.' in t:
            first_sent = t.split('.')[0].strip() + '.'
            if len(first_sent) > 20:  # Sanity check
                t = first_sent
        return t

    def _evaluate_xsum(
        self,
        prompt: Prompt,
        data: List[Dict],
        show_progress: bool = False
    ) -> float:
        """Оценка на XSum. МЕТРИКА: BERTScore F1"""
        ground_truth = [item['summary'] for item in data]

        instruction = self._build_instruction_prefix(prompt, dataset_name='XSum')
        examples_block = self._build_few_shot_block(prompt, 'XSum')
        # XSum requires single-sentence summaries — shorter max_tokens + explicit constraint
        concise_suffix = "\nWrite exactly ONE sentence. No preamble, no explanation.\nWrite exactly ONE sentence. No preamble, no explanation."
        prompts_and_tokens = []
        for item in data:
            doc = item['document'][:2000]
            full_prompt = f"{instruction}{examples_block}\n\nDocument: {doc}\n\nSummary:{concise_suffix}"
            prompts_and_tokens.append((full_prompt, 60))

        predictions = self._parallel_generate(prompts_and_tokens, "XSum", show_progress)
        # Post-process: strip preamble, keep first sentence
        predictions = [self._clean_generation_output(p, "XSum") for p in predictions]
        metrics = self.metrics_evaluator.evaluate_xsum(predictions, ground_truth)
        return metrics['bert_score_f1']

    def _evaluate_codesearchnet(
        self,
        prompt: Prompt,
        data: List[Dict],
        show_progress: bool = False
    ) -> float:
        """Оценка на CodeSearchNet (code summarization). МЕТРИКА: BERTScore F1"""
        ground_truth = [item['docstring'] for item in data]

        instruction = self._build_instruction_prefix(prompt, dataset_name='CodeSearchNet')
        examples_block = self._build_few_shot_block(prompt, 'CodeSearchNet')
        concise_suffix = "\nRespond with ONLY the answer, nothing else.\nRespond with ONLY the answer, nothing else."
        prompts_and_tokens = []
        for item in data:
            code = item['code'][:1500]
            full_prompt = f"{instruction}{examples_block}\n\nCode:\n{code}\n\nDocstring:{concise_suffix}"
            prompts_and_tokens.append((full_prompt, 150))

        predictions = self._parallel_generate(prompts_and_tokens, "CodeSearchNet", show_progress)
        metrics = self.metrics_evaluator.evaluate_codesearchnet(predictions, ground_truth)
        return metrics['bert_score_f1']

    # ========== Helper Methods ==========

    def _get_predictions(
        self,
        prompt: Prompt,
        dataset_name: str,
        data: List[Dict],
        show_progress: bool = False
    ) -> Tuple[List[str], List]:
        """
        Получает predictions и ground_truth для датасета.
        Использует параллельные LLM вызовы.

        Args:
            prompt: Промпт
            dataset_name: Название датасета
            data: Данные
            show_progress: Показывать прогресс

        Returns:
            Tuple (predictions, ground_truth)
        """
        ground_truth = []
        prompts_and_tokens = []

        # Используем _build_instruction_prefix для учёта use_prompt_repetition
        repeated_instruction = self._build_instruction_prefix(prompt, dataset_name=dataset_name)

        # Универсальный суффикс против verbose ответов (haiku и др.)
        concise_suffix = "\nRespond with ONLY the answer, nothing else.\nRespond with ONLY the answer, nothing else."

        # CHIMERA ENGINE: few-shot block для генеративных задач
        few_shot_block = self._build_few_shot_block(prompt, dataset_name)

        for item in data:
            if dataset_name == 'GSM8K':
                # GSM8K: без concise_suffix — нужен chain-of-thought для математики
                full_prompt = f"{repeated_instruction}\n\nQuestion: {item['question']}\n\nAnswer:"
                ground_truth.append(item['answer'])
                max_tokens = 512
            elif dataset_name == 'AG_News':
                full_prompt = f"{repeated_instruction}\n\nText: {item['text']}\n\nCategory:{concise_suffix}"
                ground_truth.append(item['label'])
                max_tokens = 50
            elif dataset_name == 'SQuAD_2':
                full_prompt = f"{repeated_instruction}\n\nContext: {item['context']}\n\nQuestion: {item['question']}\n\nAnswer:{concise_suffix}"
                ground_truth.append({
                    'answers': item['answers'],
                    'is_impossible': item['is_impossible']
                })
                max_tokens = 100
            elif dataset_name == 'CommonGen':
                concepts_str = ', '.join(item['concepts'])
                full_prompt = f"{repeated_instruction}{few_shot_block}\n\nConcepts: {concepts_str}\n\nSentence:{concise_suffix}"
                ground_truth.append(item['target'])
                max_tokens = 100
            elif dataset_name == 'XSum':
                doc = item['document'][:2000]
                # BUG FIX: use XSum-specific suffix + reduced max_tokens
                # to match _evaluate_xsum() behavior. Previously detailed_path used generic
                # suffix and max_tokens=100, producing multi-sentence outputs that hurt BERTScore.
                xsum_suffix = "\nWrite exactly ONE sentence. No preamble, no explanation.\nWrite exactly ONE sentence. No preamble, no explanation."
                full_prompt = f"{repeated_instruction}{few_shot_block}\n\nDocument: {doc}\n\nSummary:{xsum_suffix}"
                ground_truth.append(item['summary'])
                max_tokens = 60
            elif dataset_name == 'CodeSearchNet':
                code = item['code'][:1500]
                full_prompt = f"{repeated_instruction}{few_shot_block}\n\nCode:\n{code}\n\nDocstring:{concise_suffix}"
                ground_truth.append(item['docstring'])
                max_tokens = 150
            elif dataset_name == 'HotpotQA':
                full_prompt = f"{repeated_instruction}\n\nContext: {item['context'][:2000]}\n\nQuestion: {item['question']}\n\nAnswer:{concise_suffix}"
                ground_truth.append(item['answer'])
                max_tokens = 100
            else:
                logger.error(f"Unknown dataset: {dataset_name}")
                continue

            prompts_and_tokens.append((full_prompt, max_tokens))

        predictions = self._parallel_generate(prompts_and_tokens, dataset_name, show_progress)
        # BUG FIX: apply generation cleanup for XSum in detailed path too,
        # so evaluate_with_details() matches _evaluate_xsum() fitness path.
        if dataset_name == 'XSum':
            predictions = [self._clean_generation_output(p, "XSum") for p in predictions]
        return predictions, ground_truth

    def _get_cache_key(self, prompt_text: str, dataset_name: str, data_size: int,
                       few_shot_examples=None) -> str:
        """
        Генерирует ключ кэша.

        BUG FIX: include few_shot_examples signature so CHIMERA offspring
        with same text but different few-shot examples get unique cache entries.
        Without this, CHIMERA swap/crossover outputs can collide with parent cache entries.

        Args:
            prompt_text: Текст промпта
            dataset_name: Название датасета
            data_size: Размер данных
            few_shot_examples: Few-shot примеры (для CHIMERA), может быть None

        Returns:
            Строковый ключ для кэша
        """
        fs_sig = ""
        if few_shot_examples:
            import hashlib
            fs_str = str([sorted(ex.items()) if isinstance(ex, dict) else ex
                          for ex in few_shot_examples])
            fs_sig = hashlib.md5(fs_str.encode('utf-8', errors='ignore')).hexdigest()[:8]
        return f"{prompt_text}|||{dataset_name}|||{data_size}|||{fs_sig}"

    def _add_to_cache(self, key: str, value: float) -> None:
        """
        Добавляет в кэш с проверкой размера.

        Args:
            key: Ключ
            value: Значение fitness
        """
        if len(self.cache) >= self.max_cache_size:
            # Удаляем случайный элемент если кэш переполнен
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = value

    def _add_to_detailed_cache(self, key: str, value: Dict) -> None:
        """
        Добавляет в detailed кэш с проверкой размера.

        ИСПРАВЛЕНО: Новый метод для кэширования detailed evaluations.

        Args:
            key: Ключ
            value: Словарь с detailed results
        """
        if len(self.detailed_cache) >= self.max_cache_size:
            # Удаляем случайный элемент если кэш переполнен
            self.detailed_cache.pop(next(iter(self.detailed_cache)))

        self.detailed_cache[key] = value

    # ========== Cache Management ==========

    def get_cache_stats(self) -> Dict:
        """
        Получить статистику кэша.

        ИСПРАВЛЕНО: Включает статистику обоих кэшей.

        Returns:
            Словарь со статистикой кэша
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        total_detailed = self.detailed_cache_hits + self.detailed_cache_misses
        detailed_hit_rate = self.detailed_cache_hits / total_detailed if total_detailed > 0 else 0

        return {
            'cache_size': len(self.cache),
            'detailed_cache_size': len(self.detailed_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'detailed_cache_hits': self.detailed_cache_hits,
            'detailed_cache_misses': self.detailed_cache_misses,
            'detailed_hit_rate': detailed_hit_rate
        }

    def __repr__(self) -> str:
        """Строковое представление для отладки."""
        stats = self.get_cache_stats()
        return (
            "PromptEvaluator("
            f"model={self.model}, "
            f"T={self.temperature}, "
            f"cache={stats['cache_size']}/{stats['max_cache_size']}, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )
