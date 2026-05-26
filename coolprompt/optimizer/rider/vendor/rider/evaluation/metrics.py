"""
Метрики оценки для всех датасетов.

Этот модуль реализует метрики для 5 датасетов:
1. GSM8K - Exact Match (EM) для математических задач
2. AG_News - F1-macro для классификации новостей
3. SQuAD 2.0 - EM и F1 для QA с невозможными вопросами
4. CommonGen - BERTScore F1 для генерации текста из концептов
5. XSum - BERTScore F1 + ROUGE для суммаризации

Каждый датасет имеет "основную метрику" используемую для fitness:
- GSM8K: exact_match
- AG_News: f1_macro
- SQuAD_2: f1
- CommonGen: bert_score_f1
- XSum: bert_score_f1
"""

import re
import threading
import numpy as np
from typing import List, Dict
from collections import Counter
import logging

from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Класс для вычисления метрик на всех датасетах.

    Инициализирует BERTScorer и ROUGEScorer при создании.
    Предоставляет методы для оценки predictions vs ground_truth
    для каждого датасета.

    Args:
        None

    Example:
        >>> evaluator = MetricsEvaluator()
        >>>
        >>> # GSM8K evaluation
        >>> predictions = ["42", "18", "100"]
        >>> ground_truth = ["42", "17", "100"]
        >>> metrics = evaluator.evaluate_gsm8k(predictions, ground_truth)
        >>> print(metrics['exact_match'])  # 0.667
        >>>
        >>> # AG_News evaluation
        >>> preds = ["World", "Sports", "Business"]
        >>> truths = ["World", "Sports", "World"]
        >>> metrics = evaluator.evaluate_ag_news(preds, truths)
        >>> print(metrics['f1_macro'])
    """

    def __init__(self):
        """Инициализация метрик. BERTScorer загружается лениво при первом использовании."""
        self._bert_scorer = None
        self._bert_lock = threading.Lock()

        # ROUGE scorer для суммаризации
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

        logger.info("MetricsEvaluator initialized with ROUGEScorer (BERTScorer lazy)")

    @property
    def bert_scorer(self):
        """Thread-safe ленивая инициализация BERTScorer."""
        if self._bert_scorer is None:
            with self._bert_lock:
                # Double-check after acquiring lock
                if self._bert_scorer is None:
                    import torch
                    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    # DeBERTa-xlarge-mnli: officially recommended by BERTScore authors
                    # Highest correlation with human judgments (0.778)
                    # Requires A100 40GB+ GPU (OOM on 24GB A5000)
                    # Init on CPU first, then move — avoids meta tensor error on GPU
                    scorer = BERTScorer(
                        model_type="microsoft/deberta-xlarge-mnli",
                        batch_size=8,
                        device='cpu'
                    )
                    if _device == 'cuda':
                        scorer._model = scorer._model.to(torch.device('cuda'))
                        scorer.device = 'cuda'
                    self._bert_scorer = scorer
                    logger.info(f"BERTScorer initialized on {_device}")
        return self._bert_scorer

    # ========== Helper Methods ==========

    def extract_number(self, text: str) -> str:
        """
        Извлекает числовой ответ из текста.

        Ищет последнее число в тексте (для GSM8K).

        Args:
            text: Текст с ответом

        Returns:
            Строка с числом или пустая строка
        """
        numbers = re.findall(r'-?(?:\d+\.?\d*|\.\d+)', text)
        if numbers:
            return numbers[-1]
        return ""

    def normalize_answer(self, text: str) -> str:
        """
        Нормализует текст для сравнения (SQuAD-2 official style).

        Приводит к lowercase, убирает артикли, пунктуацию и лишние пробелы.

        Args:
            text: Текст для нормализации

        Returns:
            Нормализованный текст
        """
        text = text.lower().strip()
        # Убираем пунктуацию
        text = re.sub(r'[^\w\s]', '', text)
        # Убираем артикли (SQuAD-2 official evaluation)
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        return text

    # ========== GSM8K Метрики ==========

    def evaluate_gsm8k(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Оценка для GSM8K (математические задачи).

        ОСНОВНАЯ МЕТРИКА: Exact Match (EM)

        Args:
            predictions: Список предсказаний
            ground_truth: Список правильных ответов

        Returns:
            Словарь с метриками:
            {
                'exact_match': float,  # ОСНОВНАЯ МЕТРИКА для fitness
                'accuracy': float,      # Алиас для exact_match
                'correct': int,
                'total': int
            }
        """
        correct = 0
        total = len(predictions)

        for pred, truth in zip(predictions, ground_truth):
            pred_num = self.extract_number(pred)
            truth_num = self.extract_number(truth)

            if pred_num and truth_num:
                try:
                    if abs(float(pred_num) - float(truth_num)) < 1e-6:
                        correct += 1
                except ValueError:
                    logger.warning(
                        f"GSM8K: Failed to parse numbers - pred={pred_num}, truth={truth_num}"
                    )

        exact_match = correct / total if total > 0 else 0.0

        logger.debug(
            f"GSM8K evaluation: {correct}/{total} correct (EM={exact_match:.3f})"
        )

        return {
            'exact_match': exact_match,  # ОСНОВНАЯ МЕТРИКА
            'accuracy': exact_match,      # Алиас
            'correct': correct,
            'total': total
        }

    # ========== AG News Метрики ==========

    def evaluate_ag_news(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Оценка для AG News (классификация новостей).

        ОСНОВНАЯ МЕТРИКА: F1-macro

        Args:
            predictions: Список предсказанных классов
            ground_truth: Список правильных классов

        Returns:
            Словарь с метриками:
            {
                'f1_macro': float,         # ОСНОВНАЯ МЕТРИКА
                'accuracy': float,
                'valid_predictions': int,
                'total': int
            }
        """
        # Нормализация классов
        label_mapping = {
            'world': 'World',
            'sports': 'Sports',
            'business': 'Business',
            'sci/tech': 'Sci/Tech',
            'technology': 'Sci/Tech',
            'science': 'Sci/Tech',
            'sci': 'Sci/Tech',
            'tech': 'Sci/Tech'
        }

        normalized_preds = []
        for pred in predictions:
            pred_lower = pred.lower().strip()
            # Убираем markdown форматирование: "**Sports**\n\nThis text..." → "sports\n\nthis text..."
            pred_lower = pred_lower.replace('**', '')
            # Убираем типичные префиксы от LLM: "Category: Business" → "business"
            for prefix in ['category:', 'label:', 'class:', 'answer:']:
                if pred_lower.startswith(prefix):
                    pred_lower = pred_lower[len(prefix):].strip()
                    break
            # Универсальный парсинг: берём первую строку, затем ищем категорию в тексте
            matched = label_mapping.get(pred_lower)
            if not matched:
                # Попробовать первую строку (до \n)
                first_line = pred_lower.split('\n')[0].strip().rstrip('.:,;')
                matched = label_mapping.get(first_line)
            if not matched:
                # Поиск категории в любом месте текста (приоритет: первое вхождение)
                for label_key, label_val in label_mapping.items():
                    # Ищем как отдельное слово (word boundary)
                    if re.search(r'\b' + re.escape(label_key) + r'\b', pred_lower):
                        matched = label_val
                        break
            normalized_preds.append(matched if matched else pred)

        # Фильтрация валидных предсказаний
        valid_labels = ['World', 'Sports', 'Business', 'Sci/Tech']
        valid_indices = [
            i for i, (p, t) in enumerate(zip(normalized_preds, ground_truth))
            if p in valid_labels
        ]

        if len(valid_indices) == 0:
            logger.warning("AG_News: No valid predictions")
            return {
                'f1_macro': 0.0,
                'accuracy': 0.0,
                'valid_predictions': 0,
                'total': len(predictions)
            }

        valid_preds = [normalized_preds[i] for i in valid_indices]
        valid_truth = [ground_truth[i] for i in valid_indices]

        # F1-macro (ОСНОВНАЯ МЕТРИКА)
        f1_macro = f1_score(
            valid_truth,
            valid_preds,
            labels=valid_labels,
            average='macro',
            zero_division=0
        )

        # Accuracy (дополнительная)
        accuracy = sum(p == t for p, t in zip(valid_preds, valid_truth)) / len(valid_preds)

        logger.debug(
            f"AG_News evaluation: F1-macro={f1_macro:.3f}, "
            f"Accuracy={accuracy:.3f}, valid={len(valid_indices)}/{len(predictions)}"
        )

        return {
            'f1_macro': f1_macro,         # ОСНОВНАЯ МЕТРИКА
            'accuracy': accuracy,
            'valid_predictions': len(valid_indices),
            'total': len(predictions)
        }

    # ========== SQuAD 2.0 Метрики ==========

    def evaluate_squad2(
        self,
        predictions: List[str],
        ground_truth: List[Dict]
    ) -> Dict[str, float]:
        """
        Оценка для SQuAD 2.0 (QA с невозможными вопросами).

        МЕТРИКИ: EM (Exact Match), F1

        Args:
            predictions: Список предсказанных ответов
            ground_truth: Список словарей с полями:
                - 'answers': List[str] - возможные правильные ответы
                - 'is_impossible': bool - вопрос невозможен

        Returns:
            Словарь с метриками:
            {
                'exact_match': float,
                'f1': float,           # ОСНОВНАЯ МЕТРИКА
                'accuracy': float
            }
        """
        exact_matches = 0
        f1_scores = []

        for pred, truth_dict in zip(predictions, ground_truth):
            truth_answers = truth_dict.get('answers', [])
            is_impossible = truth_dict.get('is_impossible', len(truth_answers) == 0)

            pred_normalized = self.normalize_answer(pred)

            # Проверка на impossible вопрос
            if is_impossible:
                if (pred_normalized == '' or
                    'impossible' in pred_normalized or
                    'cannot' in pred_normalized or
                    'no answer' in pred_normalized):
                    exact_matches += 1
                    f1_scores.append(1.0)
                else:
                    f1_scores.append(0.0)
            else:
                # Обычная проверка с несколькими возможными ответами
                max_f1 = 0.0
                em_found = False

                for truth in truth_answers:
                    truth_normalized = self.normalize_answer(truth)

                    # Exact match
                    if pred_normalized == truth_normalized:
                        if not em_found:
                            exact_matches += 1
                            em_found = True
                        max_f1 = 1.0
                        break

                    # F1 на уровне токенов
                    pred_tokens = pred_normalized.split()
                    truth_tokens = truth_normalized.split()

                    if not pred_tokens or not truth_tokens:
                        token_f1 = 0.0
                    else:
                        common = Counter(pred_tokens) & Counter(truth_tokens)
                        num_same = sum(common.values())

                        if num_same == 0:
                            token_f1 = 0.0
                        else:
                            precision = num_same / len(pred_tokens)
                            recall = num_same / len(truth_tokens)
                            token_f1 = (
                                2 * precision * recall / (precision + recall)
                                if (precision + recall) > 0
                                else 0
                            )

                        max_f1 = max(max_f1, token_f1)

                f1_scores.append(max_f1)

        em = exact_matches / len(predictions) if predictions else 0
        f1 = np.mean(f1_scores) if f1_scores else 0

        logger.debug(
            f"SQuAD2 evaluation: EM={em:.3f}, F1={f1:.3f}, "
            f"exact_matches={exact_matches}/{len(predictions)}"
        )

        return {
            'exact_match': em,
            'f1': f1,           # ОСНОВНАЯ МЕТРИКА
            'accuracy': em
        }

    # ========== HotpotQA ==========

    def evaluate_hotpotqa(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Оценка для HotpotQA (Multi-hop QA).
        Метрика: token-level F1 (как SQuAD, но без impossible вопросов).
        """
        exact_matches = 0
        f1_scores = []

        for pred, truth in zip(predictions, ground_truth):
            pred_normalized = self.normalize_answer(pred)
            truth_normalized = self.normalize_answer(truth)

            if pred_normalized == truth_normalized:
                exact_matches += 1
                f1_scores.append(1.0)
                continue

            pred_tokens = pred_normalized.split()
            truth_tokens = truth_normalized.split()

            if not pred_tokens or not truth_tokens:
                f1_scores.append(0.0)
                continue

            common = Counter(pred_tokens) & Counter(truth_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                f1_scores.append(0.0)
            else:
                precision = num_same / len(pred_tokens)
                recall = num_same / len(truth_tokens)
                token_f1 = (2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0 else 0)
                f1_scores.append(token_f1)

        em = exact_matches / len(predictions) if predictions else 0
        f1 = float(np.mean(f1_scores)) if f1_scores else 0

        return {
            'exact_match': em,
            'f1': f1,
            'accuracy': em
        }

    # ========== CommonGen Метрики ==========

    def evaluate_commongen(
        self,
        predictions: List[str],
        ground_truth,
        data: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Оценка для CommonGen (генерация текста из концептов).

        ОСНОВНАЯ МЕТРИКА: BERTScore F1
        Дополнительно: ROUGE-1/2/L

        Multi-reference BERTScore — если data содержит 'all_targets',
        передаём list-of-lists в BERTScorer для multi-reference оценки.
        BERTScore нативно берёт max score среди всех references.

        Args:
            predictions: Список сгенерированных предложений
            ground_truth: Список эталонных предложений (single ref, backward compat)
            data: Опциональный список словарей с полем 'all_targets' для multi-ref

        Returns:
            Словарь с метриками:
            {
                'bert_score_f1': float,  # ОСНОВНАЯ МЕТРИКА
                'rouge1': float,
                'rouge2': float,
                'rougeL': float,
                'accuracy': float  # Алиас для bert_score_f1
            }
        """
        if not predictions or not ground_truth:
            logger.warning("CommonGen: Empty predictions or ground_truth")
            return {
                'bert_score_f1': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'accuracy': 0.0
            }

        # Multi-reference BERTScore: если data содержит all_targets,
        # передаём list-of-lists (BERTScorer берёт max среди refs)
        if data and any('all_targets' in item for item in data):
            refs = [item.get('all_targets', [item['target']]) for item in data]
            logger.debug(
                "CommonGen: using multi-reference BERTScore "
                f"(avg {np.mean([len(r) for r in refs]):.1f} refs per example)"
            )
        else:
            refs = ground_truth

        # BERTScore (ОСНОВНАЯ МЕТРИКА)
        _, _, F1 = self.bert_scorer.score(predictions, refs)
        bert_f1 = F1.mean().item()

        # ROUGE scores (дополнительные) — используем single ref (ground_truth)
        # для ROUGE, т.к. rouge_scorer не поддерживает multi-ref нативно
        rouge_scores = []
        for pred, truth in zip(predictions, ground_truth):
            scores = self.rouge_scorer.score(truth, pred)
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })

        avg_rouge = {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]) if rouge_scores else 0.0,
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]) if rouge_scores else 0.0,
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores]) if rouge_scores else 0.0
        }

        logger.debug(
            f"CommonGen evaluation: BERTScore F1={bert_f1:.3f}, "
            f"ROUGE-1={avg_rouge['rouge1']:.3f}"
        )

        return {
            'bert_score_f1': bert_f1,      # ОСНОВНАЯ МЕТРИКА
            'rouge1': avg_rouge['rouge1'],
            'rouge2': avg_rouge['rouge2'],
            'rougeL': avg_rouge['rougeL'],
            'accuracy': bert_f1  # Алиас
        }

    # ========== XSum Метрики ==========

    def evaluate_xsum(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Оценка для XSum (суммаризация новостей).

        ОСНОВНАЯ МЕТРИКА: BERTScore F1
        Дополнительно: ROUGE-1/2/L

        Args:
            predictions: Список сгенерированных суммаризаций
            ground_truth: Список эталонных суммаризаций

        Returns:
            Словарь с метриками:
            {
                'bert_score_f1': float,  # ОСНОВНАЯ МЕТРИКА
                'rouge1': float,
                'rouge2': float,
                'rougeL': float,
                'accuracy': float  # Алиас для bert_score_f1
            }
        """
        if not predictions or not ground_truth:
            logger.warning("XSum: Empty predictions or ground_truth")
            return {
                'bert_score_f1': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'accuracy': 0.0
            }

        # BERTScore (ОСНОВНАЯ МЕТРИКА)
        _, _, F1 = self.bert_scorer.score(predictions, ground_truth)
        bert_f1 = F1.mean().item()

        # ROUGE scores
        rouge_scores = []
        for pred, truth in zip(predictions, ground_truth):
            scores = self.rouge_scorer.score(truth, pred)
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })

        avg_rouge = {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]) if rouge_scores else 0.0,
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]) if rouge_scores else 0.0,
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores]) if rouge_scores else 0.0
        }

        logger.debug(
            f"XSum evaluation: BERTScore F1={bert_f1:.3f}, "
            f"ROUGE-L={avg_rouge['rougeL']:.3f}"
        )

        return {
            'bert_score_f1': bert_f1,      # ОСНОВНАЯ МЕТРИКА
            'rouge1': avg_rouge['rouge1'],
            'rouge2': avg_rouge['rouge2'],
            'rougeL': avg_rouge['rougeL'],
            'accuracy': bert_f1  # Алиас
        }

    # ========== CodeSearchNet Метрики ==========

    def evaluate_codesearchnet(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Оценка для CodeSearchNet (code summarization).

        ОСНОВНАЯ МЕТРИКА: BERTScore F1
        Дополнительно: ROUGE-1/2/L
        """
        if not predictions or not ground_truth:
            logger.warning("CodeSearchNet: Empty predictions or ground_truth")
            return {
                'bert_score_f1': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'accuracy': 0.0
            }

        # BERTScore (ОСНОВНАЯ МЕТРИКА)
        _, _, F1 = self.bert_scorer.score(predictions, ground_truth)
        bert_f1 = F1.mean().item()

        # ROUGE scores
        rouge_scores = []
        for pred, truth in zip(predictions, ground_truth):
            scores = self.rouge_scorer.score(truth, pred)
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })

        avg_rouge = {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]) if rouge_scores else 0.0,
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]) if rouge_scores else 0.0,
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores]) if rouge_scores else 0.0
        }

        logger.debug(
            f"CodeSearchNet evaluation: BERTScore F1={bert_f1:.3f}, "
            f"ROUGE-L={avg_rouge['rougeL']:.3f}"
        )

        return {
            'bert_score_f1': bert_f1,      # ОСНОВНАЯ МЕТРИКА
            'rouge1': avg_rouge['rouge1'],
            'rouge2': avg_rouge['rouge2'],
            'rougeL': avg_rouge['rougeL'],
            'accuracy': bert_f1  # Алиас
        }

    # ========== Generic Evaluation ==========

    def evaluate(
        self,
        dataset_name: str,
        predictions: List[str],
        ground_truth: List,
        data: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Универсальный метод оценки для любого датасета.

        Args:
            dataset_name: Название датасета
            predictions: Список предсказаний
            ground_truth: Список эталонных ответов
            data: Опциональные исходные данные (для multi-ref BERTScore в CommonGen)

        Returns:
            Словарь с метриками для данного датасета

        Raises:
            ValueError: Если dataset_name неизвестен
        """
        if dataset_name == 'GSM8K':
            return self.evaluate_gsm8k(predictions, ground_truth)
        elif dataset_name == 'AG_News':
            return self.evaluate_ag_news(predictions, ground_truth)
        elif dataset_name == 'SQuAD_2':
            return self.evaluate_squad2(predictions, ground_truth)
        elif dataset_name == 'CommonGen':
            return self.evaluate_commongen(predictions, ground_truth, data=data)
        elif dataset_name == 'XSum':
            return self.evaluate_xsum(predictions, ground_truth)
        elif dataset_name == 'CodeSearchNet':
            return self.evaluate_codesearchnet(predictions, ground_truth)
        elif dataset_name == 'HotpotQA':
            return self.evaluate_hotpotqa(predictions, ground_truth)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_primary_metric_name(self, dataset_name: str) -> str:
        """
        Получить название основной метрики для датасета.

        Args:
            dataset_name: Название датасета

        Returns:
            Название основной метрики для fitness

        Raises:
            ValueError: Если dataset_name неизвестен
        """
        primary_metrics = {
            'GSM8K': 'exact_match',
            'AG_News': 'f1_macro',
            'SQuAD_2': 'f1',
            'HotpotQA': 'f1',
            'CommonGen': 'bert_score_f1',
            'XSum': 'bert_score_f1',
            'CodeSearchNet': 'bert_score_f1'
        }

        if dataset_name not in primary_metrics:
            # RiderGenesis и другие синтетические датасеты — fallback на llm_judge
            return 'llm_judge'

        return primary_metrics[dataset_name]

    def __repr__(self) -> str:
        """Строковое представление для отладки."""
        return "MetricsEvaluator(GSM8K, AG_News, SQuAD_2, CommonGen, XSum)"
