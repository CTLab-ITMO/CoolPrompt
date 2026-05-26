"""
UnifiedDatasetLoader для загрузки всех датасетов.

Этот модуль реализует загрузку 5 датасетов из HuggingFace:
1. GSM8K - Математические задачи grade school level
2. AG_News - Классификация новостей (4 класса)
3. SQuAD 2.0 - Question Answering с невозможными вопросами
4. CommonGen - Генерация текста из концептов
5. XSum - Суммаризация новостных статей

Особенности:
- Случайная выборка samples для воспроизводимости
- Разделение на train/val/dev/test splits
- Унифицированный формат данных
- Кэширование датасетов
- Поддержка custom split sizes
"""

import os
import random
from collections import OrderedDict
from typing import Dict, List, Optional
import logging

import numpy as np
from datasets import load_dataset

# Детерминистические seed-offsets вместо hash().
# Python 3.3+ рандомизирует hash() между процессами (PYTHONHASHSEED),
# поэтому hash('CommonGen_val') давал разные val данные в разных экспериментах.
# Фиксированные константы гарантируют одинаковые данные для сравнения методов.
_SEED_OFFSETS = {
    'GSM8K': 100, 'GSM8K_test': 101,
    'AG_News': 200, 'AG_News_test': 201,
    'SQuAD_2': 300, 'SQuAD_2_test': 301,
    'CommonGen_train': 400, 'CommonGen_val': 401,
    'XSum': 500, 'XSum_test': 501,
    'CodeSearchNet': 600, 'CodeSearchNet_test': 601,
    'HotpotQA': 700, 'HotpotQA_test': 701,
}

logger = logging.getLogger(__name__)


class UnifiedDatasetLoader:
    """
    Unified loader для всех датасетов RIDER.

    Загружает датасеты из HuggingFace с random sampling для
    воспроизводимости экспериментов.

    Args:
        cache_dir: Директория для кэширования датасетов
        seed: Random seed для воспроизводимости (default: 42)

    Example:
        >>> loader = UnifiedDatasetLoader(cache_dir="./cache", seed=42)
        >>>
        >>> # Load one dataset
        >>> datasets = loader.load_dataset(
        ...     'GSM8K',
        ...     train_size=100,
        ...     val_size=100,
        ...     dev_size=50,
        ...     test_size=500
        ... )
        >>>
        >>> # Access data
        >>> gsm8k_train = datasets['train']
        >>> print(f"Loaded {len(gsm8k_train)} GSM8K train examples")
    """

    SUPPORTED_DATASETS = ['GSM8K', 'AG_News', 'SQuAD_2', 'CommonGen', 'XSum', 'CodeSearchNet', 'HotpotQA']

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        seed: int = 42
    ):
        """
        Инициализация loader.

        Args:
            cache_dir: Путь к директории для кэша
            seed: Random seed
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), '.cache', 'datasets')
        self.seed = seed

        # Устанавливаем seed
        random.seed(seed)
        np.random.seed(seed)

        logger.info(
            f"UnifiedDatasetLoader initialized with cache_dir={self.cache_dir}, seed={seed}"
        )

    def load_dataset(
        self,
        dataset_name: str,
        train_size: int = 100,
        val_size: int = 100,
        dev_size: int = 50,
        test_size: int = 500,
        data_offset: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Загружает один датасет.

        Args:
            dataset_name: Название датасета
            train_size: Размер train split
            val_size: Размер validation split
            dev_size: Размер dev split
            test_size: Размер test split
            data_offset: Количество образцов для пропуска (для последовательной загрузки)

        Returns:
            Словарь: {'train': [...], 'val': [...], 'dev': [...], 'test': [...]}

        Raises:
            ValueError: Если dataset_name неизвестен
        """
        if dataset_name == 'GSM8K':
            return self._load_gsm8k(train_size, val_size, dev_size, test_size, data_offset)
        elif dataset_name == 'AG_News':
            return self._load_ag_news(train_size, val_size, dev_size, test_size, data_offset)
        elif dataset_name == 'SQuAD_2':
            return self._load_squad2(train_size, val_size, dev_size, test_size, data_offset)
        elif dataset_name == 'CommonGen':
            return self._load_commongen(train_size, val_size, dev_size, test_size, data_offset)
        elif dataset_name == 'XSum':
            return self._load_xsum(train_size, val_size, dev_size, test_size, data_offset)
        elif dataset_name == 'CodeSearchNet':
            return self._load_codesearchnet(train_size, val_size, dev_size, test_size, data_offset)
        elif dataset_name == 'HotpotQA':
            return self._load_hotpotqa(train_size, val_size, dev_size, test_size, data_offset)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # ========== Dataset-Specific Loaders ==========

    def _load_gsm8k(
        self,
        train_size: int,
        val_size: int,
        dev_size: int,
        test_size: int,
        data_offset: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Загружает GSM8K (Grade School Math 8K).

        Формат: {'question': str, 'answer': str}

        Args:
            train_size: Train split size
            val_size: Val split size
            dev_size: Dev split size
            test_size: Test split size
            data_offset: Number of samples to skip (for sequential loading)

        Returns:
            Словарь со splits
        """
        logger.info(f"Loading GSM8K from HuggingFace (offset={data_offset})...")

        dataset = load_dataset("gsm8k", "main", cache_dir=self.cache_dir)

        # GSM8K имеет train и test
        train_data = dataset['train']
        test_data = dataset['test']

        # Random sampling с фиксированным seed: train+val+dev из train split
        total_train_needed = train_size + val_size + dev_size
        available_train = len(train_data)

        rng = random.Random(self.seed + _SEED_OFFSETS['GSM8K'])
        all_train_indices = list(range(available_train))
        rng.shuffle(all_train_indices)
        train_val_dev_indices = all_train_indices[data_offset:data_offset + total_train_needed]

        # Разделяем на train/val/dev
        train_indices = train_val_dev_indices[:train_size]
        val_indices = train_val_dev_indices[train_size:train_size + val_size]
        dev_indices = train_val_dev_indices[train_size + val_size:train_size + val_size + dev_size]

        # Random sampling для test (из test split, независимо от offset)
        available_test = len(test_data)
        all_test_indices = list(range(available_test))
        rng_test = random.Random(self.seed + _SEED_OFFSETS['GSM8K_test'])
        rng_test.shuffle(all_test_indices)
        test_indices = all_test_indices[:test_size]

        # Форматируем данные
        def format_example(item):
            # Извлекаем answer (после ####)
            answer = item['answer'].split('####')[-1].strip() if '####' in item['answer'] else item['answer']
            return {
                'question': item['question'],
                'answer': answer
            }

        return {
            'train': [format_example(train_data[i]) for i in train_indices],
            'val': [format_example(train_data[i]) for i in val_indices],
            'dev': [format_example(train_data[i]) for i in dev_indices],
            'test': [format_example(test_data[i]) for i in test_indices]
        }

    def _load_ag_news(
        self,
        train_size: int,
        val_size: int,
        dev_size: int,
        test_size: int,
        data_offset: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Загружает AG News (news classification).

        Формат: {'text': str, 'label': str}

        Args:
            train_size: Train split size
            val_size: Val split size
            dev_size: Dev split size
            test_size: Test split size

        Returns:
            Словарь со splits
        """
        logger.info("Loading AG_News from HuggingFace...")

        dataset = load_dataset("ag_news", cache_dir=self.cache_dir)

        label_names = ['World', 'Sports', 'Business', 'Sci/Tech']

        train_data = dataset['train']
        test_data = dataset['test']

        # Random sampling с фиксированным seed: train+val+dev из train split
        total_train_needed = train_size + val_size + dev_size
        available_train = len(train_data)

        rng = random.Random(self.seed + _SEED_OFFSETS['AG_News'])
        all_train_indices = list(range(available_train))
        rng.shuffle(all_train_indices)
        train_val_dev_indices = all_train_indices[data_offset:data_offset + total_train_needed]

        # Разделяем
        train_indices = train_val_dev_indices[:train_size]
        val_indices = train_val_dev_indices[train_size:train_size + val_size]
        dev_indices = train_val_dev_indices[train_size + val_size:train_size + val_size + dev_size]

        # Random sampling для test (из test split, независимо от offset)
        available_test = len(test_data)
        all_test_indices = list(range(available_test))
        rng_test = random.Random(self.seed + _SEED_OFFSETS['AG_News_test'])
        rng_test.shuffle(all_test_indices)
        test_indices = all_test_indices[:test_size]

        # Форматируем
        def format_example(item):
            return {
                'text': item['text'],
                'label': label_names[item['label']]
            }

        return {
            'train': [format_example(train_data[i]) for i in train_indices],
            'val': [format_example(train_data[i]) for i in val_indices],
            'dev': [format_example(train_data[i]) for i in dev_indices],
            'test': [format_example(test_data[i]) for i in test_indices]
        }

    def _load_squad2(
        self,
        train_size: int,
        val_size: int,
        dev_size: int,
        test_size: int,
        data_offset: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Загружает SQuAD 2.0 (Question Answering with impossible questions).

        Формат: {'context': str, 'question': str, 'answers': List[str], 'is_impossible': bool}

        Args:
            train_size: Train split size
            val_size: Val split size
            dev_size: Dev split size
            test_size: Test split size

        Returns:
            Словарь со splits
        """
        logger.info("Loading SQuAD 2.0 from HuggingFace...")

        dataset = load_dataset("squad_v2", cache_dir=self.cache_dir)

        train_data = dataset['train']
        validation_data = dataset['validation']

        # Random sampling: train+val+dev из train split
        total_train_needed = train_size + val_size + dev_size
        available_train = len(train_data)

        rng = random.Random(self.seed + _SEED_OFFSETS['SQuAD_2'])
        all_train_indices = list(range(available_train))
        rng.shuffle(all_train_indices)
        train_val_dev_indices = all_train_indices[data_offset:data_offset + total_train_needed]

        # Разделяем
        train_indices = train_val_dev_indices[:train_size]
        val_indices = train_val_dev_indices[train_size:train_size + val_size]
        dev_indices = train_val_dev_indices[train_size + val_size:train_size + val_size + dev_size]

        # Random sampling для test из validation
        available_test = len(validation_data)
        all_test_indices = list(range(available_test))
        rng_test = random.Random(self.seed + _SEED_OFFSETS['SQuAD_2_test'])
        rng_test.shuffle(all_test_indices)
        test_indices = all_test_indices[:test_size]

        # Форматируем
        def format_example(item):
            answers = item['answers']['text'] if item['answers']['text'] else ['']
            is_impossible = len(item['answers']['text']) == 0

            return {
                'context': item['context'],
                'question': item['question'],
                'answers': answers,
                'is_impossible': is_impossible
            }

        return {
            'train': [format_example(train_data[i]) for i in train_indices],
            'val': [format_example(train_data[i]) for i in val_indices],
            'dev': [format_example(train_data[i]) for i in dev_indices],
            'test': [format_example(validation_data[i]) for i in test_indices]
        }

    def _load_hotpotqa(
        self,
        train_size: int,
        val_size: int,
        dev_size: int,
        test_size: int,
        data_offset: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Загружает HotpotQA (Multi-hop Question Answering).
        Используется fullwiki конфигурация как в GEPA (ICLR 2026).

        Формат: {'question': str, 'context': str, 'answer': str}

        GEPA settings: train=150, val=300, test=300
        """
        logger.info("Loading HotpotQA (fullwiki) from HuggingFace...")

        dataset = load_dataset("hotpot_qa", "fullwiki", cache_dir=self.cache_dir)

        train_data = dataset['train']
        validation_data = dataset['validation']

        # Random sampling: train+val+dev из train split
        total_train_needed = train_size + val_size + dev_size
        available_train = len(train_data)

        rng = random.Random(self.seed + _SEED_OFFSETS['HotpotQA'])
        all_train_indices = list(range(available_train))
        rng.shuffle(all_train_indices)
        train_val_dev_indices = all_train_indices[data_offset:data_offset + total_train_needed]

        train_indices = train_val_dev_indices[:train_size]
        val_indices = train_val_dev_indices[train_size:train_size + val_size]
        dev_indices = train_val_dev_indices[train_size + val_size:train_size + val_size + dev_size]

        # Test из validation split
        available_test = len(validation_data)
        rng_test = random.Random(self.seed + _SEED_OFFSETS['HotpotQA_test'])
        all_test_indices = list(range(available_test))
        rng_test.shuffle(all_test_indices)
        test_indices = all_test_indices[:test_size]

        def format_example(item):
            # HotpotQA fullwiki: supporting_facts указывают на контексты
            # Собираем все контексты в один блок
            contexts = []
            if item.get('context') and item['context'].get('title'):
                for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                    contexts.append(f"{title}: {' '.join(sentences)}")
            context_text = '\n'.join(contexts) if contexts else ''

            return {
                'context': context_text[:3000],  # Ограничиваем длину контекста
                'question': item['question'],
                'answer': item['answer'],
            }

        return {
            'train': [format_example(train_data[i]) for i in train_indices],
            'val': [format_example(train_data[i]) for i in val_indices],
            'dev': [format_example(train_data[i]) for i in dev_indices],
            'test': [format_example(validation_data[i]) for i in test_indices]
        }

    @staticmethod
    def _group_commongen_by_concept_set(data, indices) -> List[Dict]:
        """
        Группирует CommonGen примеры по concept_set_idx.

        CommonGen имеет ~4 reference предложения на каждый набор концептов.
        Группируем их, чтобы BERTScore мог использовать multi-reference
        (берёт max score среди всех references).

        Args:
            data: HuggingFace dataset split
            indices: Индексы для выборки

        Returns:
            Список словарей с полями:
            - concepts: List[str]
            - target: str (первый reference, для backward compatibility)
            - all_targets: List[str] (все references для multi-ref BERTScore)
        """
        groups = OrderedDict()
        for i in indices:
            item = data[i]
            key = item['concept_set_idx']
            if key not in groups:
                groups[key] = {
                    'concepts': item['concepts'],
                    'target': item['target'],
                    'all_targets': [item['target']]
                }
            else:
                if item['target'] not in groups[key]['all_targets']:
                    groups[key]['all_targets'].append(item['target'])
        return list(groups.values())

    def _load_commongen(
        self,
        train_size: int,
        val_size: int,
        dev_size: int,
        test_size: int,
        data_offset: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Загружает CommonGen (concept-to-text generation).

        Формат: {'concepts': List[str], 'target': str, 'all_targets': List[str]}

        Multi-reference BERTScore — группируем по concept_set_idx.
        CommonGen имеет ~4 reference на каждый набор концептов.
        all_targets содержит все references для multi-ref оценки.

        Args:
            train_size: Train split size
            val_size: Val split size
            dev_size: Dev split size
            test_size: Test split size

        Returns:
            Словарь со splits
        """
        logger.info("Loading CommonGen from HuggingFace...")

        dataset = load_dataset("common_gen", cache_dir=self.cache_dir)

        train_data = dataset['train']
        validation_data = dataset['validation']

        # --- Train: группируем по concept_set_idx, затем сэмплируем concept sets ---
        # Сначала находим уникальные concept_set_idx в train
        train_concept_set_indices = OrderedDict()
        for i in range(len(train_data)):
            key = train_data[i]['concept_set_idx']
            if key not in train_concept_set_indices:
                train_concept_set_indices[key] = []
            train_concept_set_indices[key].append(i)

        train_concept_set_keys = list(train_concept_set_indices.keys())
        rng_train = random.Random(self.seed + _SEED_OFFSETS['CommonGen_train'])
        rng_train.shuffle(train_concept_set_keys)
        selected_train_keys = train_concept_set_keys[data_offset:data_offset + train_size]
        # Собираем все индексы для выбранных concept sets
        train_flat_indices = []
        for key in selected_train_keys:
            train_flat_indices.extend(train_concept_set_indices[key])

        # --- Validation: группируем по concept_set_idx ---
        val_concept_set_indices = OrderedDict()
        for i in range(len(validation_data)):
            key = validation_data[i]['concept_set_idx']
            if key not in val_concept_set_indices:
                val_concept_set_indices[key] = []
            val_concept_set_indices[key].append(i)

        val_concept_set_keys = list(val_concept_set_indices.keys())
        total_val_needed = val_size + dev_size + test_size
        rng_val = random.Random(self.seed + _SEED_OFFSETS['CommonGen_val'])
        rng_val.shuffle(val_concept_set_keys)
        selected_val_keys = val_concept_set_keys[data_offset:data_offset + total_val_needed]

        # Разделяем val/dev/test по concept sets
        val_keys = selected_val_keys[:val_size]
        dev_keys = selected_val_keys[val_size:val_size + dev_size]
        test_keys = selected_val_keys[val_size + dev_size:val_size + dev_size + test_size]

        val_flat_indices = []
        for key in val_keys:
            val_flat_indices.extend(val_concept_set_indices[key])
        dev_flat_indices = []
        for key in dev_keys:
            dev_flat_indices.extend(val_concept_set_indices[key])
        test_flat_indices = []
        for key in test_keys:
            test_flat_indices.extend(val_concept_set_indices[key])

        logger.info(
            f"CommonGen grouped: train={len(selected_train_keys)} concept sets "
            f"({len(train_flat_indices)} rows), "
            f"val={len(val_keys)} sets ({len(val_flat_indices)} rows), "
            f"dev={len(dev_keys)} sets ({len(dev_flat_indices)} rows), "
            f"test={len(test_keys)} sets ({len(test_flat_indices)} rows)"
        )

        return {
            'train': self._group_commongen_by_concept_set(train_data, train_flat_indices),
            'val': self._group_commongen_by_concept_set(validation_data, val_flat_indices),
            'dev': self._group_commongen_by_concept_set(validation_data, dev_flat_indices),
            'test': self._group_commongen_by_concept_set(validation_data, test_flat_indices)
        }

    def _load_xsum(
        self,
        train_size: int,
        val_size: int,
        dev_size: int,
        test_size: int,
        data_offset: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Загружает XSum (extreme summarization).

        Формат: {'document': str, 'summary': str}

        Args:
            train_size: Train split size
            val_size: Val split size
            dev_size: Dev split size
            test_size: Test split size

        Returns:
            Словарь со splits
        """
        logger.info("Loading XSum from HuggingFace...")

        dataset = load_dataset("xsum", cache_dir=self.cache_dir)

        train_data = dataset['train']
        test_data = dataset['test']

        # Random sampling с фиксированным seed: train+val+dev из train
        total_train_needed = train_size + val_size + dev_size
        available_train = len(train_data)

        rng = random.Random(self.seed + _SEED_OFFSETS['XSum'])
        all_train_indices = list(range(available_train))
        rng.shuffle(all_train_indices)
        train_val_dev_indices = all_train_indices[data_offset:data_offset + total_train_needed]

        # Разделяем
        train_indices = train_val_dev_indices[:train_size]
        val_indices = train_val_dev_indices[train_size:train_size + val_size]
        dev_indices = train_val_dev_indices[train_size + val_size:train_size + val_size + dev_size]

        # Random sampling для test (из test split, независимо от offset)
        available_test = len(test_data)
        all_test_indices = list(range(available_test))
        rng_test = random.Random(self.seed + _SEED_OFFSETS['XSum_test'])
        rng_test.shuffle(all_test_indices)
        test_indices = all_test_indices[:test_size]

        # Форматируем
        def format_example(item):
            return {
                'document': item['document'],
                'summary': item['summary']
            }

        return {
            'train': [format_example(train_data[i]) for i in train_indices],
            'val': [format_example(train_data[i]) for i in val_indices],
            'dev': [format_example(train_data[i]) for i in dev_indices],
            'test': [format_example(test_data[i]) for i in test_indices]
        }

    def _load_codesearchnet(
        self,
        train_size: int,
        val_size: int,
        dev_size: int,
        test_size: int,
        data_offset: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Загружает CodeSearchNet Python (code summarization).

        Формат: {'code': str, 'docstring': str}

        Фильтрация:
        - Функции 3-30 строк (исключаем тривиальные и слишком длинные)
        - Docstring >= 10 символов (исключаем пустые/тривиальные)
        - Код без docstring (func_code_string)
        """
        logger.info("Loading CodeSearchNet Python from HuggingFace...")

        dataset = load_dataset("code_search_net", "python", cache_dir=self.cache_dir)

        train_data = dataset['train']
        test_data = dataset['test']

        # Фильтруем качественные примеры
        def is_good_example(item):
            code = item.get('func_code_string', '')
            doc = item.get('func_documentation_string', '')
            if not code or not doc:
                return False
            code_lines = code.strip().split('\n')
            if len(code_lines) < 3 or len(code_lines) > 30:
                return False
            if len(doc.strip()) < 10:
                return False
            return True

        def format_example(item):
            return {
                'code': item['func_code_string'].strip(),
                'docstring': item['func_documentation_string'].strip()
            }

        # Фильтруем и индексируем train
        logger.info("Filtering CodeSearchNet train examples...")
        good_train_indices = [i for i in range(len(train_data)) if is_good_example(train_data[i])]
        logger.info(f"CodeSearchNet: {len(good_train_indices)}/{len(train_data)} train examples passed filter")

        total_train_needed = train_size + val_size + dev_size
        rng = random.Random(self.seed + _SEED_OFFSETS['CodeSearchNet'])
        rng.shuffle(good_train_indices)
        selected_train = good_train_indices[data_offset:data_offset + total_train_needed]

        train_indices = selected_train[:train_size]
        val_indices = selected_train[train_size:train_size + val_size]
        dev_indices = selected_train[train_size + val_size:train_size + val_size + dev_size]

        # Фильтруем и индексируем test
        good_test_indices = [i for i in range(len(test_data)) if is_good_example(test_data[i])]
        rng_test = random.Random(self.seed + _SEED_OFFSETS['CodeSearchNet_test'])
        rng_test.shuffle(good_test_indices)
        test_indices = good_test_indices[:test_size]

        return {
            'train': [format_example(train_data[i]) for i in train_indices],
            'val': [format_example(train_data[i]) for i in val_indices],
            'dev': [format_example(train_data[i]) for i in dev_indices],
            'test': [format_example(test_data[i]) for i in test_indices]
        }

    def __repr__(self) -> str:
        """Строковое представление для отладки."""
        return (
            "UnifiedDatasetLoader("
            f"cache_dir={self.cache_dir}, "
            f"seed={self.seed}, "
            f"supported={len(self.SUPPORTED_DATASETS)} datasets)"
        )
