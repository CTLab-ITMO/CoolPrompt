"""EvoPrompt-DE (Differential Evolution) - ICLR 2024.

Точная реализация из статьи "Connecting Large Language Models with
Evolutionary Algorithms Yields Powerful Prompt Optimizers" (Guo et al., ICLR 2024).

Источники:
- Статья: https://arxiv.org/abs/2309.08532
- Код: https://github.com/beeevita/EvoPrompt
"""

import logging
import random
import re
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rider.core.prompts import Prompt
from rider.evaluation.evaluator import PromptEvaluator
from rider.execution.history import EvolutionHistory

logger = logging.getLogger(__name__)


# Промпты для DE из оригинального репозитория
# https://github.com/beeevita/EvoPrompt/blob/main/data/template_de.py
# Используем v2 (упрощенная версия с 3 шагами)
DE_TEMPLATES_V2 = {
    "cls": """Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
Different parts:
"Your task is to classify the comment" vs "In this task, you are given sentences from movie reviews. The task is to classify a sentence"
"comment" vs "sentences from movie reviews"

2. Randomly mutate the different parts:
"Your task is to classify the comment" -> "The objective is to categorize the statement"
"comment" -> "phrases in movie reviews"

3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: You are a sentiment classifier. To do this, you must first understand the meaning of the sentence and any relevant context. And then you should classify it as one of the following categories: terrible, bad, okay, good, great.

Final Prompt: <prompt>As a sentiment classifier, analyze phrases in movie reviews and categorize them into one of the following categories: terrible, bad, okay, good, great, while considering the meaning and relevant context.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1. """,

    "sim": """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
Different parts:
"input text" vs "my complex sentence"
"simpler text" vs "simpler terms, but keep the meaning"

2. Randomly mutate the different parts:
"input text" -> "provided text"
"my complex sentence" -> "the difficult sentence"
"simpler text" -> "easier language"
"simpler terms, but keep the meaning" -> "simpler words while maintaining the meaning"

3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.

Final Prompt: <prompt>Transform the difficult sentence into easier language while keeping the meaning, for non-native English speakers to comprehend.</prompt>


Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1. """,

    "sum": """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1. """,

    "qa": """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Crossover the different parts with the following Prompt 3 and generate a final prompt bracketed with <prompt> and </prompt>:
Prompt 3: <prompt0>

1. """,
}


class EvoPromptDE:
    """EvoPrompt с Differential Evolution (ICLR 2024).

    Алгоритм DE/current-to-best/1:
    - Для каждого промпта в популяции генерируем потомка
    - Потомок создается через мутацию разности двух случайных промптов + текущий лучший
    - Если потомок лучше родителя, заменяем родителя

    Гиперпараметры из статьи:
    - population_size: 10
    - num_generations: 10
    - temperature: 0.5 (для DE операций)
    - top_p: 0.95
    """

    def __init__(
        self,
        llm_client,
        evaluator: PromptEvaluator,
        dataset_name: str,
        population_size: int = 10,
        num_generations: int = 10,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.5,
        top_p: float = 0.95,
        task_type: str = None,
        save_history: bool = True,
        log_detailed_evaluations: bool = True,
        experiment_name: str = None
    ):
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.population_size = population_size
        self.num_generations = num_generations
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.save_history = save_history
        self.log_detailed_evaluations = log_detailed_evaluations
        self.experiment_name = experiment_name

        # Определяем тип задачи для выбора промпта
        if task_type is None:
            task_type = self._infer_task_type(dataset_name)
        self.task_type = task_type
        self.de_template = DE_TEMPLATES_V2[task_type]

        # Evolution history for detailed logging
        if self.save_history:
            results_dir = Path("results")
            experiment_id = f"{dataset_name}_EvoPrompt-DE"
            # Use experiment_name as parent directory if provided
            if self.experiment_name:
                save_dir = results_dir / self.experiment_name / experiment_id
            else:
                save_dir = results_dir / experiment_id
            self.history = EvolutionHistory(
                save_dir=save_dir,
                experiment_id=experiment_id
            )
        else:
            self.history = None

        logger.info(f"EvoPrompt-DE initialized: pop_size={population_size}, "
                   f"generations={num_generations}, task_type={task_type}")

    def _infer_task_type(self, dataset_name: str) -> str:
        """Определяем тип задачи по названию датасета."""
        dataset_name_lower = dataset_name.lower()
        if "ag_news" in dataset_name_lower or "classification" in dataset_name_lower:
            return "cls"
        elif "xsum" in dataset_name_lower or "summarization" in dataset_name_lower:
            return "sum"
        elif "squad" in dataset_name_lower or "gsm8k" in dataset_name_lower or "qa" in dataset_name_lower:
            return "qa"
        elif "commongen" in dataset_name_lower or "generation" in dataset_name_lower:
            return "sum"
        else:
            logger.warning(f"Unknown dataset {dataset_name}, using 'qa' task type")
            return "qa"

    def _format_demos(self, demos: List[Dict]) -> str:
        """Форматируем демо-примеры для разных датасетов."""
        formatted_demos = []

        for i, demo in enumerate(demos, 1):
            if self.dataset_name == 'GSM8K':
                formatted_demos.append(
                    f"Input {i}: {demo['question']}\n"
                    f"Output {i}: {demo['answer']}"
                )
            elif self.dataset_name == 'AG_News':
                formatted_demos.append(
                    f"Input {i}: {demo['text']}\n"
                    f"Output {i}: {demo['label']}"
                )
            elif self.dataset_name == 'SQuAD_2':
                formatted_demos.append(
                    f"Input {i}: Context: {demo['context'][:100]}... Question: {demo['question']}\n"
                    f"Output {i}: {demo['answers'][0]}"
                )
            elif self.dataset_name == 'CommonGen':
                formatted_demos.append(
                    f"Input {i}: {', '.join(demo['concepts'])}\n"
                    f"Output {i}: {demo['target']}"
                )
            elif self.dataset_name == 'XSum':
                formatted_demos.append(
                    f"Input {i}: {demo['document'][:100]}...\n"
                    f"Output {i}: {demo['summary']}"
                )
            else:
                # Fallback
                formatted_demos.append(
                    f"Input {i}: {demo.get('input', demo.get('question', str(demo)))}\n"
                    f"Output {i}: {demo.get('output', demo.get('answer', ''))}"
                )

        return "\n\n".join(formatted_demos)

    def _generate_initial_population(self, train_data: List) -> List[Prompt]:
        """Генерируем начальную популяцию через zero-order generation.

        Используем temperature sweep [0.5, 0.7, 0.9, 1.1, 1.3] и разные
        случайные demo-примеры для каждого промпта — для честного сравнения
        с RIDER (который делает то же самое в initialize_population).
        """
        logger.info(f"Generating initial population of {self.population_size} prompts...")

        temperatures = [0.5, 0.7, 0.9, 1.1, 1.3]

        def _gen_init_prompt(i, temp):
            # Каждый промпт получает разные случайные demo-примеры
            demo_sample = random.sample(train_data, min(5, len(train_data)))
            demo_str = self._format_demos(demo_sample)
            generation_prompt = (
                "Generate a concise instruction (one sentence) for solving these problems:\n\n"
                f"{demo_str}\n\n"
                "Instruction:"
            )
            try:
                response = self.llm_client.generate(
                    prompt=generation_prompt,
                    model=self.model,
                    temperature=temp,
                    top_p=self.top_p,
                    max_tokens=200
                )
                instruction = response.strip().strip('"').strip("'").strip()
                if instruction.lower().startswith("instruction:"):
                    instruction = instruction[12:].strip()
                if len(instruction) < 10:
                    return i, None
                return i, Prompt(text=instruction, id=i)
            except Exception as e:
                logger.warning(f"Init prompt {i} generation failed: {e}")
                return i, None

        results: Dict[int, Prompt] = {}
        max_workers = min(32, self.population_size)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_gen_init_prompt, i, temperatures[i % len(temperatures)]): i
                for i in range(self.population_size)
            }
            for future in as_completed(futures):
                idx, prompt = future.result()
                if prompt is not None:
                    results[idx] = prompt

        # Ретраи для неудачных генераций — используем другую температуру
        missing = [i for i in range(self.population_size) if i not in results]
        retry_attempt = 0
        while missing and retry_attempt < 3:
            retry_attempt += 1
            with ThreadPoolExecutor(max_workers=min(32, len(missing))) as executor:
                # Сдвигаем температуру для разнообразия при ретрае
                futures = {
                    executor.submit(
                        _gen_init_prompt, i,
                        temperatures[(i + retry_attempt) % len(temperatures)]
                    ): i
                    for i in missing
                }
                for future in as_completed(futures):
                    idx, prompt = future.result()
                    if prompt is not None:
                        results[idx] = prompt
            missing = [i for i in range(self.population_size) if i not in results]

        # Если всё ещё не хватает — дублируем существующие (крайний случай)
        if missing and results:
            fallback_prompts = list(results.values())
            for i in missing:
                src = fallback_prompts[i % len(fallback_prompts)]
                results[i] = Prompt(text=src.text, id=i)

        population = [results[i] for i in range(self.population_size) if i in results]
        return population

    def _de_mutation(self, best_prompt: Prompt, prompt1: Prompt, prompt2: Prompt, current: Prompt) -> Prompt:
        """DE mutation: DE/current-to-best/1.

        Используем промпт из оригинального EvoPrompt:
        1. Identify different parts между prompt1 и prompt2
        2. Randomly mutate different parts
        3. Crossover с current (prompt0) — текущий индивид, а не лучший

        Формула DE/current-to-best/1: trial = current + F*(best - current) + F*(r1 - r2)
        <prompt0> = current (base vector для crossover)
        <prompt1>, <prompt2> = два случайных промпта (donor vectors)
        best_prompt передаётся как контекст для направления мутации
        """
        # <prompt0> = current (текущий индивид — base vector для crossover)
        # <prompt1>, <prompt2> = два случайных промпта (donor vectors)
        # best_prompt добавляется как контекст для направления мутации
        de_prompt = (self.de_template
                     .replace("<prompt0>", current.text)
                     .replace("<prompt1>", prompt1.text)
                     .replace("<prompt2>", prompt2.text))
        # Добавляем best prompt как ориентир для DE/current-to-best/1
        de_prompt += f"\nNote: The current best-performing prompt is: \"{best_prompt.text}\". Use it as a reference direction for improvement."

        # Запрашиваем LLM
        response = self.llm_client.generate(
            prompt=de_prompt,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=300  # Больше токенов для DE (3 шага)
        )

        # Извлекаем финальный промпт из тегов <prompt>...</prompt>
        match = re.search(r'<prompt>(.*?)</prompt>', response, re.DOTALL)
        if match:
            offspring_text = match.group(1).strip()
        else:
            # Fallback: берем весь ответ после "3." или "Final Prompt:"
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if 'final prompt' in line.lower() or line.strip().startswith('3.'):
                    offspring_text = '\n'.join(lines[i:]).strip()
                    offspring_text = re.sub(r'^(3\.|Final Prompt:)\s*', '', offspring_text, flags=re.IGNORECASE).strip()
                    break
            else:
                offspring_text = response.strip()

        # Очистка
        offspring_text = offspring_text.replace("<prompt>", "").replace("</prompt>", "").strip()

        return Prompt(text=offspring_text)

    def run(self, train_data, val_data, dev_data, test_data=None):
        """Запуск EvoPrompt-DE алгоритма.

        Алгоритм DE/current-to-best/1:
        1. Генерация начальной популяции
        2. Цикл эволюции (num_generations):
           - Evaluate population
           - Для каждого промпта:
             * Выбираем best и два случайных (prompt1, prompt2)
             * DE mutation
             * Если потомок лучше родителя, заменяем
        3. Возврат лучшего промпта
        """
        # 1. Инициализация
        population = self._generate_initial_population(train_data)

        # Evaluate начальную популяцию — PARALLEL
        logger.info("Evaluating initial population...")

        de_init_results = {}

        def _eval_de_init(idx_prompt):
            idx, p = idx_prompt
            try:
                if self.log_detailed_evaluations and self.history:
                    result = self.evaluator.evaluate_with_details(
                        prompt=p, dataset_name=self.dataset_name, data=val_data
                    )
                    metrics = result['metrics']
                    primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                    p.fitness = metrics[primary_metric]
                    return idx, result
                else:
                    p.fitness = self.evaluator.evaluate_prompt(
                        prompt=p, dataset_name=self.dataset_name, data=val_data
                    )
                    return idx, None
            except Exception as e:
                logger.error(f"DE init eval failed for prompt {idx}: {e}")
                p.fitness = 0.0
                return idx, None

        eval_workers = min(8, len(population))
        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
            for idx, result in executor.map(_eval_de_init, enumerate(population)):
                de_init_results[idx] = result

        # Sequential logging
        for i, prompt in enumerate(population):
            result = de_init_results.get(i)
            if result and self.log_detailed_evaluations and self.history:
                predictions = result['predictions']
                ground_truth = result['ground_truth']
                metrics = result['metrics']
                primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                fitness = metrics[primary_metric]
                error_indices = [j for j, (pred, truth) in enumerate(zip(predictions, ground_truth)) if pred != truth]

                self.history.log_detailed_evaluation(
                    prompt_id=prompt.id,
                    generation=0,
                    dataset_name=self.dataset_name,
                    evaluation_details={
                        'fitness': fitness,
                        'predictions': predictions,
                        'ground_truth': ground_truth,
                        'error_indices': error_indices,
                        'metrics': metrics
                    }
                )

                self.history.log_evolution_step(
                    generation=0,
                    operator_used='zero_order',
                    parent_ids=[],
                    parent_fitnesses=[],
                    offspring=prompt,
                    temperature=self.temperature,
                    top_p=1.0,
                    diversity_score=0.0,
                    accepted=True,
                    metadata={'initial_population': True}
                )

        best_fitness = max(p.fitness for p in population)
        logger.info(f"Initial population - Best fitness: {best_fitness:.4f}")

        # 2. Эволюционный цикл
        for gen in range(self.num_generations):
            logger.info(f"\n=== Generation {gen + 1}/{self.num_generations} ===")

            # Находим лучший промпт
            best_prompt = max(population, key=lambda p: p.fitness)

            # Генерируем все мутанты параллельно (batch DE — стандартный вариант)
            def _gen_de_mutant(idx):
                current_prompt = population[idx]
                candidates = [p for p in population if p.id != current_prompt.id]
                if len(candidates) < 2:
                    return idx, None, current_prompt
                prompt1, prompt2 = random.sample(candidates, 2)
                offspring = self._de_mutation(best_prompt, prompt1, prompt2, current_prompt)
                offspring.id = len(population) + idx
                return idx, offspring, current_prompt

            de_results = {}
            max_workers = min(32, len(population))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_gen_de_mutant, idx): idx for idx in range(len(population))}
                for future in as_completed(futures):
                    idx, offspring, current_prompt = future.result()
                    de_results[idx] = (offspring, current_prompt)

            # Evaluate ALL offspring in PARALLEL, then replace sequentially
            valid_indices = [idx for idx in range(len(population)) if de_results[idx][0] is not None]

            de_gen_eval = {}

            def _eval_de_offspring(idx):
                offspring, _ = de_results[idx]
                try:
                    if self.log_detailed_evaluations and self.history:
                        result = self.evaluator.evaluate_with_details(
                            prompt=offspring, dataset_name=self.dataset_name, data=val_data
                        )
                        metrics = result['metrics']
                        primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                        offspring.fitness = metrics[primary_metric]
                        return idx, result
                    else:
                        offspring.fitness = self.evaluator.evaluate_prompt(
                            prompt=offspring, dataset_name=self.dataset_name, data=val_data
                        )
                        return idx, None
                except Exception as e:
                    logger.error(f"DE gen eval failed for offspring {idx}: {e}")
                    offspring.fitness = 0.0
                    return idx, None

            eval_workers = min(8, len(valid_indices))
            if valid_indices:
                with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                    for idx, result in executor.map(_eval_de_offspring, valid_indices):
                        de_gen_eval[idx] = result

            # Sequential replacement + logging
            replacements = 0
            for idx in range(len(population)):
                offspring, current_prompt = de_results[idx]
                if offspring is None:
                    continue

                result = de_gen_eval.get(idx)
                if result and self.log_detailed_evaluations and self.history:
                    predictions = result['predictions']
                    ground_truth = result['ground_truth']
                    metrics = result['metrics']
                    primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                    fitness = metrics[primary_metric]
                    error_indices = [j for j, (pred, truth) in enumerate(zip(predictions, ground_truth)) if pred != truth]

                    self.history.log_detailed_evaluation(
                        prompt_id=offspring.id,
                        generation=gen + 1,
                        dataset_name=self.dataset_name,
                        evaluation_details={
                            'fitness': fitness,
                            'predictions': predictions,
                            'ground_truth': ground_truth,
                            'error_indices': error_indices,
                            'metrics': metrics
                        }
                    )

                    accepted = offspring.fitness > current_prompt.fitness
                    self.history.log_evolution_step(
                        generation=gen + 1,
                        operator_used='de_mutation',
                        parent_ids=[str(best_prompt.id)],
                        parent_fitnesses=[best_prompt.fitness],
                        offspring=offspring,
                        temperature=self.temperature,
                        top_p=1.0,
                        diversity_score=0.0,
                        accepted=accepted,
                        metadata={
                            'current_fitness': current_prompt.fitness,
                            'offspring_fitness': offspring.fitness,
                            'replaced': accepted
                        }
                    )
                else:
                    accepted = offspring.fitness > current_prompt.fitness

                if accepted:
                    population[idx] = offspring
                    replacements += 1

            best_fitness = max(p.fitness for p in population)
            avg_fitness = sum(p.fitness for p in population) / len(population)
            logger.info(f"Gen {gen + 1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, Replacements={replacements}/{self.population_size}")

        # 3. Финальный результат
        best_prompt = max(population, key=lambda p: p.fitness)
        logger.info(f"\nBest prompt found: {best_prompt.text}")
        logger.info(f"Best fitness: {best_prompt.fitness:.4f}")

        # Optional: test evaluation
        if test_data:
            best_prompt.test_fitness = self.evaluator.evaluate_prompt(
                prompt=best_prompt,
                dataset_name=self.dataset_name,
                data=test_data
            )
            logger.info(f"Test fitness: {best_prompt.test_fitness:.4f}")

        # Save history
        if self.save_history and self.history:
            self.history.save()
            logger.info("Evolution history saved")

        return best_prompt
