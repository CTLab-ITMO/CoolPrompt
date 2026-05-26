"""EvoPrompt-GA (Genetic Algorithm) - ICLR 2024.

Точная реализация из статьи "Connecting Large Language Models with
Evolutionary Algorithms Yields Powerful Prompt Optimizers" (Guo et al., ICLR 2024).

Источники:
- Статья: https://arxiv.org/abs/2309.08532
- Код: https://github.com/beeevita/EvoPrompt
"""

import logging
import random
import re
from typing import List, Dict, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rider.core.prompts import Prompt
from rider.evaluation.evaluator import PromptEvaluator
from rider.execution.history import EvolutionHistory

logger = logging.getLogger(__name__)


# Промпты для GA из оригинального репозитория
# https://github.com/beeevita/EvoPrompt/blob/main/data/template_ga.py
GA_TEMPLATES = {
    "cls": """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,

    "sim": """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts to generate a new prompt:
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: In this task, you are given comments from movie reviews. Your task is to classify each comment as one of the following categories: terrible, bad, okay, good, great.
2. <prompt>Given a sentence from a movie review, classify it into one of the following categories: terrible, bad, okay, good, or great.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,

    "sum": """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Simplify the complex text while maintaining its meaning.
2. <prompt>Simplify the complex text while maintaining its meaning.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,

    "qa": """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Simplify the complex text while maintaining its meaning.
2. <prompt>Simplify the complex text while maintaining its meaning.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,
}


class EvoPromptGA:
    """EvoPrompt с Genetic Algorithm (ICLR 2024).

    Гиперпараметры из статьи:
    - population_size: 10 (для text classification/generation)
    - num_generations: 10
    - temperature: 0.5 (для GA операций)
    - top_p: 0.95
    - selection: "wheel" (roulette wheel selection)
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
        self.ga_template = GA_TEMPLATES[task_type]

        # Evolution history for detailed logging
        if self.save_history:
            results_dir = Path("results")
            experiment_id = f"{dataset_name}_EvoPrompt-GA"
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

        logger.info(f"EvoPrompt-GA initialized: pop_size={population_size}, "
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
            return "sum"  # Похоже на генерацию
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

        Используем простую генерацию на основе примеров из train_data.

        Fair seed diversity (v17.1): temperature sweep [0.5, 0.7, 0.9, 1.1, 1.3]
        вместо одной фиксированной температуры T=0.5. Плюс разные случайные
        демо-примеры для каждого промпта. Это выравнивает baseline с RIDER —
        раньше все 10 промптов генерировались с одним T и одними демо, что
        давало почти идентичные seed-ы и искусственно занижало EvoPrompt-GA.
        """
        logger.info(f"Generating initial population of {self.population_size} prompts...")

        # Fair seed diversity: temperature sweep instead of single T=0.5
        temperatures = [0.5, 0.7, 0.9, 1.1, 1.3]

        def _gen_init_prompt(i):
            # Варьируем температуру по индексу (равномерный sweep)
            temp = temperatures[i % len(temperatures)]

            # Варьируем демо-примеры для каждого промпта (разные случайные сэмплы)
            if len(train_data) > 0:
                demo_examples = random.sample(train_data, min(5, len(train_data)))
            else:
                demo_examples = []
            demo_str = self._format_demos(demo_examples)

            generation_prompt = (
                "Generate a concise instruction (one sentence) for solving these problems:\n\n"
                f"{demo_str}\n\n"
                "Instruction:"
            )

            try:
                response = self.llm_client.generate(
                    prompt=generation_prompt,
                    model=self.model,
                    temperature=temp,  # VARY temperature
                    top_p=self.top_p,
                    max_tokens=200
                )
                instruction = response.strip().strip('"').strip("'").strip()
                if instruction.lower().startswith("instruction:"):
                    instruction = instruction[12:].strip()
                if len(instruction) < 10:
                    # Fallback: простой generic промпт с той же температурой
                    instruction = "Solve the following task step by step and provide the answer."
                return i, temp, Prompt(text=instruction, id=i)
            except Exception as e:
                logger.warning(f"Init prompt {i} generation failed (T={temp}): {e}")
                # Fallback template с варьирующейся температурой (уже выбрана выше)
                fallback_text = "Solve the following task step by step and provide the answer."
                return i, temp, Prompt(text=fallback_text, id=i)

        results = {}
        temps_used = {}
        max_workers = min(32, self.population_size)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gen_init_prompt, i): i for i in range(self.population_size)}
            for future in as_completed(futures):
                idx, temp, prompt = future.result()
                results[idx] = prompt
                temps_used[idx] = temp

        population = [results[i] for i in range(self.population_size)]

        # Логируем распределение температур для debug
        from collections import Counter
        temp_dist = Counter(temps_used.values())
        logger.info(
            "Initial population temperature distribution: "
            f"{dict(sorted(temp_dist.items()))} "
            f"(total: {len(population)} prompts)"
        )

        return population

    def _roulette_wheel_selection(self, population: List[Prompt]) -> Tuple[Prompt, Prompt]:
        """Roulette wheel selection для выбора двух родителей.

        Вероятность выбора пропорциональна fitness.
        """
        # Нормализуем fitness (сдвигаем если есть отрицательные)
        min_fitness = min(p.fitness for p in population)
        if min_fitness < 0:
            fitnesses = [p.fitness - min_fitness + 0.01 for p in population]
        else:
            fitnesses = [p.fitness + 0.01 for p in population]  # +0.01 чтобы избежать 0

        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]

        # Выбираем двух родителей
        parent1 = random.choices(population, weights=probabilities, k=1)[0]
        parent2 = random.choices(population, weights=probabilities, k=1)[0]

        return parent1, parent2

    def _ga_crossover_mutation(self, parent1: Prompt, parent2: Prompt) -> Prompt:
        """GA: Crossover + Mutation через LLM.

        Используем промпт из оригинального EvoPrompt:
        1. Crossover двух родителей
        2. Mutation результата
        """
        # Заполняем шаблон
        ga_prompt = self.ga_template.replace("<prompt1>", parent1.text).replace("<prompt2>", parent2.text)

        # Запрашиваем LLM
        response = self.llm_client.generate(
            prompt=ga_prompt,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=200  # Больше токенов для crossover + mutation
        )

        # Извлекаем финальный промпт из тегов <prompt>...</prompt>
        match = re.search(r'<prompt>(.*?)</prompt>', response, re.DOTALL)
        if match:
            offspring_text = match.group(1).strip()
        else:
            # Если теги не найдены, берем весь ответ после "2."
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('2.'):
                    offspring_text = '\n'.join(lines[i:]).strip()
                    # Удаляем префикс "2."
                    offspring_text = re.sub(r'^2\.\s*', '', offspring_text).strip()
                    break
            else:
                # Fallback: берем весь ответ
                offspring_text = response.strip()

        # Очистка
        offspring_text = offspring_text.replace("<prompt>", "").replace("</prompt>", "").strip()

        return Prompt(text=offspring_text)

    def run(self, train_data, val_data, dev_data, test_data=None):
        """Запуск EvoPrompt-GA алгоритма.

        Алгоритм:
        1. Генерация начальной популяции
        2. Цикл эволюции (num_generations):
           - Evaluate population
           - Для каждого потомка:
             * Roulette wheel selection (2 родителя)
             * GA crossover + mutation
           - Merge old + new population
           - Keep top-N
        3. Evaluate на dev set
        4. Возврат лучшего промпта
        """
        # 1. Инициализация
        population = self._generate_initial_population(train_data)

        # Evaluate начальную популяцию на dev set — PARALLEL
        logger.info("Evaluating initial population...")

        ga_init_results = {}

        def _eval_ga_init(idx_prompt):
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
                logger.error(f"GA init eval failed for prompt {idx}: {e}")
                p.fitness = 0.0
                return idx, None

        eval_workers = min(8, len(population))
        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
            for idx, result in executor.map(_eval_ga_init, enumerate(population)):
                ga_init_results[idx] = result

        # Sequential logging
        for i, prompt in enumerate(population):
            result = ga_init_results.get(i)
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

        logger.info(f"Initial population - Best fitness: {max(p.fitness for p in population):.4f}")

        # 2. Эволюционный цикл
        for gen in range(self.num_generations):
            logger.info(f"\n=== Generation {gen + 1}/{self.num_generations} ===")

            # Генерируем потомков параллельно (LLM generation)
            def _gen_ga_child(i):
                parent1, parent2 = self._roulette_wheel_selection(population)
                child = self._ga_crossover_mutation(parent1, parent2)
                child.id = len(population) + i
                return i, child, parent1, parent2

            offspring_data = {}
            max_workers = min(32, self.population_size)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_gen_ga_child, i): i for i in range(self.population_size)}
                for future in as_completed(futures):
                    idx, child, p1, p2 = future.result()
                    offspring_data[idx] = (child, p1, p2)

            # Evaluate offspring in PARALLEL
            ga_gen_results = {}

            def _eval_ga_child(idx):
                child, _, _ = offspring_data[idx]
                try:
                    if self.log_detailed_evaluations and self.history:
                        result = self.evaluator.evaluate_with_details(
                            prompt=child, dataset_name=self.dataset_name, data=val_data
                        )
                        metrics = result['metrics']
                        primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                        child.fitness = metrics[primary_metric]
                        return idx, result
                    else:
                        child.fitness = self.evaluator.evaluate_prompt(
                            prompt=child, dataset_name=self.dataset_name, data=val_data
                        )
                        return idx, None
                except Exception as e:
                    logger.error(f"GA gen eval failed for child {idx}: {e}")
                    child.fitness = 0.0
                    return idx, None

            eval_workers = min(8, self.population_size)
            with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                for idx, result in executor.map(_eval_ga_child, range(self.population_size)):
                    ga_gen_results[idx] = result

            # Sequential logging + collect offspring
            offspring = []
            for i in range(self.population_size):
                child, parent1, parent2 = offspring_data[i]
                result = ga_gen_results.get(i)

                if result and self.log_detailed_evaluations and self.history:
                    predictions = result['predictions']
                    ground_truth = result['ground_truth']
                    metrics = result['metrics']
                    primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                    fitness = metrics[primary_metric]
                    error_indices = [j for j, (pred, truth) in enumerate(zip(predictions, ground_truth)) if pred != truth]

                    self.history.log_detailed_evaluation(
                        prompt_id=child.id,
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
                    self.history.log_evolution_step(
                        generation=gen + 1,
                        operator_used='ga_crossover_mutation',
                        parent_ids=[str(parent1.id), str(parent2.id)],
                        parent_fitnesses=[parent1.fitness, parent2.fitness],
                        offspring=child,
                        temperature=self.temperature,
                        top_p=1.0,
                        diversity_score=0.0,
                        accepted=True,
                        metadata={
                            'parent1_fitness': parent1.fitness,
                            'parent2_fitness': parent2.fitness,
                            'improvement': child.fitness - max(parent1.fitness, parent2.fitness)
                        }
                    )

                offspring.append(child)

            # Merge и keep top-N
            combined = population + offspring
            combined.sort(key=lambda p: p.fitness, reverse=True)
            population = combined[:self.population_size]

            best_fitness = population[0].fitness
            avg_fitness = sum(p.fitness for p in population) / len(population)
            logger.info(f"Gen {gen + 1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

        # 3. Финальный результат
        best_prompt = population[0]
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
