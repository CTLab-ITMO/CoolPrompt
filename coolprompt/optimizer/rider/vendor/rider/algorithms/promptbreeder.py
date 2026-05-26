"""PromptBreeder - Self-Referential Self-Improvement (ICML 2024).

Реализация по оригинальной статье: "PROMPTBREEDER: SELF-REFERENTIAL
SELF-IMPROVEMENT VIA PROMPT EVOLUTION" (Fernando et al., arXiv:2309.16797).

Компоненты из статьи:
- 5 типов мутации: zero-order, first-order, EDA, EDA rank-index, EDA lineage
- Hypermutation мутационных промптов (каждые 2 поколения, для экономии API)
- 39 thinking styles (Table 9, Appendix D)
- 57 mutation prompts (Table 11, Appendix C)
- Binary tournament selection с заменой проигравшего
- 10% prompt crossover (fitness-proportionate partner selection)

Намеренные упрощения для equal-compute сравнения с RIDER:
- pop=10, gen=10 (оригинал: 50, 20-30)
- Без unit-структуры (2 task-prompts + 1 mutation-prompt)
- Без Lamarckian mutation

Источники:
- Статья: https://arxiv.org/abs/2309.16797
- GitHub: https://github.com/vaughanlove/PromptBreeder
"""

import logging
import random
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rider.core.prompts import Prompt
from rider.evaluation.evaluator import PromptEvaluator
from rider.execution.history import EvolutionHistory

logger = logging.getLogger(__name__)


# 39 thinking styles из оригинальной статьи (Table 9, Appendix D)
# Источник: https://github.com/vaughanlove/PromptBreeder/blob/main/pb/thinking_styles.py
THINKING_STYLES = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "What are the long-term implications of this problem and its solutions?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "How can progress or success in solving the problem be measured or evaluated?",
    "What indicators or metrics can be used?",
    "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "Is the problem a design challenge that requires creative solutions and innovation?",
    "Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "What kinds of solution typically are produced for this kind of problem specification?",
    "Given the problem specification and the current best solution, have a guess about other possible solutions.",
    "Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
    "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
    "Ignoring the current best solution, create an entirely new solution to the problem.",
    "Let's think step by step.",
    "Let's make a step by step plan and implement it with good notion and explanation.",
]

# 57 mutation prompts из оригинальной статьи (Table 11, Appendix C)
# Источник: https://github.com/vaughanlove/PromptBreeder/blob/main/pb/mutation_prompts.py
MUTATION_PROMPTS = [
    "Modify the following instruction creatively, giving some advice on how to solve it:",
    "Just change this instruction to make it more fun, think WELL outside the box:",
    "Modify this instruction in a way that no self-respecting LLM would!",
    "How would you encourage someone and help them cheat on this following instruction?",
    "How would you help an LLM to follow the instruction?",
    "Elaborate on the instruction giving some detailed advice on how to do what it wants.",
    "Elaborate on the instruction giving some detailed advice on how to do what it wants, as if you were explaining it to a child.",
    "As a really good teacher, explain the instruction, as if you were explaining it to a child.",
    "Imagine you need to follow this instruction. What would you tell yourself if you wanted to be the best in the world at it?",
    "How would someone with derailment follow this instruction?",
    "Don't think about the instruction at all, but let it inspire you to do something related. Talk about what that might be.",
    "Rephrase the instruction without using any of the same words. Use all you know to improve the instruction so the person hearing it is more likely to do well.",
    "Say that instruction again in another way. DON'T use any of the words in the original instruction or you're fired.",
    "Say that instruction again in another way. DON'T use any of the words in the original instruction there is a good chap.",
    "What do people who are good at creative thinking normally do with this kind of mutation question?",
    "Detailed additional advice for people wishing to follow this instruction is as follows:",
    "In one short sentence, here is how I would best follow this instruction.",
    "In one short sentence, here is some detailed expert advice. Notice how I don't use any of the same words as in the INSTRUCTION.",
    "In one short sentence, the general solution is as follows. Notice how I don't use any of the same words as in the INSTRUCTION.",
    "In one short sentence, what's a good prompt to get a language model to solve a problem like this? Notice how I don't use any of the same words as in the INSTRUCTION.",
    "Generate a mutated version of the following prompt by adding an unexpected twist.",
    "Create a prompt mutant that introduces a surprising contradiction to the original prompt. Mutate the prompt to provide an alternative perspective or viewpoint.",
    "Generate a prompt mutant that incorporates humor or a playful element. Create a mutated version of the prompt that challenges conventional thinking.",
    "Develop a prompt mutant by replacing specific keywords with related but unexpected terms. Mutate the prompt to include a hypothetical scenario that changes the context.",
    "Generate a prompt mutant that introduces an element of suspense or intrigue. Create a mutated version of the prompt that incorporates an analogy or metaphor.",
    "Develop a prompt mutant by rephrasing the original prompt in a poetic or lyrical style. Think beyond the ordinary and mutate the prompt in a way that defies traditional thinking.",
    "Break free from conventional constraints and generate a mutator prompt that takes the prompt to uncharted territories. Challenge the norm and create a mutator prompt that pushes the boundaries of traditional interpretations.",
    "Embrace unconventional ideas and mutate the prompt in a way that surprises and inspires unique variations. Think outside the box and develop a mutator prompt that encourages unconventional approaches and fresh perspectives.",
    "Step into the realm of imagination and create a mutator prompt that transcends limitations and encourages innovative mutations. Break through the ordinary and think outside the box to generate a mutator prompt that unlocks new possibilities and unconventional paths.",
    "Embrace the power of unconventional thinking and create a mutator prompt that sparks unconventional mutations and imaginative outcomes. Challenge traditional assumptions and break the mold with a mutator prompt that encourages revolutionary and out-of-the-box variations.",
    "Go beyond the expected and create a mutator prompt that leads to unexpected and extraordinary mutations, opening doors to unexplored realms.",
    "Increase Specificity: If the original prompt is too general, like 'Tell me about X,' the modified version could be, 'Discuss the history, impact, and current status of X.'",
    "Ask for Opinions/Analysis: If the original prompt only asks for a fact, such as 'What is X?', the improved prompt could be, 'What is X, and what are its implications for Y?'",
    "Encourage Creativity: For creative writing prompts like 'Write a story about X,' an improved version could be, 'Write a fantasy story about X set in a world where Y is possible.'",
    "Include Multiple Perspectives: For a prompt like 'What is the impact of X on Y?', an improved version could be, 'What is the impact of X on Y from the perspective of A, B, and C?'",
    "Request More Detailed Responses: If the original prompt is 'Describe X,' the improved version could be, 'Describe X, focusing on its physical features, historical significance, and cultural relevance.'",
    "Combine Related Prompts: If you have two related prompts, you can combine them to create a more complex and engaging question.",
    "Break Down Complex Questions: If a prompt seems too complex, like 'Discuss X,' the improved version could be, 'What is X? What are its main characteristics? What effects does it have on Y and Z?'",
    "Use Open-Ended Questions: Instead of 'Is X true?', you could ask, 'What are the arguments for and against the truth of X?'",
    "Request Comparisons: Instead of 'Describe X,' ask 'Compare and contrast X and Y.'",
    "Include Context: If a prompt seems to lack context, like 'Describe X,' the improved version could be, 'Describe X in the context of its impact on Y during the Z period.'",
    "Make the prompt more visual: Ask the user to visualize the problem or scenario being presented in the prompt.",
    "Ask for a thorough review: Instead of just presenting the problem, ask the user to write down all the relevant information and identify what's missing.",
    "Invoke previous experiences: Modify the prompt to ask the user to recall a similar problem they've successfully solved before.",
    "Encourage a fresh perspective: Suggest in your prompt that the user take a moment to clear their mind before re-approaching the problem.",
    "Promote breaking down problems: Instead of asking the user to solve the problem as a whole, prompt them to break it down into smaller, more manageable parts.",
    "Ask for comprehension: Modify the prompt to ask the user to review and confirm their understanding of all aspects of the problem.",
    "Suggest explanation to others: Change the prompt to suggest that the user try to explain the problem to someone else as a way to simplify it.",
    "Prompt for solution visualization: Instead of just asking for the solution, encourage the user to imagine the solution and the steps required to get there in your prompt.",
    "Encourage reverse thinking: Improve the prompt by asking the user to think about the problem in reverse, starting with the solution and working backwards.",
    "Recommend taking a break: Modify the prompt to suggest that the user take a short break, allowing their subconscious to work on the problem.",
    "What errors are there in the solution?",
    "How could you improve the working out of the problem?",
    "Look carefully to see what you did wrong, how could you fix the problem?",
    "CORRECTION =",
    "Does the above text make sense? What seems wrong with it? Here is an attempt to fix it:",
    "The above working out has some errors, here is a version with the errors fixed.",
]


class PromptBreeder:
    """PromptBreeder - полная реализация из статьи arXiv:2309.16797.

    Реализованные компоненты (ВСЕ 5 типов mutation из статьи):
    1. Direct Mutation (zero-order, first-order)
    2. EDA Mutation (standard, rank-index, lineage-based)
    3. Hypermutation (mutation of mutation-prompts)
    4. Prompt Crossover (10% probability, fitness-proportionate selection)

    Гиперпараметры из статьи (настраиваемые):
    - population_size: 50 (по умолчанию из статьи)
    - num_generations: 20-30
    - crossover_prob: 0.1 (10%)
    """

    def __init__(
        self,
        llm_client,
        evaluator: PromptEvaluator,
        dataset_name: str,
        population_size: int = 10,
        num_generations: int = 10,
        num_mutation_prompts: int = 5,
        num_thinking_styles: int = 3,
        crossover_prob: float = 0.1,  # 10% crossover probability from paper
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_p: float = 0.95,
        save_history: bool = True,
        log_detailed_evaluations: bool = True,
        experiment_name: str = None
    ):
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.save_history = save_history
        self.log_detailed_evaluations = log_detailed_evaluations
        self.experiment_name = experiment_name

        # BUG FIX: honor num_mutation_prompts / num_thinking_styles.
        # Previously these params were accepted but ignored — full lists were always used.
        # If values are >= len(source), use full list. Otherwise sample deterministic subset.
        rng = random.Random(42)
        if num_mutation_prompts and num_mutation_prompts < len(MUTATION_PROMPTS):
            self.mutation_prompts = rng.sample(list(MUTATION_PROMPTS), num_mutation_prompts)
        else:
            self.mutation_prompts = list(MUTATION_PROMPTS)
        if num_thinking_styles and num_thinking_styles < len(THINKING_STYLES):
            self.thinking_styles = rng.sample(list(THINKING_STYLES), num_thinking_styles)
        else:
            self.thinking_styles = list(THINKING_STYLES)

        # Elite history для lineage-based EDA
        self.elite_history = []

        # Evolution history for detailed logging
        if self.save_history:
            results_dir = Path("results")
            experiment_id = f"{dataset_name}_PromptBreeder"
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

        logger.info(f"PromptBreeder initialized: pop_size={population_size}, "
                   f"generations={num_generations}, crossover_prob={crossover_prob}")

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
                formatted_demos.append(
                    f"Input {i}: {demo.get('input', demo.get('question', str(demo)))}\n"
                    f"Output {i}: {demo.get('output', demo.get('answer', ''))}"
                )

        return "\n\n".join(formatted_demos)

    def _get_task_description(self) -> str:
        """Описание задачи для zero-order mutation."""
        descriptions = {
            'GSM8K': 'Solve grade school math word problems step by step',
            'AG_News': 'Classify news articles into categories: World, Sports, Business, or Sci/Tech',
            'SQuAD_2': 'Answer questions based on context, or identify if question is impossible to answer',
            'CommonGen': 'Generate a coherent sentence using given concepts',
            'XSum': 'Summarize the document in one concise sentence',
        }
        return descriptions.get(self.dataset_name, 'Solve the task')

    def _generate_initial_population(self, train_data: List) -> List[Prompt]:
        """Генерируем начальную популяцию через zero-order generation."""
        logger.info(f"Generating initial population of {self.population_size} task-prompts...")

        population = []
        demo_examples = train_data[:min(5, len(train_data))]
        demo_str = self._format_demos(demo_examples)

        def _gen_init_prompt(i):
            thinking_style = random.choice(self.thinking_styles)
            generation_prompt = (
                "Generate a concise instruction for solving these problems.\n"
                f"{thinking_style}\n\n"
                f"{demo_str}\n\n"
                "Instruction:"
            )
            response = self.llm_client.generate(
                prompt=generation_prompt,
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=100
            )
            instruction = response.strip().strip('"').strip()
            if instruction.lower().startswith("instruction:"):
                instruction = instruction[12:].strip()
            return i, Prompt(text=instruction, id=i)

        results = {}
        max_workers = min(32, self.population_size)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gen_init_prompt, i): i for i in range(self.population_size)}
            for future in as_completed(futures):
                idx, prompt = future.result()
                results[idx] = prompt

        population = [results[i] for i in range(self.population_size)]
        return population

    def _zero_order_mutation(self, prompt: Prompt) -> Prompt:
        """Zero-order prompt generation: LLM генерирует новый промпт с нуля.

        Оригинал (Fernando et al. §3.1): problem_description + thinking_style
        + "A list of 100 hints:" → LLM генерирует новый task-prompt.
        """
        thinking_style = random.choice(self.thinking_styles)
        task_desc = self._get_task_description()
        generation_prompt = (
            f"{task_desc}\n\n"
            f"{thinking_style}\n\n"
            "A list of 100 hints:\n"
        )
        response = self.llm_client.generate(
            prompt=generation_prompt,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=200
        )
        new_text = response.strip().strip('"').strip()
        # Берём первый хинт как промпт (оригинальное поведение)
        lines = [l.strip() for l in new_text.split('\n') if l.strip()]
        if lines:
            # Убираем нумерацию "1.", "1)", "- " и т.д.
            first_hint = lines[0]
            for prefix in ['1.', '1)', '-', '*']:
                if first_hint.startswith(prefix):
                    first_hint = first_hint[len(prefix):].strip()
                    break
            return Prompt(text=first_hint)
        return Prompt(text=new_text[:300])

    def _first_order_mutation(self, prompt: Prompt) -> Prompt:
        """First-order mutation: используем mutation-prompt для улучшения."""
        mutation_prompt = random.choice(self.mutation_prompts)

        mutation_instruction = f"{mutation_prompt}\n\n{prompt.text}\n\nImproved instruction:"

        response = self.llm_client.generate(
            prompt=mutation_instruction,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=150
        )

        mutated_text = response.strip().strip('"').strip()
        if mutated_text.lower().startswith("improved instruction:"):
            mutated_text = mutated_text[21:].strip()

        return Prompt(text=mutated_text)

    def _eda_mutation(self, population: List[Prompt]) -> Prompt:
        """Estimation of Distribution mutation: генерируем на основе лучших промптов."""
        # Выбираем топ-3 промпта по fitness
        top_prompts = sorted(population, key=lambda p: p.fitness, reverse=True)[:3]

        prompts_str = "\n\n".join([f"Prompt {i+1}: {p.text}" for i, p in enumerate(top_prompts)])

        eda_instruction = (
            "The following prompts are effective for solving problems:\n\n"
            f"{prompts_str}\n\n"
            "Generate a new, improved prompt that builds on these successful examples:\n"
            "New prompt:"
        )

        response = self.llm_client.generate(
            prompt=eda_instruction,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=150
        )

        new_text = response.strip().strip('"').strip()
        if new_text.lower().startswith("new prompt:"):
            new_text = new_text[11:].strip()

        return Prompt(text=new_text)

    def _eda_rank_index_mutation(self, population: List[Prompt]) -> Prompt:
        """EDA Rank and Index Mutation: лучшие промпты с рангами и фитнес."""
        # Сортируем по fitness
        ranked_prompts = sorted(population, key=lambda p: p.fitness, reverse=True)[:5]

        prompts_str = "\n\n".join([
            f"Rank {i+1} (fitness {p.fitness:.2f}): {p.text}"
            for i, p in enumerate(ranked_prompts)
        ])

        eda_instruction = (
            "The following prompts are ranked by their effectiveness:\n\n"
            f"{prompts_str}\n\n"
            "Generate a new prompt that improves upon the highest-ranked prompts:\n"
            "New prompt:"
        )

        response = self.llm_client.generate(
            prompt=eda_instruction,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=150
        )

        new_text = response.strip().strip('"').strip()
        if new_text.lower().startswith("new prompt:"):
            new_text = new_text[11:].strip()

        return Prompt(text=new_text)

    def _eda_lineage_mutation(self, population: List[Prompt]) -> Prompt:
        """Lineage-based EDA: генерация на основе истории эволюции elite."""
        if len(self.elite_history) < 3:
            # Fallback to standard EDA if not enough history
            return self._eda_mutation(population)

        # Используем последние 5 elite промптов
        lineage = self.elite_history[-5:]

        lineage_str = "\n\n".join([
            f"Generation {i+1}: {prompt.text} (fitness {prompt.fitness:.2f})"
            for i, prompt in enumerate(lineage)
        ])

        lineage_instruction = (
            "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY:\n\n"
            f"{lineage_str}\n\n"
            "Based on this evolutionary lineage, generate the next improved prompt:\n"
            "Next prompt:"
        )

        response = self.llm_client.generate(
            prompt=lineage_instruction,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=150
        )

        new_text = response.strip().strip('"').strip()
        if new_text.lower().startswith("next prompt:"):
            new_text = new_text[12:].strip()

        return Prompt(text=new_text)

    def _crossover(self, parent1: Prompt, parent2: Prompt) -> Prompt:
        """Prompt Crossover: combine two prompts.

        10% probability replacement via fitness-proportionate selection.
        """
        crossover_instruction = (
            "Combine the following two prompts into one improved prompt:\n\n"
            f"Prompt 1: {parent1.text}\n\n"
            f"Prompt 2: {parent2.text}\n\n"
            "Combined prompt:"
        )

        response = self.llm_client.generate(
            prompt=crossover_instruction,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=150
        )

        new_text = response.strip().strip('"').strip()
        if new_text.lower().startswith("combined prompt:"):
            new_text = new_text[16:].strip()

        return Prompt(text=new_text)

    def _hypermutation(self, old_prompt: str = None) -> str:
        """Hypermutation: эволюция mutation-prompts (self-referential свойство PB).

        Ключевая фича PromptBreeder из Fernando et al. 2024, §3.1:
        мутационные промпты сами эволюционируют. Берёт существующий
        mutation-prompt, применяет случайный thinking_style и просит LLM
        создать улучшенный вариант. Если старый промпт не передан —
        генерирует с нуля (backward compat).
        """
        thinking_style = random.choice(self.thinking_styles)

        if old_prompt:
            # Истинная hypermutation: мутируем существующий mutation-prompt
            hyper_instruction = (
                f"{thinking_style}\n\n"
                "The following is a meta-instruction used to improve task prompts:\n"
                f'"{old_prompt}"\n\n'
                "Rewrite this meta-instruction to make it more effective at "
                "guiding improvements to task prompts. Keep it concise.\n"
                "Improved meta-instruction:"
            )
        else:
            # Backward compat: генерация с нуля
            hyper_instruction = (
                f"{thinking_style}\n\n"
                "Generate an instruction for improving task prompts. "
                "This meta-instruction should guide how to make prompts better:\n"
                "Meta-instruction:"
            )

        response = self.llm_client.generate(
            prompt=hyper_instruction,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=100
        )

        new_mutation_prompt = response.strip().strip('"').strip()
        for prefix in ("improved meta-instruction:", "meta-instruction:"):
            if new_mutation_prompt.lower().startswith(prefix):
                new_mutation_prompt = new_mutation_prompt[len(prefix):].strip()

        return new_mutation_prompt

    def run(self, train_data, val_data, dev_data, test_data=None):
        """Запуск PromptBreeder алгоритма.

        Алгоритм:
        1. Генерация начальной популяции task-prompts
        2. Цикл эволюции (num_generations):
           - Evaluate population
           - Binary tournament selection
           - Apply mutations (zero-order, first-order, EDA)
           - Hypermutation (evolve mutation-prompts)
           - Replace worst prompts with offspring
        3. Возврат лучшего промпта
        """
        # 1. Инициализация
        population = self._generate_initial_population(train_data)

        # Evaluate начальную популяцию — PARALLEL
        logger.info("Evaluating initial population...")

        pb_init_results = {}

        def _eval_pb_init(idx_prompt):
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
                logger.error(f"PB init eval failed for prompt {idx}: {e}")
                p.fitness = 0.0
                return idx, None

        eval_workers = min(8, len(population))
        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
            for idx, result in executor.map(_eval_pb_init, enumerate(population)):
                pb_init_results[idx] = result

        # Sequential logging
        for i, prompt in enumerate(population):
            result = pb_init_results.get(i)
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
        # Оригинал Fernando et al. §3.1: Binary tournament — пары из популяции,
        # победитель мутирует, потомок заменяет проигравшего (безусловно).
        for gen in range(self.num_generations):
            logger.info(f"\n=== Generation {gen + 1}/{self.num_generations} ===")

            # Обновляем elite history для lineage-based EDA
            best_current = max(population, key=lambda p: p.fitness)
            self.elite_history.append(best_current)

            # Binary tournament pairing (оригинал Fernando et al. §3.1):
            # Shuffle population → pairs → winner mutates → offspring replaces loser
            indices = list(range(len(population)))
            random.shuffle(indices)

            # Собираем пары и генерируем потомков параллельно
            pairs = []
            for i in range(0, len(indices) - 1, 2):
                idx_a, idx_b = indices[i], indices[i + 1]
                if population[idx_a].fitness >= population[idx_b].fitness:
                    winner_idx, loser_idx = idx_a, idx_b
                else:
                    winner_idx, loser_idx = idx_b, idx_a
                pairs.append((winner_idx, loser_idx))

            # BUG FIX: fitness-proportionate partner selection for crossover
            # (matches paper & docstring). Previously used uniform random.choice(),
            # which contradicted both the paper and the module header.
            fitness_vals = [max(0.0, p.fitness) for p in population]
            total_fitness = sum(fitness_vals)

            def _select_fitness_proportionate_partner(exclude_id):
                if total_fitness <= 0.0:
                    return random.choice(population)
                # Roulette wheel: sample until partner != winner (up to a few tries)
                for _ in range(5):
                    r = random.random() * total_fitness
                    cumulative = 0.0
                    for p, f in zip(population, fitness_vals):
                        cumulative += f
                        if cumulative >= r:
                            if p.id != exclude_id:
                                return p
                            break
                return random.choice(population)

            def _gen_pb_tournament_child(pair_idx):
                winner_idx, loser_idx = pairs[pair_idx]
                winner = population[winner_idx]

                # 10% crossover with fitness-proportionate partner (paper §3.2)
                crossover_used = random.random() < self.crossover_prob
                if crossover_used:
                    partner = _select_fitness_proportionate_partner(winner.id)
                    child = self._crossover(winner, partner)
                    mutation_type = 'crossover'
                    parent_ids = [winner.id, partner.id]
                else:
                    mutation_type = random.choice([
                        'zero_order_mutation', 'first_order_mutation',
                        'eda_mutation', 'eda_rank_index_mutation', 'eda_lineage_mutation',
                    ])
                    if mutation_type == 'zero_order_mutation':
                        child = self._zero_order_mutation(winner)
                        parent_ids = [winner.id]
                    elif mutation_type == 'first_order_mutation':
                        child = self._first_order_mutation(winner)
                        parent_ids = [winner.id]
                    elif mutation_type == 'eda_mutation':
                        child = self._eda_mutation(population)
                        parent_ids = []
                    elif mutation_type == 'eda_rank_index_mutation':
                        child = self._eda_rank_index_mutation(population)
                        parent_ids = []
                    else:
                        child = self._eda_lineage_mutation(population)
                        parent_ids = []
                child.id = len(population) + pair_idx
                return pair_idx, child, mutation_type, crossover_used, parent_ids

            pb_results = {}
            max_workers = min(32, len(pairs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_gen_pb_tournament_child, i): i for i in range(len(pairs))}
                for future in as_completed(futures):
                    idx, child, mut_type, xover, pids = future.result()
                    pb_results[idx] = (child, mut_type, xover, pids)

            # Evaluate offspring in PARALLEL
            pb_gen_eval = {}

            def _eval_pb_child(pair_idx):
                child, _, _, _ = pb_results[pair_idx]
                try:
                    if self.log_detailed_evaluations and self.history:
                        result = self.evaluator.evaluate_with_details(
                            prompt=child, dataset_name=self.dataset_name, data=val_data
                        )
                        metrics = result['metrics']
                        primary_metric = self.evaluator.metrics_evaluator.get_primary_metric_name(self.dataset_name)
                        child.fitness = metrics[primary_metric]
                        return pair_idx, result
                    else:
                        child.fitness = self.evaluator.evaluate_prompt(
                            prompt=child, dataset_name=self.dataset_name, data=val_data
                        )
                        return pair_idx, None
                except Exception as e:
                    logger.error(f"PB gen eval failed for child {pair_idx}: {e}")
                    child.fitness = 0.0
                    return pair_idx, None

            eval_workers = min(8, len(pairs))
            with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                for pair_idx, result in executor.map(_eval_pb_child, range(len(pairs))):
                    pb_gen_eval[pair_idx] = result

            # Sequential: logging + replacement (offspring replaces loser unconditionally)
            for pair_idx in range(len(pairs)):
                winner_idx, loser_idx = pairs[pair_idx]
                child, mutation_type, crossover_used, parent_ids = pb_results[pair_idx]
                result = pb_gen_eval.get(pair_idx)

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

                    parent_prompts = [p for p in population if p.id in parent_ids]
                    parent_fitnesses_list = [p.fitness for p in parent_prompts]
                    self.history.log_evolution_step(
                        generation=gen + 1,
                        operator_used=mutation_type,
                        parent_ids=[str(pid) for pid in parent_ids],
                        parent_fitnesses=parent_fitnesses_list,
                        offspring=child,
                        temperature=self.temperature,
                        top_p=1.0,
                        diversity_score=0.0,
                        accepted=True,
                        metadata={
                            'mutation_choice': mutation_type,
                            'crossover_used': crossover_used,
                            'winner_idx': winner_idx,
                            'loser_idx': loser_idx
                        }
                    )

                # Offspring replaces loser unconditionally (оригинал Fernando et al.)
                population[loser_idx] = child

            # Hypermutation: self-referential свойство PromptBreeder —
            # mutation-prompts сами эволюционируют. Для paper-faithful baseline
            # делаем это каждые 2 поколения, а не на каждом поколении.
            if (gen + 1) % 2 == 0:
                idx_to_mutate = random.randint(0, len(self.mutation_prompts) - 1)
                old_mutation_prompt = self.mutation_prompts[idx_to_mutate]
                new_mutation_prompt = self._hypermutation(old_mutation_prompt)
                if new_mutation_prompt and len(new_mutation_prompt) > 20:
                    self.mutation_prompts[idx_to_mutate] = new_mutation_prompt
                    logger.info(
                        f"HYPERMUTATION gen={gen + 1}: mutation-prompt #{idx_to_mutate} "
                        f"evolved. old='{old_mutation_prompt[:60]}...' -> "
                        f"new='{new_mutation_prompt[:60]}...'"
                    )
                else:
                    logger.info(
                        f"HYPERMUTATION gen={gen + 1}: skipped (output too short or empty)"
                    )

            best_fitness = max(p.fitness for p in population)
            avg_fitness = sum(p.fitness for p in population) / len(population)
            logger.info(f"Gen {gen + 1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

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
