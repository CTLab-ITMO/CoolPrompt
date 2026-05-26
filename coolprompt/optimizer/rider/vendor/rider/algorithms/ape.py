"""
APE (Automatic Prompt Engineer) - Zhou et al., ICLR 2023 (arXiv:2211.01910)

Paper: "Large Language Models Are Human-Level Prompt Engineers"
Conference: ICLR 2023

Algorithm:
1. Generate M instruction candidates using LLM (forward mode)
2. Evaluate each candidate on validation data using likelihood scoring
3. Select best instruction based on evaluation scores
4. Test on held-out test set

Key Innovation: Treats instruction as "program" optimized by searching over
LLM-generated candidates to maximize a score function.

Reference Implementation: https://github.com/keirp/automatic_prompt_engineer
"""

import logging
import random
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rider.core.prompts import Prompt
from rider.evaluation.evaluator import PromptEvaluator
from rider.execution.history import EvolutionHistory

logger = logging.getLogger(__name__)


class APE:
    """
    APE (Automatic Prompt Engineer) - Zhou et al., ICLR 2023 (arXiv:2211.01910)

    Generates M instruction candidates via LLM and selects best based on
    likelihood-based evaluation on validation data.

    Algorithm Steps:
    1. Generate M=50 candidate instructions using forward mode generation
    2. Evaluate each candidate on dev set using likelihood scoring
    3. Select instruction with highest evaluation score
    4. Test best instruction on test set
    """

    def __init__(
        self,
        llm_client,
        evaluator: PromptEvaluator,
        dataset_name: str,
        num_prompts: int = 50,
        num_demos: int = 5,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        top_p: float = 0.99,
        save_history: bool = True,
        log_detailed_evaluations: bool = True,
        experiment_name: str = None
    ):
        """
        Initialize APE.

        Args:
            llm_client: LLM client for generation
            evaluator: Prompt evaluator for scoring
            dataset_name: Dataset name
            num_prompts: Number of candidate instructions to generate (default: 50)
            num_demos: Number of demos for generation template (default: 5)
            model: LLM model name
            temperature: Sampling temperature for generation (default: 1.0)
            top_p: Nucleus sampling parameter (default: 0.99)
            save_history: Save evolution history (default: True)
            log_detailed_evaluations: Log predictions vs ground truth (default: True)
            experiment_name: Experiment name for results directory (default: None)
        """
        self.llm_client = llm_client
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.num_prompts = num_prompts
        self.num_demos = num_demos
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.save_history = save_history
        self.log_detailed_evaluations = log_detailed_evaluations
        self.experiment_name = experiment_name

        # Evolution history for detailed logging
        if self.save_history:
            results_dir = Path("results")
            experiment_id = f"{dataset_name}_APE"
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

        logger.info(
            f"APE initialized for {dataset_name}: "
            f"num_prompts={num_prompts}, num_demos={num_demos}, "
            f"model={model}, T={temperature}"
        )

    def run(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        dev_data: List[Dict],
        test_data: Optional[List[Dict]] = None
    ) -> Prompt:
        """
        Run APE algorithm.

        Args:
            train_data: Training examples (used for demo in generation)
            val_data: Validation data (used for evaluation, same as RIDER)
            dev_data: Development data (not used, reserved for final selection)
            test_data: Test data for final evaluation

        Returns:
            Best prompt with fitness scores
        """
        logger.info(f"Running APE on {self.dataset_name}...")

        # Step 1: Generate M candidate instructions
        logger.info(f"Generating {self.num_prompts} candidate instructions...")
        candidates = self._generate_candidates(train_data)
        logger.info(f"Generated {len(candidates)} candidates")

        # Step 2: Evaluate candidates on val set (same split as RIDER for fair comparison) — PARALLEL
        logger.info("Evaluating candidates on val set...")

        ape_eval_results = {}

        def _eval_ape_candidate(idx_prompt):
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
                logger.error(f"APE eval failed for candidate {idx}: {e}")
                p.fitness = 0.0
                return idx, None

        eval_workers = max(1, min(8, len(candidates)))
        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
            for idx, result in executor.map(_eval_ape_candidate, enumerate(candidates)):
                ape_eval_results[idx] = result

        # Sequential logging
        for i, prompt in enumerate(candidates):
            result = ape_eval_results.get(i)
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
                    operator_used='ape_generation',
                    parent_ids=[],
                    parent_fitnesses=[],
                    offspring=prompt,
                    temperature=self.temperature,
                    top_p=1.0,
                    diversity_score=0.0,
                    accepted=True,
                    metadata={'candidate_num': i, 'num_demos': self.num_demos}
                )

        # Step 3: Iterative Monte Carlo search via paraphrasing
        # (Zhou et al., ICLR 2023, §3.3) — 3 rounds of top-5 → 25 paraphrased variants
        num_mc_rounds = 3
        logger.info(
            f"APE iterative Monte Carlo search: {num_mc_rounds} rounds "
            "(paraphrasing top-5 → 25 variants per round)"
        )

        for mc_round in range(num_mc_rounds):
            # Select top-K from current candidate pool by fitness
            top_k = sorted(
                [p for p in candidates if p.fitness is not None],
                key=lambda p: p.fitness,
                reverse=True
            )[:5]

            if not top_k:
                logger.warning(f"APE MC round {mc_round+1}: top_k is empty, skipping round")
                continue

            logger.info(
                f"APE MC round {mc_round+1}/{num_mc_rounds}: "
                f"paraphrasing top-{len(top_k)} (best fitness={top_k[0].fitness:.4f})"
            )

            # Generate paraphrased variants of the top prompts (§3.3)
            variants = self._monte_carlo_resample(top_k, num_variants_per_prompt=5)

            if not variants:
                logger.warning(f"APE MC round {mc_round+1}: no variants produced, skipping eval")
                continue

            # Evaluate variants on val set in parallel
            def _eval_variant(p):
                try:
                    p.fitness = self.evaluator.evaluate_prompt(
                        prompt=p, dataset_name=self.dataset_name, data=val_data
                    )
                except Exception as e:
                    logger.warning(f"APE MC variant eval failed: {e}")
                    p.fitness = 0.0
                return p

            eval_workers = max(1, min(8, len(variants)))
            with ThreadPoolExecutor(max_workers=eval_workers) as executor:
                list(executor.map(_eval_variant, variants))

            candidates.extend(variants)
            best_variant_fitness = max((p.fitness for p in variants), default=0.0)
            logger.info(
                f"APE MC round {mc_round+1}: {len(variants)} variants generated, "
                f"best variant fitness={best_variant_fitness:.4f}, "
                f"total candidates={len(candidates)}"
            )

        logger.info(f"Total candidates after MC search: {len(candidates)}")

        # Step 4: Select best instruction
        if not candidates:
            logger.error("APE: no candidates generated, returning fallback prompt")
            return Prompt(text=f"Solve the following {self.dataset_name} task.", id=0,
                         fitness=0.0, metadata={'method': 'APE', 'fallback': True})
        best_prompt = max(candidates, key=lambda p: p.fitness)
        logger.info(f"Best prompt fitness: {best_prompt.fitness:.4f}")
        logger.info(f"Best prompt text: {best_prompt.text}")

        # Step 4: Optional test evaluation
        if test_data:
            logger.info("Evaluating best prompt on test set...")
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

    def _build_generation_prompt(self, demos: List[Dict]) -> str:
        """Build forward-mode generation prompt from demos.

        Оригинальный forward mode шаблон из Zhou et al., ICLR 2023 (arXiv:2211.01910) (§3.1).
        """
        demo_str = self._format_demos(demos)
        return (
            "I gave a friend an instruction and some input-output pairs. "
            "Based on the instruction, they produced the following input-output pairs:\n\n"
            f"{demo_str}\n\n"
            "The instruction was (reply with ONLY the instruction text, nothing else):"
        )

    def _parse_candidate(self, response: str) -> Optional[str]:
        """Parse LLM response into a candidate instruction."""
        if response is None:
            return None
        instruction = response.strip().rstrip('"').strip()
        if '\n' in instruction:
            instruction = instruction.split('\n')[0].strip()
        instruction = instruction.replace("**", "").replace("*", "")
        instruction = " ".join(instruction.split())
        if len(instruction) < 10:
            return None
        if instruction.startswith("Input 1:") or instruction.startswith("Input:"):
            return None
        return instruction

    def _generate_candidates(self, train_data: List[Dict]) -> List[Prompt]:
        """
        Generate M instruction candidates using forward mode generation.

        Forward Mode: "I gave a friend an instruction. Based on the instruction
        they produced the following input-output pairs: [DEMOS]. The instruction was to"

        Используем multiple demo subsamples (Zhou et al. §3.1): каждая подгруппа
        кандидатов генерируется с разным набором демо-примеров.

        Args:
            train_data: Training examples for demos

        Returns:
            List of candidate prompts
        """
        # Multiple demo subsamples для diversity (оригинал Zhou et al.)
        num_subsamples = 5

        # Pre-generate demo sets and generation prompts
        subsample_prompts = []
        for s in range(num_subsamples):
            demos = random.sample(train_data, min(self.num_demos, len(train_data)))
            gen_prompt = self._build_generation_prompt(demos)
            subsample_prompts.append(gen_prompt)

        logger.info(f"Generation prompt (first 500 chars):\n{subsample_prompts[0][:500]}")

        # Generate M candidates in parallel (with extra attempts for invalid ones)
        max_attempts = self.num_prompts * 3  # Allow some retries

        def _generate_one_candidate(attempt_idx):
            """Generate one candidate instruction via LLM."""
            # Выбираем subsample по индексу
            subsample_idx = attempt_idx % num_subsamples
            generation_prompt = subsample_prompts[subsample_idx]

            response = self.llm_client.generate(
                prompt=generation_prompt,
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=100,
                stop=['"', '\n\n']
            )
            return self._parse_candidate(response)

        # Launch all attempts in parallel
        candidates = []
        max_workers = min(32, max_attempts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_generate_one_candidate, i) for i in range(max_attempts)]
            for future in as_completed(futures):
                if len(candidates) >= self.num_prompts:
                    break
                instruction = future.result()
                if instruction is None:
                    continue
                if len(candidates) < 3:
                    logger.info(f"Candidate {len(candidates)}: {instruction}")
                prompt = Prompt(
                    text=instruction,
                    id=len(candidates),
                    fitness=0.0,
                    metadata={
                        'method': 'APE',
                        'generation_mode': 'forward',
                        'num_demos': self.num_demos
                    }
                )
                candidates.append(prompt)

        if len(candidates) < self.num_prompts:
            logger.warning(
                f"Only generated {len(candidates)}/{self.num_prompts} valid candidates "
                f"after {max_attempts} parallel attempts"
            )

        return candidates

    def _monte_carlo_resample(
        self,
        top_prompts: List[Prompt],
        num_variants_per_prompt: int = 5,
    ) -> List[Prompt]:
        """Monte Carlo search via paraphrasing (Zhou et al., ICLR 2023, §3.3).

        Для каждого из top-K промптов генерируем paraphrased варианты,
        которые сохраняют смысл, но меняют формулировку. Это paraphrasing,
        а не re-generation — именно так описано в оригинальной статье APE (§3.3).

        Args:
            top_prompts: Top-K prompts (by fitness) to paraphrase.
            num_variants_per_prompt: Number of paraphrased variants per prompt.

        Returns:
            List of fresh (unevaluated) Prompt variants. Caller must evaluate.
        """
        if not top_prompts:
            logger.warning("APE paraphrase: empty top_prompts, returning []")
            return []

        paraphrase_template = (
            "Generate a variation of the following instruction that preserves "
            "its meaning while using different wording:\n\n"
            "Original instruction: {original_prompt}\n\n"
            "Variation:"
        )

        tasks = []
        for parent in top_prompts:
            for v in range(num_variants_per_prompt):
                tasks.append((parent, v))

        logger.info(
            f"APE paraphrase: generating {len(tasks)} variants "
            f"({len(top_prompts)} prompts × {num_variants_per_prompt} variants)"
        )

        def _gen_paraphrase(args):
            parent_prompt, var_idx = args
            paraphrase_prompt = paraphrase_template.format(
                original_prompt=parent_prompt.text
            )
            try:
                response = self.llm_client.generate(
                    prompt=paraphrase_prompt,
                    model=self.model,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=200,
                )
                variant_text = self._parse_candidate(response)
                if not variant_text or len(variant_text) < 10:
                    return None
                return Prompt(
                    text=variant_text,
                    id=1000 + hash((parent_prompt.id, var_idx)) % 100000,
                    fitness=0.0,
                    metadata={
                        'method': 'APE',
                        'mc_operation': 'paraphrase',
                        'resampled_from': parent_prompt.id,
                    },
                )
            except Exception as e:
                logger.warning(f"APE MC paraphrase failed: {e}")
                return None

        all_variants: List[Prompt] = []
        max_workers = max(1, min(16, len(tasks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for variant in executor.map(_gen_paraphrase, tasks):
                if variant is not None:
                    all_variants.append(variant)

        logger.info(
            f"APE paraphrase: {len(all_variants)}/{len(tasks)} valid variants produced"
        )
        return all_variants

    def _format_demos(self, demos: List[Dict]) -> str:
        """
        Format demonstration examples for generation template.

        Args:
            demos: List of demo examples

        Returns:
            Formatted demo string
        """
        formatted_demos = []

        for i, demo in enumerate(demos, 1):
            # Dataset-specific formatting
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
                    f"Input {i}: {' '.join(demo['concepts'])}\n"
                    f"Output {i}: {demo['target']}"
                )
            elif self.dataset_name == 'XSum':
                formatted_demos.append(
                    f"Input {i}: {demo['document'][:100]}...\n"
                    f"Output {i}: {demo['summary']}"
                )
            else:
                # Generic formatting
                formatted_demos.append(
                    f"Input {i}: {demo.get('input', demo.get('question', ''))}\n"
                    f"Output {i}: {demo.get('output', demo.get('answer', ''))}"
                )

        return "\n\n".join(formatted_demos)

