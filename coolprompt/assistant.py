from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from random import sample
from langchain_core.language_models.base import BaseLanguageModel
from sklearn.model_selection import train_test_split

from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.task_detector.detector import TaskDetector
from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.utils.logging_config import logger, set_verbose, setup_logging
from coolprompt.utils.var_validation import (
    validate_model,
    validate_run,
    validate_verbose,
)
from coolprompt.utils.enums import PD_Method

from coolprompt.utils.correction.corrector import correct
from coolprompt.utils.correction.rule import LanguageRule

from coolprompt.optimizer.autoprompting_method import AutoPromptingMethod
from coolprompt.evaluator import Evaluator, validate_and_create_metric


class PromptTuner:
    """Prompt optimization tool supporting multiple methods.

    This class provides a unified interface to run various prompt
    optimization algorithms (HyPER Light, HyPER, Reflective, Distill, Compress, ReGPS)
    on a target language model. It handles dataset splitting, metric
    evaluation, logging, and optional synthetic data generation.
    """

    NUMBER_OF_EXAMPLES_FOR_DATASET_BASED_PD_METHOD = 5
    """Number of input‑output examples used when generating a problem
    description from the dataset (only for DATASET_BASED method)."""

    def __init__(
        self,
        target_model: BaseLanguageModel = None,
        system_model: BaseLanguageModel = None,
        logs_dir: str | Path = None,
    ) -> None:
        """Initialize the PromptTuner with language models and logging.

        Args:
            target_model (BaseLanguageModel): LangChain model used for
                prompt optimization (e.g., inference during evaluation).
                If None, a default LLM is created.
            system_model (BaseLanguageModel): LangChain model used for
                auxiliary tasks (synthetic data generation, feedback,
                task detection, etc.). If None, it defaults to
                `target_model`.
            logs_dir (str | Path, optional): Directory where log files
                will be stored. If None, logs are not written to disk.
        """
        setup_logging(logs_dir)
        self._target_model = target_model or DefaultLLM.init()
        self._system_model = system_model or self._target_model

        self.init_metric = None
        self.init_prompt = None
        self.final_metric = None
        self.final_prompt = None
        self.assistant_feedback = None

        self.synthetic_dataset = None
        self.synthetic_target = None

        logger.info("Validating the target model")
        validate_model(self._target_model)

        if self._system_model is not self._target_model:
            logger.info("Validating the system model")
            validate_model(self._system_model)

        logger.info("PromptTuner successfully initialized")

    def get_stats(self):
        """Retrieve usage statistics from the target model, if available.

        Returns:
            Any statistics object returned by the target model's
            `get_stats()` method, or None if the model does not provide it.
        """
        if hasattr(self._target_model, "get_stats"):
            return self._target_model.get_stats()
        return None

    def reset_stats(self):
        """Reset usage statistics of the target model, if supported."""
        if hasattr(self._target_model, "reset_stats"):
            self._target_model.reset_stats()

    def _get_dataset_split(
        self,
        dataset: Iterable[str],
        target: Iterable[str],
        validation_size: float,
        train_as_test: bool,
    ) -> Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
        """Split the dataset into training and validation sets.

        Args:
            dataset (Iterable[str]): Input texts.
            target (Iterable[str]): Corresponding labels/targets.
            validation_size (float): Fraction of data to use for validation.
            train_as_test (bool): If True, use the full dataset as both
                train and validation (ignoring `validation_size`).

        Returns:
            Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
                A tuple (train_data, val_data, train_targets, val_targets).
        """
        if train_as_test:
            return (dataset, dataset, target, target)
        train_data, val_data, train_targets, val_targets = train_test_split(
            dataset, target, test_size=validation_size
        )
        return (train_data, val_data, train_targets, val_targets)

    def run(
        self,
        start_prompt: str,
        task: Optional[str] = None,
        dataset: Optional[Iterable[str]] = None,
        target: Optional[Iterable[str] | Iterable[int]] = None,
        method: str | AutoPromptingMethod | type[AutoPromptingMethod] = "hyper_light",
        metric: Optional[str] = None,
        problem_description: Optional[str] = None,
        problem_description_generation_method: str = "base",
        validation_size: float = 0.25,
        train_as_test: bool = False,
        generate_num_samples: int = 10,
        batch_size: int = 25,
        verbose: int = 1,
        llm_as_judge_criteria: str | list[str] = "relevance",
        llm_as_judge_custom_templates: Optional[dict[str, str]] = None,
        llm_as_judge_metric_ceil: int = 10,
        geval_criteria: Optional[str] = None,
        geval_evaluation_steps: Optional[list[str]] = None,
        geval_evaluation_params: Optional[list] = None,
        geval_strict_mode: bool = False,
        return_final_prompt: bool = True,
        meta_prompt_context: dict = None,
        **kwargs,
    ) -> Optional[str]:
        """Run prompt optimization using the selected method.

        This method orchestrates task detection, dataset preparation,
        problem description generation, evaluation, and the actual
        optimization loop.

        Args:
            start_prompt (str): Initial prompt text to optimize.
            task (str | None): Type of task – "classification" or "generation".
                If None, it will be auto‑detected from `start_prompt`.
            dataset (Iterable[str] | None): Input dataset (list of strings)
                for training/validation. Required for data‑driven methods.
            target (Iterable[str] | Iterable[int] | None): Target labels
                corresponding to the dataset. Required if `dataset` is given.
            method (str | AutoPromptingMethod | type[AutoPromptingMethod]):
                Registered name (e.g. ``hyper_light``), an instance, or a concrete subclass
                (constructed inside ``validate_method`` with no arguments).
            metric (str | None): Evaluation metric name.
                If None, defaults to "f1" for classification,
                "meteor" for generation. Special metrics `llm_as_judge` and
                `geval` require additional configuration parameters below.
            problem_description (str | None): Natural language description
                of the task. If None, it will be generated automatically
                according to `problem_description_generation_method`.
            problem_description_generation_method (str): Method to generate
                the problem description: "base" (prompt‑only) or
                "dataset‑based" (uses examples from the dataset).
            validation_size (float): Fraction of the dataset to hold out
                for validation (0.0 to 1.0). Ignored if `train_as_test` True.
            train_as_test (bool): If True, the entire dataset is used for
                both training and validation (no split).
            generate_num_samples (int): Number of synthetic samples to
                generate when no dataset is provided.
            batch_size (int): Number of examples processed in one batch
                during evaluation.
            verbose (int): Logging verbosity: 0 = silent, 1 = steps,
                2 = steps + prompts.
            llm_as_judge_criteria (str | list[str]): Criterion or list of
                criteria for the LLM‑as‑judge metric.
            llm_as_judge_custom_templates (dict[str, str] | None): Custom
                prompt templates for each criterion.
            llm_as_judge_metric_ceil (int): Maximum integer score expected
                from the judge (1..ceil); normalized to [0,1].
            geval_criteria (str | None): High‑level description for GEval.
                Mutually exclusive with `geval_evaluation_steps`.
            geval_evaluation_steps (list[str] | None): Step‑by‑step
                instructions for GEval.
            geval_evaluation_params (list | None): GEval evaluation
                parameters (LLMTestCaseParams). If None, default is
                [INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT].
            geval_strict_mode (bool): If True, GEval uses strict binary
                pass/fail with threshold forced to 1.
            return_final_prompt (bool): If True, return the final prompt;
                otherwise return None (the prompt is still stored in
                `self.final_prompt`).
            meta_prompt_context (dict | None): Optional extra key-value pairs
                merged into the meta-info block for ``hyper_light`` (same role as
                ``config['meta_info']`` in YAML benchmarks).
            **kwargs: Additional arguments passed to the optimization method.

        Returns:
            Optional[str]: The optimized prompt if `return_final_prompt` is
            True, otherwise None.

        Raises:
            ValueError: On invalid task, missing required dataset for
                data‑driven methods, length mismatch between dataset and
                target, or missing problem description when required.
        """
        if verbose is not None:
            validate_verbose(verbose)
            set_verbose(verbose)

        task_detector = TaskDetector(self._system_model)
        if task is None:
            task = task_detector.generate(start_prompt)

        logger.info("Validating args for PromptTuner running")

        task_value, method_impl, pd_method = validate_run(
            start_prompt,
            task,
            dataset,
            target,
            method,
            problem_description,
            problem_description_generation_method,
            validation_size,
        )

        base_metric = validate_and_create_metric(
            task_value,
            metric,
            model=(
                self._system_model
                if metric in ("llm_as_judge", "geval")
                else None
            ),
            llm_as_judge_criteria=llm_as_judge_criteria,
            llm_as_judge_custom_templates=llm_as_judge_custom_templates,
            llm_as_judge_metric_ceil=llm_as_judge_metric_ceil,
            geval_criteria=geval_criteria,
            geval_evaluation_steps=geval_evaluation_steps,
            geval_evaluation_params=geval_evaluation_params,
            geval_strict_mode=geval_strict_mode,
        )
        evaluator = Evaluator(
            self._target_model, task_value, base_metric, batch_size=batch_size
        )
        final_prompt = ""
        generator = SyntheticDataGenerator(self._system_model)

        if dataset is None:
            dataset, target, problem_description = generator.generate(
                prompt=start_prompt,
                task=task_value,
                problem_description=problem_description,
                num_samples=generate_num_samples,
            )
            self.synthetic_dataset = dataset
            self.synthetic_target = target

        dataset_split = self._get_dataset_split(
            dataset=dataset,
            target=target,
            validation_size=validation_size,
            train_as_test=train_as_test,
        )

        if problem_description is None:
            if pd_method is PD_Method.BASE:
                problem_description = generator._generate_problem_description(
                    prompt=start_prompt
                )
            elif pd_method is PD_Method.DATASET_BASED:
                k = min(
                    self.NUMBER_OF_EXAMPLES_FOR_DATASET_BASED_PD_METHOD,
                    len(dataset_split[0]),
                )
                indices = sample(range(len(dataset_split[0])), k)
                examples = [
                    (dataset_split[0][ind], dataset_split[2][ind])
                    for ind in indices
                ]
                problem_description = generator._generate_problem_description(
                    prompt=start_prompt, examples=examples
                )

        logger.info("=== Starting Prompt Optimization ===")
        logger.info(f"Method: {method_impl.name}, Task: {task}")
        logger.info(f"Metric: {metric}, Validation size: {validation_size}")
        if dataset:
            logger.info(f"Dataset: {len(dataset)} samples")
        else:
            logger.info("No dataset provided")
        if target:
            logger.info(f"Target: {len(target)} samples")
        else:
            logger.info("No target provided")
        if kwargs:
            logger.debug(f"Additional kwargs: {kwargs}")

        if meta_prompt_context is not None:
            kwargs = {**kwargs, "meta_prompt_context": meta_prompt_context}

        final_prompt = method_impl.optimize(
            model=self._target_model,
            initial_prompt=start_prompt,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            **kwargs,
        )

        logger.info("Running the prompt format checking...")
        final_prompt = correct(
            prompt=final_prompt,
            rule=LanguageRule(self._system_model),
            start_prompt=start_prompt,
        )

        logger.debug(f"Final prompt:\n{final_prompt}")

        template = method_impl.get_template(task_value)

        logger.info(f"Evaluating on given dataset for {task} task...")
        self.init_metric = evaluator.evaluate(
            prompt=start_prompt,
            dataset=dataset_split[1],
            targets=dataset_split[3],
            template=template,
        )
        self.final_metric = evaluator.evaluate(
            prompt=final_prompt,
            dataset=dataset_split[1],
            targets=dataset_split[3],
            template=template,
        )
        logger.info(
            f"Initial {metric} score: {self.init_metric}, "
            f"final {metric} score: {self.final_metric}"
        )

        self.init_prompt = start_prompt
        self.final_prompt = final_prompt

        logger.info("=== Prompt Optimization Completed ===")

        return final_prompt if return_final_prompt else None

    def test(
        self,
        dataset: Iterable[str],
        task: str,
        batch_size: int = 25,
    ) -> List[str]:
        """
        Generate model predictions for a test dataset using final_prompt.

        For each sample in the dataset,
        the final prompt is formatted with the sample using the task template,
        passed to the model to generate an output,
        and all outputs are collected and returned as strings.

        Args:
            dataset (Iterable[str]): Input samples to process.
            task (str): Task type ("classification" or "generation").
            batch_size (int, default=25): Number of samples per inference batch.

        Returns:
            List[str]: Raw model outputs, one per input sample, in order.

        Raises:
            ValueError: If final_prompt is not set or task is missing.
        """
        if self.final_prompt is None:
            raise ValueError("Final prompt is not set. Call .run() first.")

        task_str = task.lower()
        if task_str not in ("classification", "generation"):
            raise ValueError("task must be 'classification' or 'generation'.")

        from coolprompt.evaluator import Evaluator, validate_and_create_metric
        from coolprompt.utils.enums import Task

        task_enum = Task.CLASSIFICATION if task_str == "classification" else Task.GENERATION
        metric_name = "accuracy" if task_enum == Task.CLASSIFICATION else "meteor"
        metric = validate_and_create_metric(task_enum, metric_name)

        evaluator = Evaluator(
            model=self._target_model,
            task=task_enum,
            metric=metric,
            batch_size=batch_size,
        )

        dataset_list = list(dataset)
        dummy_targets = [""] * len(dataset_list)

        result = evaluator.evaluate(
            prompt=self.final_prompt,
            dataset=dataset_list,
            targets=dummy_targets,
            template=None,
            return_detailed=True,
        )

        return result.raw_outputs