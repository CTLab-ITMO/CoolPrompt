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
from coolprompt.utils.enums import PD_Method, Task
from coolprompt.utils.correction.corrector import correct
from coolprompt.utils.correction.rule import LanguageRule

from coolprompt.optimizer.autoprompting_method import AutoPromptingMethod


class PromptTuner:
    """Prompt optimization tool supporting multiple methods.

    This class provides a unified interface to run various prompt
    optimization algorithms (HyPER Light, HyPER, Reflective, Distill, Compress,
    ReGPS, RIDER) on a target language model. It handles dataset splitting, metric
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
        corner_ratio: float = 0.4,
        llm_as_judge_criteria: str | list[str] = "relevance",
        llm_as_judge_custom_templates: Optional[dict[str, str]] = None,
        llm_as_judge_metric_ceil: int = 10,
        bertscore_model_type: Optional[str] = None,
        geval_criteria: Optional[str] = None,
        geval_evaluation_steps: Optional[list[str]] = None,
        geval_evaluation_params: Optional[list] = None,
        geval_strict_mode: bool = False,
        return_final_prompt: bool = True,
        hyper_meta_info: dict = None,
        system_model_as_optimizer: bool = False,
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
                "bertscore" for generation. Special metrics `llm_as_judge` and
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
            bertscore_model_type (str | None): Optional HF ``model_type`` for
                `bertscore` and `multiref_bertscore`.
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
            hyper_meta_info (dict | None): Optional extra key-value pairs
                merged into the meta-info block for ``hyper`` and ``hyper_light``.
            system_model_as_optimizer (bool): If True, use the system model for
                optimizing processes, while target model will be used for inference.
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
            bertscore_model_type=bertscore_model_type,
            geval_criteria=geval_criteria,
            geval_evaluation_steps=geval_evaluation_steps,
            geval_evaluation_params=geval_evaluation_params,
            geval_strict_mode=geval_strict_mode,
        )
        metric_name = base_metric._get_name()
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
                corner_ratio=corner_ratio,
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
        logger.info(f"Metric: {base_metric}, Validation size: {validation_size}")
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

        if hyper_meta_info is not None:
            kwargs = {**kwargs, "meta_info": hyper_meta_info}

        final_prompt = method_impl.optimize(
            model=self._system_model if system_model_as_optimizer else self._target_model,
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
            f"Initial {base_metric} score: {self.init_metric}, "
            f"final {base_metric} score: {self.final_metric}"
        )

        self.init_prompt = start_prompt
        self.final_prompt = final_prompt

        logger.info("=== Prompt Optimization Completed ===")

        return final_prompt if return_final_prompt else None

    def test(
        self,
        dataset: Iterable[str],
        prompt: Optional[str] = None,
        task: Optional[str] = None,
        targets: Optional[Iterable[str | int]] = None,
        metric: Optional[str] = None,
        bertscore_model_type: Optional[str] = None,
        batch_size: int = 25,
        return_raw_outputs: bool = True,
    ) -> List[str] | Tuple[List[str], float]:
        """
        Generate model predictions for a test dataset and optionally compute a metric.

        For each sample in the dataset,
        the prompt is formatted with the sample using the task template,
        passed to the model to generate an output,
        and all outputs are collected. If targets are provided,
        the metric is computed and returned alongside the outputs.

        Args:
            dataset (Iterable[str]): Input samples to process.
            prompt (Optional[str]): Prompt to use. If None, falls back to self.final_prompt.
            task (Optional[str]): Task type ("classification" or "generation").
                If None, auto-detected via TaskDetector.
            targets (Optional[Iterable[str|int]]): Ground truth labels.
                If provided, metric is computed and returned.
            metric (Optional[str]): Metric name. If None, defaults to "accuracy"
                for classification or "bertscore" for generation.
            bertscore_model_type (Optional[str]): Optional HF ``model_type``
                for `bertscore` and `multiref_bertscore`.
            batch_size (int, default=25): Number of samples per inference batch.
            return_raw_outputs (bool, default=True): If True, return raw model outputs;
                if False, return parsed outputs via metric.parse_output().

        Returns:
            If targets is None: List[str] of raw/parsed outputs.
            If targets provided: Tuple[List[str], float] of outputs and aggregate metric score.

        Raises:
            ValueError: If no prompt is available or task cannot be determined.
        """
        use_prompt = prompt if prompt is not None else self.final_prompt
        if use_prompt is None:
            raise ValueError(
                "No prompt provided and self.final_prompt is not set. "
                "Either call .run() first or pass prompt explicitly."
            )
        
        if task is None:
            task_detector = TaskDetector(self._system_model)
            task = task_detector.generate(use_prompt)
        
        task_str = task.lower()
        if task_str not in ("classification", "generation"):
            raise ValueError("task must be 'classification' or 'generation'.")
        
        task_enum = Task.CLASSIFICATION if task_str == "classification" else Task.GENERATION
        
        if metric is None:
            metric = "accuracy" if task_enum == Task.CLASSIFICATION else "meteor"
        
        metric_impl = validate_and_create_metric(
            task_enum,
            metric,
            bertscore_model_type=bertscore_model_type,
        )
        
        evaluator = Evaluator(
            model=self._target_model,
            task=task_enum,
            metric=metric_impl,
            batch_size=batch_size,
        )
        
        dataset_list = list(dataset)
        use_targets = list(targets) if targets is not None else [""] * len(dataset_list)
        
        result = evaluator.evaluate(
            prompt=use_prompt,
            dataset=dataset_list,
            targets=use_targets,
            template=None,
            return_detailed=True,
        )
        
        outputs = result.raw_outputs if return_raw_outputs else [
            metric_impl.parse_output(a) for a in result.raw_outputs
        ]
        
        if targets is not None:
            return outputs, result.aggregate_score
        return outputs
