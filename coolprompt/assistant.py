from pathlib import Path
from typing import Iterable, Optional, Tuple
from langchain_core.language_models.base import BaseLanguageModel
from sklearn.model_selection import train_test_split

from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.task_detector.detector import TaskDetector
from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.language_model.llm import DefaultLLM
from coolprompt.optimizer.hype import hype_optimizer
from coolprompt.optimizer.reflective_prompt import reflectiveprompt
from coolprompt.optimizer.distill_prompt.run import distillprompt
from coolprompt.utils.logging_config import logger, set_verbose, setup_logging
from coolprompt.utils.var_validation import (
    validate_model,
    validate_task,
    validate_method,
    validate_run,
    validate_verbose,
)
from coolprompt.utils.enums import Method, Task
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)
from coolprompt.utils.correction.corrector import correct
from coolprompt.utils.correction.rule import LanguageRule
from coolprompt.prompt_assistant.prompt_assistant import PromptAssistant


class PromptTuner:
    """Prompt optimization tool supporting multiple methods."""

    TEMPLATE_MAP = {
        (Task.CLASSIFICATION, Method.HYPE): CLASSIFICATION_TASK_TEMPLATE_HYPE,
        (Task.CLASSIFICATION, Method.REFLECTIVE): CLASSIFICATION_TASK_TEMPLATE,
        (Task.CLASSIFICATION, Method.DISTILL): CLASSIFICATION_TASK_TEMPLATE,
        (Task.GENERATION, Method.HYPE): GENERATION_TASK_TEMPLATE_HYPE,
        (Task.GENERATION, Method.REFLECTIVE): GENERATION_TASK_TEMPLATE,
        (Task.GENERATION, Method.DISTILL): GENERATION_TASK_TEMPLATE,
    }

    def __init__(
        self,
        target_model: BaseLanguageModel = None,
        system_model: BaseLanguageModel = None,
        logs_dir: str | Path = None,
    ) -> None:
        """Initializes the tuner with a LangChain-compatible language model.

        Args:
            target_model (BaseLanguageModel): Any LangChain BaseLanguageModel
                instance which supports invoke(str) -> str. Used for
                optimization processes. Will use DefaultLLM if not provided.
            system_model (BaseLanguageModel): Any LangChain BaseLanguageModel
                instance which supports invoke(str) -> str. Used for
                synthetic data generation, feedback generation, etc.
                Will use the `target_model` if not provided.
            logs_dir (str | Path, optional): logs saving directory.
                Defaults to None.
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

    def get_task_prompt_template(self, task: str, method: str) -> str:
        """Returns the prompt template for the given task.

        Args:
            task (str):
                The type of task, either "classification" or "generation".
            method (str):
                Optimization method to use.
                Available methods are: ['hype', 'reflective', 'distill']

        Returns:
            str: The prompt template for the given task.
        """

        logger.debug(
            f"Getting prompt template for {task} task and {method} method"
        )
        task = validate_task(task)
        method = validate_method(method)
        return self.TEMPLATE_MAP[(task, method)]

    def _get_dataset_split(
        self,
        dataset: Iterable[str],
        target: Iterable[str],
        validation_size: float,
        train_as_test: bool,
    ) -> Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
        """Provides a train/val dataset split.

        Args:
            dataset (Iterable[str]):
                Provided dataset.
            target (Iterable[str]):
                Provided targets for the dataset.
            validation_size (float):
                Provided size of validation subset.
            train_as_test (bool):
                Either to use all data for train and validation or split it.

        Returns:
            Tuple[Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
                a tuple of train dataset, validation dataset,
                train targets and validation targets.
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
        task: str = None,
        dataset: Optional[Iterable[str]] = None,
        target: Optional[Iterable[str] | Iterable[int]] = None,
        method: str = "hype",
        metric: Optional[str] = None,
        problem_description: Optional[str] = None,
        validation_size: float = 0.25,
        train_as_test: bool = False,
        generate_num_samples: int = 10,
        feedback: bool = True,
        verbose: int = 1,
        llm_as_judge_criteria: str | list[str] = "relevance",
        llm_as_judge_custom_templates: Optional[dict[str, str]] = None,
        llm_as_judge_metric_ceil: int = 10,
        geval_criteria: Optional[str] = None,
        geval_evaluation_steps: Optional[list[str]] = None,
        geval_evaluation_params: Optional[list] = None,
        geval_strict_mode: bool = False,
        return_final_prompt: bool = True,
        **kwargs,
    ) -> str:
        """Optimizes prompts using provided model.

        Args:
            start_prompt (str): Initial prompt text to optimize.
            task (str):
                Type of task to optimize for (classification or generation).
                Defaults to generation.
            dataset (Iterable):
                Dataset iterable object for autoprompting optimization.
            target (Iterable):
                Target iterable object for autoprompting optimization.
            method (str): Optimization method to use.
                Available methods are: ['hype', 'reflective', 'distill']
                Defaults to hype.
            metric (str): Metric to use for optimization.
            problem_description (str): a string that contains
                short description of problem to optimize.
            validation_size (float):
                A float that must be between 0.0 and 1.0 and
                represent the proportion of the dataset
                to include in the validation split.
                Defaults to 0.25.
            train_as_test (bool):
                Either to use all the provided data as
                the train and the test dataset at the same time or not.
                If sets to True, the validation_size parameter will be ignored.
                Defaults to False.
            generate_num_samples (int):
                A number of dataset and target samples to generate with PromptAssistant
            feedback (bool):
                PromptAssistant interpretation of optimization results
                Defaults to True.
            verbose (int): Parameter for logging configuration:
                0 - no logging
                1 - steps logging
                2 - steps and prompts logging
            llm_as_judge_criteria (str | list[str]): Criteria for LLM-as-judge metric when
                metric == 'llm_as_judge'. Accepts a single criterion (e.g., "relevance")
                or a list of criteria (e.g., ["relevance", "fluency"]). Built‑in
                keys: "accuracy", "coherence", "fluency", "relevance". Custom
                names are supported when paired with `llm_as_judge_custom_templates`.
            llm_as_judge_custom_templates (dict[str, str] | None): Optional mapping
                from criterion name to a custom judge prompt template. Each
                template must include placeholders: `{metric_ceil}`, `{request}`
                and `{response}`; the judge must return ONLY a single number.
            llm_as_judge_metric_ceil (int): Maximum integer score expected from the
                judge (1..ceil). Judge outputs are clipped to [0, ceil] and
                normalized to [0, 1] for averaging.
            geval_criteria (str | None): High-level natural language description
                of what GEval should evaluate. Mutually exclusive with
                `geval_evaluation_steps`. If both are provided, GEvalMetric
                will raise a ValueError.
            geval_evaluation_steps (list[str] | None): Explicit step-by-step
                instructions for GEval. If provided, `geval_criteria` must be
                None.
            geval_evaluation_params (list | None): Optional list of
                LLMTestCaseParams controlling which fields of each
                LLMTestCase are visible to GEval. Defaults to
                [INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT] inside GEvalMetric
                when left as None.
            geval_strict_mode (bool): When True, GEval behaves in strict mode
                (binary pass/fail with threshold forced to 1).
            **kwargs (dict[str, Any]): other key-word arguments.

        Returns:
            final_prompt: str - The resulting optimized prompt
            after applying the selected method.

        Raises:
            ValueError: If an invalid task type is provided.
            ValueError: If a problem description is not provided
                for ReflectivePrompt.

        Note:
            Uses HyPE optimization
            when dataset or method parameters are not provided.

            Uses default metric for the task type
            if metric parameter is not provided:
            f1 for classisfication, meteor for generation.

            if dataset is not None, you can find evaluation results
            in self.init_metric and self.final_metric
        """
        if verbose is not None:
            validate_verbose(verbose)
            set_verbose(verbose)

        task_detector = TaskDetector(self._system_model)
        if task is None:
            task = task_detector.generate(start_prompt)

        logger.info("Validating args for PromptTuner running")
        task, method = validate_run(
            start_prompt,
            task,
            dataset,
            target,
            method,
            problem_description,
            validation_size,
        )
        metric = validate_and_create_metric(
            task,
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
        evaluator = Evaluator(self._target_model, task, metric)
        final_prompt = ""
        generator = SyntheticDataGenerator(self._system_model)

        if dataset is None:
            dataset, target, problem_description = generator.generate(
                prompt=start_prompt,
                task=task,
                problem_description=problem_description,
                num_samples=generate_num_samples
            )
            self.synthetic_dataset = dataset
            self.synthetic_target = target

        if problem_description is None:
            problem_description = generator._generate_problem_description(
                prompt=start_prompt
            )

        dataset_split = self._get_dataset_split(
            dataset=dataset,
            target=target,
            validation_size=validation_size,
            train_as_test=train_as_test,
        )

        logger.info("=== Starting Prompt Optimization ===")
        logger.info(f"Method: {method}, Task: {task}")
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

        if method is Method.HYPE:
            final_prompt = hype_optimizer(
                model=self._target_model,
                prompt=start_prompt,
                problem_description=problem_description,
            )
        elif method is Method.REFLECTIVE:
            final_prompt = reflectiveprompt(
                model=self._target_model,
                dataset_split=dataset_split,
                evaluator=evaluator,
                problem_description=problem_description,
                initial_prompt=start_prompt,
                **kwargs,
            )
        elif method is Method.DISTILL:
            final_prompt = distillprompt(
                model=self._target_model,
                dataset_split=dataset_split,
                evaluator=evaluator,
                initial_prompt=start_prompt,
                **kwargs,
            )

        logger.info("Running the prompt format checking...")
        final_prompt = correct(
            prompt=final_prompt,
            rule=LanguageRule(self._system_model),
            start_prompt=start_prompt,
        )

        logger.debug(f"Final prompt:\n{final_prompt}")
        template = self.TEMPLATE_MAP[(task, method)]
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

        if feedback:
            prompt_assistant = PromptAssistant(self._target_model)
            self.assistant_feedback = correct(
                prompt=prompt_assistant.get_feedback(
                    start_prompt, final_prompt
                ),
                rule=LanguageRule(self._system_model),
                start_prompt=start_prompt,
            )

            logger.info("=== Assistant's feedback ===")
            logger.info(self.assistant_feedback)

        return final_prompt if return_final_prompt else None
