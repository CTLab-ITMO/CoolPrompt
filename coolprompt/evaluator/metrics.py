from abc import ABC, abstractmethod
import re
from typing import Optional, Dict, List, Tuple

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from evaluate import load
from code_bert_score import BERTScorer
import numpy as np
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage

from coolprompt.language_model.deepeval_model import DeepEvalLangChainModel
from coolprompt.utils.arithmetics import (
    clip,
    extract_number_from_text,
    mean,
)
from coolprompt.utils.enums import Task
from coolprompt.utils.language_detection import detect_language
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_answer
from coolprompt.utils.prompt_templates.llm_as_judge_templates import (
    ACCURACY_QA_TEMPLATE,
    COHERENCE_TEMPLATE,
    FLUENCY_TEMPLATE,
    RELEVANCE_TEMPLATE,
)
from coolprompt.utils.structured_schemas.evaluator import JudgeScoreResponse


class HFEvaluateMetric(ABC):
    def __init__(self, name: str) -> None:
        """Initialize metric with specified evaluate library metric name.

        Args:
            name (str): Name of metric to load from evaluate library
        """

        self._return_parameter = name
        self._metric = load(name)
        self._compute_kwargs_func = lambda outputs, targets: {}
        super().__init__()

    def _postprocessing(self, metric: float | List[float]) -> float:
        """Unwraps the list when HF metric returns [Metric Value]
            instead of Metric Value.

        Args:
            metric (float | List[float]): Metric Value from HF metric.
        Returns:
            float: Unwrapped value.
        """
        if isinstance(metric, list):
            return metric[0]
        return metric

    def _compute_raw(
        self,
        outputs: list[str | int],
        targets: list[str | int],
        dataset: Optional[list[str]] = None,
    ) -> List[float]:
        """Compute metric values from preprocessed model answers.
        Returs a list of float values corresponding for each answer.

        Args:
            outputs (list[str|int]): Model predictions (text for generation,
            labels for classification)
            targets (list[str|int]): Ground truth labels
        Returns:
            List[float]: List of float metrics (for each model answer).
        """

        return [
            self._postprocessing(
                self._metric.compute(
                    predictions=[output],
                    references=[target],
                    **self._compute_kwargs_func([output], [target]),
                )[self._return_parameter]
            )
            for output, target in zip(outputs, targets)
        ]


class BaseMetric(ABC):
    """Abstract base class for implementing evaluation metrics.

    Provides common infrastructure for loading metrics
    from HuggingFace's evaluate library and defining
    metric computation interfaces.

    Attributes:
        ANS_TAGS: tuple - Start and end tags for answer extraction
        FORMAT_MISMATCH_LABEL: int - Special value indicating parsing failure
    """

    ANS_TAGS = ("<ans>", "</ans>")

    def __init__(self) -> None:
        """Initialize metric"""

        super().__init__()

    @abstractmethod
    def _compute_raw(
        self,
        outputs: list[str | int],
        targets: list[str | int],
        dataset: Optional[list[str]] = None,
    ) -> List[float]:
        """Compute metric values from preprocessed model answers.
        Returs a list of float values corresponding for each answer.

        Args:
            outputs (list[str|int]): Model predictions (text for generation,
            labels for classification)
            targets (list[str|int]): Ground truth labels
        Returns:
            List[float]: List of float metrics (for each model answer).
        """
        pass

    @abstractmethod
    def _encode_labels(
        self, output_labels: list[str | int], targets: list[str | int]
    ) -> tuple[list[int] | list[str], list[int] | list[str]]:
        """Encode labels into internal representation for both
        outputs and targets.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[int], list[int]]: Encoded output labels
            and encoded targets.
        """

        pass

    def _extract_bad_examples(
        self,
        results: List[float],
        dataset: List[str],
        outputs: List[str | int],
        targets: List[str | int],
        failed_examples: int,
    ) -> List[Dict[str, Tuple[str, str]]]:
        """Taking bad examples via processed metrics.

        Args:
            outputs (list[str|int]): Model predictions (text for generation,
            labels for classification)
            targets (list[str|int]): Ground truth labels
        Returns:
            List[float]: List of float metrics (for each model answer).
        """

        indices = np.argsort(results)[:failed_examples]

        return [
            {
                "input": dataset[ind],
                "output": outputs[ind],
                "correct": targets[ind],
            }
            for ind in indices
        ]

    def compute(
        self,
        outputs: list[str | int],
        targets: list[str | int],
        dataset: Optional[list[str]] = None,
        failed_examples: Optional[int] = None,
        return_per_task: bool = False,
    ) -> float | Tuple[float, List[Dict[str, Tuple[str, str]]]] | Tuple[float, List[float], List[Dict]]:
        """Compute metric value from text model outputs.

        Must be implemented by subclasses to handle input formatting.
        Args:
            outputs (list[str|int]): Model predictions (just text)
            targets (list[str|int]): Ground truth labels
            failed_examples (int, Optional): Number of bad examples to return
            return_per_task (bool, False): If True, returns (aggregate, results_per_task, bad_examples)

        Returns:
            float | Tuple[float, List[float], List[Dict]]:
                aggregate score, optionally per-task scores, optionally bad examples
        """
        output_labels = list(
            map(
                lambda x: extract_answer(x, self.ANS_TAGS, self.FORMAT_MISMATCH_LABEL),
                outputs,
            )
        )
        targets = list(map(str, targets))
        encoded_output_labels, encoded_targets = self._encode_labels(
            output_labels, targets
        )

        results = self._compute_raw(
            encoded_output_labels, encoded_targets, dataset
        )

        if results is None or any(r is None for r in results):
            return None

        self._failed_examples_requested = failed_examples
        aggregate = float(np.mean(results))

        if return_per_task:
            bad_examples = self._extract_bad_examples(
                results, dataset, outputs, targets, failed_examples or 0
            ) if failed_examples else []
            return aggregate, results, bad_examples

        if failed_examples is not None and failed_examples > 0:
            bad_examples = self._extract_bad_examples(
                results, dataset, outputs, targets, failed_examples
            )
            return aggregate, bad_examples

        return aggregate

    def parse_output(self, output: str) -> str:
        """Extract parsed answer from model output.

        Args:
            output: Raw model output string.

        Returns:
            Extracted answer from <ans> tags, or original output if not found.
        """
        return extract_answer(output, self.ANS_TAGS, format_mismatch_label=output)

    def __str__(self) -> str:
        return self._get_name()

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return self._get_name() == other._get_name()


class ClassificationMetric(BaseMetric):
    """Base class for classification metrics with answer parsing functionality.

    Handles extraction of labels from model outputs
    containing XML-style <ans> tags
    and label encoding for metric computation.
    """

    FORMAT_MISMATCH_LABEL = -1

    def __init__(self):
        """Initialize metric"""

        super().__init__()
        self.label_to_id = None

    def _encode_labels(
        self, output_labels: list[str | int], targets: list[str | int]
    ) -> tuple[list[int], list[int]]:
        """Encode string labels into integer IDs for both outputs and targets.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[int], list[int]]: Encoded output labels
            and encoded targets.
        """

        if self.label_to_id is None:
            self.extract_labels(targets)

        encoded_output_labels = [
            self.label_to_id[label] if label in self.label_to_id else -1
            for label in output_labels
        ]
        encoded_targets = [self.label_to_id[label] for label in targets]
        return encoded_output_labels, encoded_targets

    def extract_labels(self, targets: list[str | int]) -> None:
        """Extract unique labels from targets and encode them into IDs.

        Args:
            targets (list[str  |  int]): Ground truth labels.
        """

        self.label_to_id = dict()
        for x in targets:
            label = str(x)
            if label not in self.label_to_id:
                self.label_to_id[label] = len(self.label_to_id)


class GenerationMetric(BaseMetric):
    """Base class for generation metrics.

    Provides a generic implementation for metrics that compare generated text
    to reference text.
    """

    FORMAT_MISMATCH_LABEL = ""

    def __init__(self):
        """Initialize metric"""

        super().__init__()

    def _encode_labels(
        self, output_labels: list[str | int], targets: list[str | int]
    ) -> tuple[list[int] | list[str], list[int] | list[str]]:
        """Returns labels without encoding for generation metrics.

        Args:
            output_labels (list[str|int]): Extracted labels from model outputs.
            targets (list[str|int]): Ground truth labels.
        Returns:
            tuple[list[str], list[str]]: input values
        """

        return output_labels, targets


class AccuracyMetric(HFEvaluateMetric, ClassificationMetric):
    """Accuracy metric for classification tasks."""

    @staticmethod
    def _get_name():
        return "accuracy"

    def __init__(self):
        super().__init__(self._get_name())


class F1Metric(HFEvaluateMetric, ClassificationMetric):
    """F1 metric for classification tasks with macro averaging."""

    @staticmethod
    def _get_name():
        return "f1"

    def __init__(self):
        super().__init__(self._get_name())
        self._compute_kwargs_func = lambda outputs, targets: {
            "average": "macro"
        }


class BleuMetric(HFEvaluateMetric, GenerationMetric):
    """BLEU metric for generation tasks."""

    @staticmethod
    def _get_name():
        return "bleu"

    def __init__(self):
        super().__init__(self._get_name())


class RougeMetric(HFEvaluateMetric, GenerationMetric):
    """ROUGE metric for generation tasks."""

    @staticmethod
    def _get_name():
        return "rouge"

    def __init__(self):
        super().__init__(self._get_name())
        self._return_parameter = "rougeL"


class MeteorMetric(HFEvaluateMetric, GenerationMetric):
    """METEOR metric for generation tasks."""

    @staticmethod
    def _get_name():
        return "meteor"

    def __init__(self):
        super().__init__(self._get_name())


class BertScoreMetric(HFEvaluateMetric, GenerationMetric):
    """BertScore metric for generation tasks."""

    @staticmethod
    def _get_name():
        return "bertscore"

    def __init__(self):
        super().__init__(self._get_name())
        self._compute_kwargs_func = lambda outputs, targets: {
            "model_type": "bert-base-multilingual-cased"
        }
        self._return_parameter = "f1"


class LLMAsJudge(GenerationMetric):
    """LLM-as-a-judge metric for generation tasks."""

    @staticmethod
    def _get_name():
        return "llm_as_judge"

    def __init__(
        self,
        model: BaseLanguageModel,
        criteria: str | list[str] = "relevance",
        prompt_template: Optional[str] = None,
        custom_templates: Optional[dict[str, str]] = None,
        metric_ceil: int = 10,
        use_structured_output: bool = False,
    ):
        """Initialize the LLM-as-a-judge metric.

        Args:
            model (BaseLanguageModel): LangChain judge model.
            criteria (str | list[str]): Criterion name(s) to score on.
            prompt_template (Optional[str]): Unused legacy argument kept
                for backwards compatibility.
            custom_templates (Optional[dict[str, str]]): Optional mapping
                ``criterion -> template`` that overrides/extends the
                built-in templates.
            metric_ceil (int): Maximum integer score the judge can give.
            use_structured_output (bool): If ``True``, the judge is
                invoked via ``model.with_structured_output(JudgeScoreResponse,
                method="json_schema")`` so the score is returned as a
                validated integer instead of being regex-parsed from a
                free-text response. Defaults to ``False``.
        """
        super().__init__()
        self.model = model
        self.prompt_template = prompt_template
        self.metric_ceil = metric_ceil
        self.use_structured_output = use_structured_output

        self.prompt_templates = {
            "accuracy": ACCURACY_QA_TEMPLATE,
            "coherence": COHERENCE_TEMPLATE,
            "fluency": FLUENCY_TEMPLATE,
            "relevance": RELEVANCE_TEMPLATE,
        }

        if custom_templates:
            self.prompt_templates.update(custom_templates)

        if isinstance(criteria, str):
            criteria = [criteria]
        self.criteria = criteria

        self.templates = {
            crit: self.prompt_templates[crit] for crit in self.criteria
        }

    def _compute_raw(self, outputs, targets, dataset):
        scores = []
        runner = self.model
        if self.use_structured_output:
            runner = self.model.with_structured_output(
                JudgeScoreResponse, method="json_schema"
            )
        for _, template in self.templates.items():
            requests = [
                template.format(
                    metric_ceil=self.metric_ceil,
                    request=request,
                    response=response,
                )
                for request, response in zip(dataset, outputs)
            ]
            answers = runner.batch(requests)

            parsed = []
            if self.use_structured_output:
                for a in answers:
                    try:
                        parsed.append(int(a.score))
                    except (AttributeError, TypeError, ValueError):
                        parsed.append(0)
            else:
                for a in answers:
                    if isinstance(a, AIMessage):
                        content = (
                            a.content
                            if isinstance(a.content, str)
                            else str(a.content)
                        )
                        match = re.search(r"\d+", content)
                        parsed.append(int(match.group()) if match else 0)
                    else:
                        parsed.append(0)

            normalized = [
                clip(ans, 0, self.metric_ceil) / self.metric_ceil
                for ans in parsed
            ]
            scores.append(mean(normalized))

        return scores


class GEvalMetric(GenerationMetric):
    @staticmethod
    def _get_name() -> str:
        return "geval"

    def __init__(
        self,
        model: BaseLanguageModel,
        criteria: str | None = None,
        evaluation_steps: Optional[list[str]] = None,
        evaluation_params: Optional[list[LLMTestCaseParams]] = None,
        strict_mode: bool = False,
        use_structured_output: bool = False,
    ) -> None:
        super().__init__()
        self.use_structured_output = use_structured_output
        # The flag is forwarded to the DeepEval wrapper so that every
        # LLM call DeepEval issues for this metric goes through
        # ``with_structured_output(schema, method="json_schema")`` when
        # DeepEval provides a pydantic ``schema`` (per the
        # ``DeepEvalBaseLLM`` contract). When DeepEval does not supply a
        # schema, the wrapper falls back to a plain ``invoke`` that
        # returns the model's textual answer (legacy behaviour).
        wrapped_model = DeepEvalLangChainModel(
            model, use_structured_output=use_structured_output
        )

        if criteria is not None and evaluation_steps is not None:
            raise ValueError(
                "GEvalMetric: provide either `criteria` or "
                "`evaluation_steps`, but not both."
            )

        if evaluation_params is None:
            evaluation_params = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ]

        self._metric = GEval(
            name=self._get_name(),
            criteria=criteria,
            evaluation_steps=evaluation_steps,
            evaluation_params=evaluation_params,
            model=wrapped_model,
            strict_mode=strict_mode,
        )

    def _compute_raw(self, outputs, targets, dataset):
        scores = []

        if dataset is None:
            dataset = [""] * len(outputs)

        for output, target, request in zip(outputs, targets, dataset):
            test_case = LLMTestCase(
                input=request,
                actual_output=str(output),
                expected_output=str(target),
            )
            score = self._metric.measure(test_case, _show_indicator=False)

            scores.append(score)

        return scores


class ExactMatchMetric(GenerationMetric):
    """EM Metric for generation tasks."""

    @staticmethod
    def _get_name():
        return "em"

    def __init__(self):
        super().__init__()

    def _compute_raw(
        self,
        outputs: list[str | int],
        targets: list[str | int],
        dataset: Optional[list[str]] = None,
    ) -> List[float]:
        targets = [extract_number_from_text(item) for item in targets]
        outputs = [extract_number_from_text(item) for item in outputs]
        return [float(o == t) for o, t in zip(outputs, targets)]


class CodeBertScore(GenerationMetric):

    @staticmethod
    def _get_name():
        return "codebertscore"

    def __init__(self):
        super().__init__()
        self.scorer = BERTScorer(lang="java")

    def _compute_raw(self, outputs, targets, dataset=None):
        _, _, F1 = self.scorer.score(cands=outputs, refs=targets)
        f1_list = list(F1.numpy())
        return f1_list

    def parse_output(self, output: str) -> str:
        extracted = extract_answer(output, self.ANS_TAGS, format_mismatch_label=output)
        return extract_number_from_text(extracted)


def define_lang(outputs, targets):
    langs = [detect_language(target) for target in targets]
    return max(set(langs), key=langs.count)


CLASSIFICATION_METRIC_NAME_MAPPING = {
    metric._get_name(): metric for metric in ClassificationMetric.__subclasses__()
}

GENERATION_METRIC_NAME_MAPPING = {
    metric._get_name(): metric for metric in GenerationMetric.__subclasses__()
}


def validate_and_create_metric(
    task: Task,
    metric: str | None,
    model: BaseLanguageModel | None = None,
    **kwargs,
) -> BaseMetric:
    """
    Validates given metric in order to correspond the given task.
    Returns the given metric name back if the validation succeeded.

    Args:
        task (Task): The type of task, either "classification" or "generation".
        metric (str): Name of the metric to validate.
        model (BaseLanguageModel): model to use for evaluation
            (for LLM-as-judge and GEval)
    Returns:
        str: the name of the metric.
    Raises:
        ValueError: If the specified task name is not recognized
        ValueError: If the specified metric name is not
            matched to the specified task name.
    """

    if metric is None:
        metric = get_default_metric(task)
    match task:
        case Task.CLASSIFICATION:
            if metric in CLASSIFICATION_METRIC_NAME_MAPPING.keys():
                return CLASSIFICATION_METRIC_NAME_MAPPING[metric]()
            error_msg = (
                f"Invalid metric for {task} task: {metric}. "
                f"Available metrics: {
                    ', '.join(CLASSIFICATION_METRIC_NAME_MAPPING.keys())
                }."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        case Task.GENERATION:
            if metric == "llm_as_judge":
                if model is None:
                    error_msg = "Model for llm_as_judge metric must not be None"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                return LLMAsJudge(
                    model=model,
                    criteria=kwargs.get("llm_as_judge_criteria", "relevance"),
                    custom_templates=kwargs.get(
                        "llm_as_judge_custom_templates"
                    ),
                    metric_ceil=kwargs.get("llm_as_judge_metric_ceil", 10),
                    use_structured_output=kwargs.get(
                        "use_structured_output", False
                    ),
                )
            if metric == "geval":
                if model is None:
                    error_msg = "Model for geval metric must not be None"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                return GEvalMetric(
                    model=model,
                    criteria=kwargs.get("geval_criteria"),
                    evaluation_steps=kwargs.get("geval_evaluation_steps"),
                    evaluation_params=kwargs.get("geval_evaluation_params"),
                    strict_mode=kwargs.get("geval_strict_mode", False),
                    use_structured_output=kwargs.get(
                        "use_structured_output", False
                    ),
                )
            if metric in GENERATION_METRIC_NAME_MAPPING.keys():
                return GENERATION_METRIC_NAME_MAPPING[metric]()
            error_msg = (
                f"Invalid metric for {task} task: {metric}. "
                f"Available metrics: {
                    ', '.join(GENERATION_METRIC_NAME_MAPPING.keys())
                }."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    error_msg = (
        f"Invalid task: {task}" f"Available tasks: classification, generation"
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


def get_default_metric(task: Task) -> str:
    """
    Returns default metric names for the provided task name.

    Args:
        task (Task): The type of task, either "classification" or "generation".
    Returns:
        str: the name of the default metric for the specified task.
    """

    match task:
        case Task.CLASSIFICATION:
            return "f1"
        case Task.GENERATION:
            return "bertscore"
