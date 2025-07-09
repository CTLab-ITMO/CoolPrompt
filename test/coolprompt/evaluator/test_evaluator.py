import unittest
from unittest.mock import MagicMock, patch, ANY
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator.evaluator import Evaluator
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)
from coolprompt.utils.enums import Task


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock(spec=BaseLanguageModel)
        self.mock_task = MagicMock()
        self.mock_metric = MagicMock()

        self.patcher = patch(
            "coolprompt.evaluator.validate_and_create_metric")
        self.mock_validate_and_create_metric = self.patcher.start()
        self.mock_validate_and_create_metric.return_value = self.mock_metric

        self.evaluator = Evaluator(
            model=self.mock_model,
            task=self.mock_task,
            metric=self.mock_metric
        )

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        """Testing initialization of Evaluator"""

        self.assertEqual(self.evaluator.model, self.mock_model)
        self.assertEqual(self.evaluator.task, self.mock_task)
        self.assertEqual(self.evaluator.metric, self.mock_metric)

    def test_get_default_template_classification(self):
        """Testing default template getter"""

        self.mock_task.__eq__.side_effect = (
            lambda other: other is Task.CLASSIFICATION)

        template = self.evaluator._get_default_template()

        self.assertEqual(template, CLASSIFICATION_TASK_TEMPLATE)

    def test_get_default_template_generation(self):
        """Testing default template getter"""

        self.mock_task.__eq__.side_effect = (
            lambda other: other is Task.GENERATION)

        template = self.evaluator._get_default_template()

        self.assertEqual(template, GENERATION_TASK_TEMPLATE)

    def test_get_full_prompt_classification(self):
        """Testing usage right classification prompt template"""

        self.mock_task.__eq__.side_effect = (
            lambda other: other is Task.CLASSIFICATION)
        self.mock_metric.label_to_id = {"positive": 0, "negative": 1}

        prompt = "Analyze sentiment"
        sample = "I love this product!"
        full_prompt = self.evaluator._get_full_prompt(prompt, sample)

        expected_labels = "positive, negative"
        expected_prompt = CLASSIFICATION_TASK_TEMPLATE.format(
            PROMPT=prompt, LABELS=expected_labels, INPUT=sample
        )
        self.assertEqual(full_prompt, expected_prompt)

        full_prompt_hype = self.evaluator._get_full_prompt(
            prompt, sample, CLASSIFICATION_TASK_TEMPLATE_HYPE
        )

        expected_prompt_hype = CLASSIFICATION_TASK_TEMPLATE_HYPE.format(
            PROMPT=prompt, LABELS=expected_labels, INPUT=sample
        )
        self.assertEqual(full_prompt_hype, expected_prompt_hype)

    def test_get_full_prompt_generation(self):
        """Testing usage right generation prompt template"""
        self.mock_task.__eq__.side_effect = (
            lambda other: other is Task.GENERATION)

        prompt = "Summarize this text"
        sample = "Long article about AI..."
        full_prompt = self.evaluator._get_full_prompt(prompt, sample)

        expected_prompt = GENERATION_TASK_TEMPLATE.format(
            PROMPT=prompt, INPUT=sample
        )
        self.assertEqual(full_prompt, expected_prompt)

        full_prompt_hype = self.evaluator._get_full_prompt(
            prompt, sample, GENERATION_TASK_TEMPLATE_HYPE
        )

        expected_prompt_hype = GENERATION_TASK_TEMPLATE_HYPE.format(
            PROMPT=prompt, INPUT=sample
        )
        self.assertEqual(full_prompt_hype, expected_prompt_hype)

    def test_evaluate_classification(self):
        """Testing the work of classification task evaluation"""

        prompt = "Classify sentiment"
        dataset = ["Great movie!", "Terrible experience"]
        targets = [1, 0]
        template = CLASSIFICATION_TASK_TEMPLATE

        self.mock_model.batch.return_value = ["positive", "negative"]

        self.mock_task.__eq__.side_effect = (
            lambda other: other is Task.CLASSIFICATION)

        self.mock_metric.compute.return_value = 1.0

        result = self.evaluator.evaluate(
            prompt, dataset, targets, template
        )

        self.mock_metric.extract_labels.assert_called_once_with(targets)
        self.mock_model.batch.assert_called_once()

        self.mock_metric.compute.assert_called_once_with(
            ["positive", "negative"], targets
        )
        self.assertEqual(result, 1.0)

    def test_evaluate_generation(self):
        """Testing the work of generation task evaluation"""

        prompt = "Summarize this text"
        dataset = ["First long text...", "Second long text..."]
        targets = ["Short summary 1", "Short summary 2"]
        template = GENERATION_TASK_TEMPLATE

        self.mock_model.batch.return_value = ["Summary 1", "Summary 2"]

        self.mock_metric.compute.return_value = 0.666

        result = self.evaluator.evaluate(
            prompt, dataset, targets, template
        )

        self.mock_model.batch.assert_called_once()

        self.mock_metric.compute.assert_called_once_with(
            ["Summary 1", "Summary 2"], targets
        )
        self.assertEqual(result, 0.666)

    def test_evaluate_empty_dataset(self):
        """Testing that evaluation can be called with empty dataset"""

        result = self.evaluator.evaluate(
            "Prompt", [], [], CLASSIFICATION_TASK_TEMPLATE
        )
        self.mock_model.batch.assert_called_once_with([])
        self.mock_metric.compute.assert_called_once_with(ANY, [])
        self.assertEqual(result, self.mock_metric.compute.return_value)

    def test_evaluate_label_extraction_classification(self):
        """Testing the work of label extraction function"""

        self.mock_task.__eq__.side_effect = (
            lambda other: other is Task.CLASSIFICATION)

        self.mock_metric.label_to_id = {}
        self.mock_metric.extract_labels.side_effect = lambda labels: setattr(
            self.mock_metric,
            "label_to_id",
            {str(lbl): i for i, lbl in enumerate(sorted(set(labels)))},
        )

        targets = ["cat", "dog", "cat"]

        self.evaluator.evaluate(
            "Classify",
            ["sample"],
            targets,
            CLASSIFICATION_TASK_TEMPLATE,
        )

        self.mock_metric.extract_labels.assert_called_once_with(targets)
        self.assertEqual(self.mock_metric.label_to_id, {"cat": 0, "dog": 1})
