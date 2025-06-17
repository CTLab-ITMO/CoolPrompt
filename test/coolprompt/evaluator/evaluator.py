import sys
import os
import unittest
from unittest.mock import MagicMock, patch, ANY
from langchain_core.language_models.base import BaseLanguageModel

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)

from coolprompt.evaluator.evaluator import Evaluator
from coolprompt.utils.prompt_template import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE
)


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock(spec=BaseLanguageModel)
        self.mock_metric = MagicMock()

        self.patcher = patch('coolprompt.evaluator.evaluator.create_metric')
        self.mock_create_metric = self.patcher.start()
        self.mock_create_metric.return_value = self.mock_metric

        self.evaluator = Evaluator(model=self.mock_model, metric="f1")

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        self.assertEqual(self.evaluator.model, self.mock_model)
        self.assertEqual(self.evaluator.metric, self.mock_metric)
        self.mock_create_metric.assert_called_once_with("f1")

    def test_get_full_prompt_classification(self):
        self.mock_metric.label_to_id = {"positive": 0, "negative": 1}

        prompt = "Analyze sentiment"
        sample = "I love this product!"
        full_prompt = self.evaluator._get_full_prompt(
            prompt,
            sample,
            "classification"
        )

        expected_labels = "positive, negative"
        expected_prompt = CLASSIFICATION_TASK_TEMPLATE.format(
            PROMPT=prompt,
            LABELS=expected_labels,
            INPUT=sample
        )
        self.assertEqual(full_prompt, expected_prompt)

    def test_get_full_prompt_generation(self):
        prompt = "Summarize this text"
        sample = "Long article about AI..."
        full_prompt = self.evaluator._get_full_prompt(
            prompt,
            sample,
            "generation"
        )

        expected_prompt = GENERATION_TASK_TEMPLATE.format(
            PROMPT=prompt,
            INPUT=sample
        )
        self.assertEqual(full_prompt, expected_prompt)

    def test_get_full_prompt_invalid_task(self):
        with self.assertRaises(ValueError):
            self.evaluator._get_full_prompt("prompt", "sample", "invalid_task")

    def test_evaluate_classification(self):
        prompt = "Classify sentiment"
        dataset = ["Great movie!", "Terrible experience"]
        targets = [1, 0]
        task = "classification"

        self.mock_model.batch.return_value = ["positive", "negative"]

        self.mock_metric.compute.return_value = 1.0

        result = self.evaluator.evaluate(prompt, dataset, targets, task)

        self.mock_metric.extract_labels.assert_called_once_with(targets)
        self.mock_model.batch.assert_called_once()

        self.mock_metric.compute.assert_called_once_with(
            ["positive", "negative"],
            targets
        )
        self.assertEqual(result, 1.0)

    def test_evaluate_generation(self):
        prompt = "Summarize this text"
        dataset = ["First long text...", "Second long text..."]
        targets = ["Short summary 1", "Short summary 2"]
        task = "generation"

        self.mock_model.batch.return_value = ["Summary 1", "Summary 2"]

        self.mock_metric.compute.return_value = 0.666

        result = self.evaluator.evaluate(prompt, dataset, targets, task)

        self.mock_model.batch.assert_called_once()

        self.mock_metric.compute.assert_called_once_with(
            ["Summary 1", "Summary 2"], 
            targets
        )
        self.assertEqual(result, 0.666)

    def test_evaluate_empty_dataset(self):
        result = self.evaluator.evaluate("Prompt", [], [], "classification")
        self.mock_model.batch.assert_called_once_with([])
        self.mock_metric.compute.assert_called_once_with(ANY, [])
        self.assertEqual(result, self.mock_metric.compute.return_value)

    def test_evaluate_label_extraction_classification(self):
        self.mock_metric.label_to_id = {}
        self.mock_metric.extract_labels.side_effect = lambda labels: setattr(
            self.mock_metric,
            'label_to_id',
            {str(lbl): i for i, lbl in enumerate(sorted(set(labels)))}
        )

        targets = ["cat", "dog", "cat"]

        self.evaluator.evaluate(
            "Classify",
            ["sample"],
            targets,
            "classification"
        )

        self.mock_metric.extract_labels.assert_called_once_with(targets)
        self.assertEqual(
            self.mock_metric.label_to_id,
            {"cat": 0, "dog": 1}
        )


if __name__ == "__main__":
    unittest.main()
