import sys
import os
import numpy as np
import unittest
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)

from coolprompt.evaluator.metrics import (
    ClassificationMetric,
    GenerationMetric,
    CLASSIFICATION_METRICS,
    GENERATION_METRICS,
    create_metric,
    validate_metric,
    get_default_metric
)


class TestClassificationMetric(unittest.TestCase):

    def setUp(self):
        self.mock_metric = MagicMock()
        self.patcher = patch('coolprompt.evaluator.metrics.load')
        self.mock_create_metric = self.patcher.start()
        self.mock_create_metric.return_value = self.mock_metric

        self.name = np.random.choice(list(CLASSIFICATION_METRICS))
        self.metric = ClassificationMetric(name=self.name)

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        self.assertEqual(self.name, self.metric._name)
        self.mock_create_metric.assert_called_once_with(self.name)
        self.assertEqual(self.metric._metric, self.mock_metric)
        if self.name == 'f1':
            self.assertDictEqual(
                self.metric._compute_kwargs,
                {"average": "macro"}
            )
        else:
            self.assertDictEqual(self.metric._compute_kwargs, {})
        self.assertIsNone(self.metric.label_to_id)

    def test_encode_labels(self):
        outputs = ['1', '2']
        targets = ['1', '2']
        encoded_outputs, encoded_targets = self.metric._encode_labels(
            outputs,
            targets
        )
        self.assertListEqual(encoded_outputs, [0, 1])
        self.assertListEqual(encoded_targets, [0, 1])

    def test_encode_labels_mismatch(self):
        outputs = ['213]', '2']
        targets = ['1', '2']
        encoded_outputs, encoded_targets = self.metric._encode_labels(
            outputs,
            targets
        )
        self.assertListEqual(encoded_outputs, [-1, 1])
        self.assertListEqual(encoded_targets, [0, 1])

    def test_extract_labels(self):
        targets = ['target1', 'target2']
        self.metric.extract_labels(targets)
        self.assertDictEqual(
            self.metric.label_to_id,
            {'target1': 0, 'target2': 1}
        )

    def test_empty_targets_extract_labels(self):
        empty_targets = []
        self.metric.extract_labels(empty_targets)
        self.assertDictEqual(self.metric.label_to_id, {})

    def test_compute(self):
        outputs = ['<ans>1</ans>', '<ans>2</ans>']
        targets = ['1', '2']
        self.mock_metric.compute.return_value = {self.name: 1.0}
        result = self.metric.compute(outputs, targets)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 1)

    def test_different_length_compute(self):
        different_length_outputs = ['just one output']
        targets = ['1', '2']
        self.mock_metric.compute.side_effect = ValueError()
        with self.assertRaises(ValueError):
            self.metric.compute(different_length_outputs, targets)


class TestGenerationMetric(unittest.TestCase):

    def setUp(self):
        self.mock_metric = MagicMock()
        self.patcher = patch('coolprompt.evaluator.metrics.load')
        self.mock_create_metric = self.patcher.start()
        self.mock_create_metric.return_value = self.mock_metric

        self.name = np.random.choice(list(GENERATION_METRICS))
        self.metric = GenerationMetric(name=self.name)

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        if self.name == 'rouge':
            self.name = 'rougeL'
        self.assertEqual(self.name, self.metric._name)
        self.mock_create_metric.assert_called_once_with(self.name)
        self.assertEqual(self.metric._metric, self.mock_metric)

    def test_compute(self):
        outputs = ['some', 'outputs']
        slightly_mismatched_targets = ['some', 'targets']
        self.mock_metric.compute.return_value = {self.name: 1.0}
        self.assertTrue(
            0 <= self.metric.compute(outputs, slightly_mismatched_targets) <= 1
        )

    def test_different_length_compute(self):
        different_length_outputs = ['just one output']
        targets = ['1', '2']
        self.mock_metric.compute.side_effect = ValueError()
        with self.assertRaises(ValueError):
            self.metric.compute(different_length_outputs, targets)


class TestUtilityFunctions(unittest.TestCase):

    def setUp(self):
        self.mock_metric = MagicMock()
        self.patcher = patch('coolprompt.evaluator.metrics.load')
        self.mock_create_metric = self.patcher.start()
        self.mock_create_metric.return_value = self.mock_metric

    def tearDown(self):
        self.patcher.stop()

    def test_create_classification_metric(self):
        self.assertIsInstance(
            create_metric(np.random.choice(list(CLASSIFICATION_METRICS))),
            ClassificationMetric
        )

    def test_create_generation_metric(self):
        self.assertIsInstance(
            create_metric(np.random.choice(list(GENERATION_METRICS))),
            GenerationMetric
        )

    def test_create_invalid_metric(self):
        with self.assertRaises(ValueError):
            create_metric('random_not_metric')

    def test_validate_metric_correct(self):
        metric = np.random.choice(list(CLASSIFICATION_METRICS))
        output = validate_metric('classification', metric)
        self.assertIsInstance(output, str)
        self.assertEqual(output, metric)

    def test_validate_metric_incorrect_task(self):
        with self.assertRaises(ValueError):
            validate_metric('incorrect task', 'metric')

    def test_validate_metric_incorrect_metric(self):
        with self.assertRaises(ValueError):
            validate_metric('classification', 'incorrect metric')

    def test_validate_metric_mismatched_metric(self):
        with self.assertRaises(ValueError):
            validate_metric('classification', 'meteor')

    def test_get_default_metric(self):
        task = np.random.choice(['classification', 'generation'])
        metric = get_default_metric(task)
        if task == 'classification':
            self.assertEqual(metric, 'f1')
        else:
            self.assertEqual(metric, 'meteor')

    def test_get_default_metric_incorrect_task(self):
        self.assertIsNone(get_default_metric('incorrect task'))


if __name__ == '__main__':
    unittest.main()
