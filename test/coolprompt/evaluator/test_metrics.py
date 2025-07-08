import numpy as np
import unittest
from unittest.mock import MagicMock, patch

from coolprompt.evaluator.metrics import (
    ClassificationMetric,
    GenerationMetric,
    CLASSIFICATION_METRICS,
    GENERATION_METRICS,
    validate_and_create_metric,
    get_default_metric
)
from coolprompt.utils.var_validation import Task


class TestClassificationMetric(unittest.TestCase):

    def setUp(self):
        self.mock_metric = MagicMock()
        self.patcher = patch('coolprompt.evaluator.metrics.load')
        self.mock_validate_and_create_metric = self.patcher.start()
        self.mock_validate_and_create_metric.return_value = self.mock_metric

        self.name = np.random.choice(list(CLASSIFICATION_METRICS))
        self.metric = ClassificationMetric(name=self.name)

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        """Testing the initialization of classification metric"""

        self.assertEqual(self.name, self.metric._name)
        self.mock_validate_and_create_metric.assert_called_once_with(self.name)
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
        """Testing the work of encode labels function"""

        outputs = ['1', '2']
        targets = ['1', '2']
        encoded_outputs, encoded_targets = self.metric._encode_labels(
            outputs,
            targets
        )
        self.assertListEqual(encoded_outputs, [0, 1])
        self.assertListEqual(encoded_targets, [0, 1])

    def test_encode_labels_mismatch(self):
        """
        Testing the work of encode labels function
        when outputs mismatch targets
        """

        outputs = ['213]', '2']
        targets = ['1', '2']
        encoded_outputs, encoded_targets = self.metric._encode_labels(
            outputs,
            targets
        )
        self.assertListEqual(encoded_outputs, [-1, 1])
        self.assertListEqual(encoded_targets, [0, 1])

    def test_extract_labels(self):
        """Testing the work of extract labels function"""

        targets = ['target1', 'target2']
        self.metric.extract_labels(targets)
        self.assertDictEqual(
            self.metric.label_to_id,
            {'target1': 0, 'target2': 1}
        )

    def test_empty_targets_extract_labels(self):
        """Testing that the extraction can be produced for empty targets"""

        empty_targets = []
        self.metric.extract_labels(empty_targets)
        self.assertDictEqual(self.metric.label_to_id, {})

    def test_compute(self):
        """Testing the work of compute method"""

        outputs = ['<ans>1</ans>', '<ans>2</ans>']
        targets = ['1', '2']
        self.mock_metric.compute.return_value = {self.name: 1.0}
        result = self.metric.compute(outputs, targets)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 1)

    def test_different_length_compute(self):
        """
        Testing that compute raises exception
        when the length of outputs doesn't match the length of targets
        """

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
        self.name = 'rouge'
        self.metric = GenerationMetric(name=self.name)

    def tearDown(self):
        self.patcher.stop()

    def _fix_name(self):
        if self.name == 'rouge':
            self.name = 'rougeL'

    def test_initialization(self):
        """Testing the initialization of generation metric"""

        self.mock_create_metric.assert_called_once_with(self.name)
        self.assertEqual(self.metric._metric, self.mock_metric)
        self._fix_name()
        self.assertEqual(self.name, self.metric._name)

    def test_compute(self):
        """Testing the work of compute method"""

        outputs = ['some', 'outputs']
        slightly_mismatched_targets = ['some', 'targets']
        self._fix_name()
        self.mock_metric.compute.return_value = {self.name: 1.0}
        self.assertTrue(
            0 <= self.metric.compute(outputs, slightly_mismatched_targets) <= 1
        )

    def test_different_length_compute(self):
        """
        Testing that compute raises exception
        when the length of outputs doesn't match the length of targets
        """

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
        """
        Testing the work of create_metric function for classification task
        """

        self.assertIsInstance(
            validate_and_create_metric(
                Task.CLASSIFICATION,
                np.random.choice(list(CLASSIFICATION_METRICS))),
            ClassificationMetric
        )

    def test_create_generation_metric(self):
        """
        Testing the work of create_metric function for generation task
        """

        self.assertIsInstance(
            validate_and_create_metric(
                Task.GENERATION,
                np.random.choice(list(GENERATION_METRICS))),
            GenerationMetric
        )

    def test_create_invalid_metric(self):
        """
        Testing that create_metric raises an exception when
        the incorrect metric name is provided
        """

        with self.assertRaises(ValueError):
            validate_and_create_metric(Task.CLASSIFICATION, 'random_not_metric')

        with self.assertRaises(ValueError):
            validate_and_create_metric(Task.GENERATION, 'random_not_metric')

    def test_validate_metric_mismatched_metric(self):
        """
        Testing that validate_metric raises an exception when
        the metric doesn't match the task
        """

        with self.assertRaises(ValueError):
            validate_and_create_metric(Task.CLASSIFICATION, 'meteor')

        with self.assertRaises(ValueError):
            validate_and_create_metric(Task.GENERATION, 'f1')

    def test_get_default_metric(self):
        """Testing the work of get_default_metric function"""

        self.assertEqual(get_default_metric(Task.CLASSIFICATION), 'f1')
        self.assertEqual(get_default_metric(Task.GENERATION), 'meteor')
