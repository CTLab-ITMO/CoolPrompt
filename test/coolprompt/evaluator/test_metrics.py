import numpy as np
import unittest
from unittest.mock import MagicMock, patch

from coolprompt.evaluator.metrics import (
    ClassificationMetric,
    GenerationMetric,
    UtilsMetric,
    UTILS_METRIC,
    CLASSIFICATION_METRICS,
    GENERATION_METRICS,
    validate_and_create_metric,
    get_default_metric
)
from coolprompt.utils.enums import Task


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

class TestUtilsMetrics(unittest.TestCase):

    def setUp(self):
        self.mock_embedder = MagicMock()
        self.mock_embedder.encode.return_value = MagicMock()
        
    def test_utils_metric_cosine_initialization(self):
        """Testing the initialization of cosine metric"""
        metric = UtilsMetric("cosine", embedder=self.mock_embedder)
        self.assertEqual(metric._name, "cosine")
        self.assertEqual(metric._embedder, self.mock_embedder)
        self.assertIsNone(metric._metric)
        self.assertDictEqual(metric._compute_kwargs, {})

    def test_utils_metric_perplexity_initialization(self):
        """Testing the initialization of perplexity metric"""
        model_name = "gpt2"
        metric = UtilsMetric("perplexity", model_name=model_name)
        self.assertEqual(metric._name, "perplexity")
        self.assertIsNone(metric._embedder)
        self.assertIsNotNone(metric._metric)
        self.assertDictEqual(metric._compute_kwargs, {"model_id": model_name})

    def test_utils_metric_cosine_encode_labels(self):
        """Testing the work of encode labels function for cosine metric"""
        metric = UtilsMetric("cosine", embedder=self.mock_embedder)
        outputs = ["hello world", "test sentence"]
        targets = ["hello", "test"]
        
        emb_outputs, emb_targets = metric._encode_labels(outputs, targets)
        
        self.mock_embedder.encode.assert_any_call(outputs, convert_to_tensor=True)
        self.mock_embedder.encode.assert_any_call(targets, convert_to_tensor=True)
        self.assertEqual(emb_outputs, self.mock_embedder.encode.return_value)
        self.assertEqual(emb_targets, self.mock_embedder.encode.return_value)

    def test_utils_metric_perplexity_encode_labels(self):
        """Testing the work of encode labels function for perplexity metric"""
        metric = UtilsMetric("perplexity")
        outputs = ["hello world", "test sentence"]
        targets = ["hello", "test"]
        
        result_outputs, result_targets = metric._encode_labels(outputs, targets)
        
        self.assertListEqual(result_outputs, outputs)
        self.assertListEqual(result_targets, targets)

    @patch('coolprompt.evaluator.metrics.util.cos_sim')
    def test_utils_metric_cosine_compute_raw(self, mock_cos_sim):
        """Testing the work of compute_raw function for cosine metric"""
        metric = UtilsMetric("cosine", embedder=self.mock_embedder)
        
        mock_outputs = MagicMock()
        mock_targets = MagicMock()
        mock_sims = MagicMock()
        mock_diag = MagicMock()
        mock_mean = MagicMock()
        
        mock_cos_sim.return_value = mock_sims
        mock_sims.diag.return_value = mock_diag
        mock_diag.mean.return_value = mock_mean
        mock_mean.item.return_value = 0.8
        
        result = metric._compute_raw(mock_outputs, mock_targets)
        
        mock_cos_sim.assert_called_once_with(mock_outputs, mock_targets)
        mock_sims.diag.assert_called_once()
        mock_diag.mean.assert_called_once()
        mock_mean.item.assert_called_once()
        self.assertEqual(result, 0.8)

    @patch('coolprompt.evaluator.metrics.load')
    def test_utils_metric_perplexity_compute_raw(self, mock_load):
        """Testing the work of compute_raw function for perplexity metric"""
        mock_metric = MagicMock()
        mock_load.return_value = mock_metric
        mock_metric.compute.return_value = {"perplexity": 15.3}
        
        metric = UtilsMetric("perplexity", model_name="gpt2")
        outputs = ["hello world", "test sentence"]
        
        result = metric._compute_raw(outputs, []) 
        
        mock_metric.compute.assert_called_once_with(predictions=outputs, model_id="gpt2")
        self.assertEqual(result, 15.3)

    def test_utils_metric_compute_with_cosine(self):
        """Testing the complete compute method for cosine metric"""
        metric = UtilsMetric("cosine", embedder=self.mock_embedder)
        
        with patch.object(metric, '_encode_labels') as mock_encode, patch.object(metric, '_compute_raw') as mock_compute:
            
            mock_encode.return_value = (MagicMock(), MagicMock())
            mock_compute.return_value = 0.9
            
            outputs = ["<ans>hello world</ans>", "<ans>test sentence</ans>"]
            targets = ["hello", "test"]
            
            result = metric.compute(outputs, targets)
            
            mock_encode.assert_called_once()
            mock_compute.assert_called_once()
            self.assertEqual(result, 0.9)

    def test_utils_metric_compute_with_perplexity(self):
        metric = UtilsMetric("perplexity", model_name="gpt2")
        
        with patch.object(metric, '_encode_labels') as mock_encode, \
             patch.object(metric, '_compute_raw') as mock_compute:
            
            mock_encode.return_value = (["hello", "hi"], [])
            mock_compute.return_value = 12.5
            
            outputs = ["<ans>hello</ans>", "<ans>hi</ans>"]
            targets = ["dummy", "targets"]  
            
            result = metric.compute(outputs, targets)

            mock_encode.assert_called_once()
            mock_compute.assert_called_once()
            self.assertEqual(result, 12.5)

    def test_utils_metric_str_representation(self):
        metric = UtilsMetric("cosine", embedder=self.mock_embedder)
        self.assertEqual(str(metric), "cosine")

    def test_utils_metric_equality(self):
        metric1 = UtilsMetric("cosine", embedder=self.mock_embedder)
        metric2 = UtilsMetric("cosine", embedder=self.mock_embedder)
        metric3 = UtilsMetric("perplexity")
        self.assertEqual(metric1, metric2)
        self.assertNotEqual(metric1, metric3)

    def test_validate_and_create_utils_metric(self):
        for metric_name in UTILS_METRIC:
            if metric_name == "cosine": metric = validate_and_create_metric(Task.UTILS, metric_name, utils_embedder=self.mock_embedder)
            else: metric = validate_and_create_metric(Task.UTILS, metric_name)
            self.assertIsInstance(metric, UtilsMetric)
            self.assertEqual(metric._name, metric_name)

    def test_validate_and_create_utils_metric_cosine_without_embedder(self):
        with self.assertRaises(ValueError):
            validate_and_create_metric(Task.UTILS, "cosine")

    def test_get_default_metric_for_utils(self):
        self.assertEqual(get_default_metric(Task.UTILS), "cosine")

    def test_utils_metric_perplexity_empty_outputs(self):
        metric = UtilsMetric("perplexity", model_name="gpt2")

        outputs = []
        targets = []

        with self.assertRaises(Exception):
            metric.compute(outputs, targets)
