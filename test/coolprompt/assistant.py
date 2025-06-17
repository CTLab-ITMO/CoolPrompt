import sys
import os
import unittest
from unittest.mock import MagicMock, patch, ANY

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(project_root)

from coolprompt.assistant import PromptTuner
from coolprompt.utils.prompt_template import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE
)


class TestPromptTuner(unittest.TestCase):
    START_PROMPT = "start prompt"
    FINAL_PROMPT = "final prompt"

    def setUp(self):
        self.validate_model_patcher = patch(
            'coolprompt.assistant.validate_model'
        )
        self.mock_validate_model = self.validate_model_patcher.start()

        self.mock_evaluator = MagicMock()
        self.mock_evaluator.evaluate.return_value = 0.5
        self.evaluator_patcher = patch(
            'coolprompt.assistant.Evaluator.__new__'
        )
        self.mock_evaluator_init = self.evaluator_patcher.start()
        self.mock_evaluator_init.return_value = self.mock_evaluator

        self.naive_optimizer_patcher = patch(
            'coolprompt.assistant.naive_optimizer'
        )
        self.mock_naive_optimizer = self.naive_optimizer_patcher.start()
        self.mock_naive_optimizer.return_value = self.FINAL_PROMPT

        self.reflective_optimizer_patcher = patch(
            'coolprompt.assistant.reflectiveprompt'
        )
        self.mock_reflective_optimizer = \
            self.reflective_optimizer_patcher.start()
        self.mock_reflective_optimizer.return_value = self.FINAL_PROMPT

        self.mock_model = MagicMock()
        self.prompt_tuner = PromptTuner(self.mock_model)

    def tearDown(self):
        self.validate_model_patcher.stop()
        self.evaluator_patcher.stop()
        self.naive_optimizer_patcher.stop()
        self.reflective_optimizer_patcher.stop()

    def _test_init(self, prompt_tuner):
        self.mock_validate_model.assert_called()
        self.assertIsNone(prompt_tuner.init_metric)
        self.assertIsNone(prompt_tuner.final_metric)
        self.assertEqual(prompt_tuner._model, self.mock_model)

    def test_initialization(self):
        self._test_init(self.prompt_tuner)

    def test_initialization_without_model(self):
        patcher = patch('coolprompt.assistant.DefaultLLM.init')
        mock_model_init = patcher.start()
        mock_model_init.return_value = self.mock_model
        prompt_tuner = PromptTuner()

        self._test_init(prompt_tuner)

        patcher.stop()

    def test_get_task_prompt_template_classification(self):
        self.assertEqual(
            self.prompt_tuner.get_task_prompt_template('classification'),
            CLASSIFICATION_TASK_TEMPLATE
        )

    def test_get_task_prompt_template_generation(self):
        self.assertEqual(
            self.prompt_tuner.get_task_prompt_template('generation'),
            GENERATION_TASK_TEMPLATE
        )

    def test_run_unsupported_method(self):
        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT,
                method='unsupported method'
            )

    def test_run_dataset_but_no_target(self):
        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT,
                dataset=[]
            )

    def test_run_target_is_smaller_than_dataset(self):
        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT,
                dataset=['s'],
                target=[]
            )

    def test_run_incorrect_metric(self):
        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT,
                dataset=[],
                metric="incorrect metric"
            )

    def test_naive_optimizer_without_dataset(self):
        final_prompt = self.prompt_tuner.run(self.START_PROMPT)
        self.mock_naive_optimizer.assert_called_once_with(
            self.mock_model,
            self.START_PROMPT
        )
        self.assertEqual(final_prompt, self.FINAL_PROMPT)

    def test_run_metrics_check(self):
        self.prompt_tuner.run(
            self.START_PROMPT,
            dataset=[]
        )
        self.assertEqual(self.prompt_tuner.init_metric, 0.5)
        self.assertEqual(self.prompt_tuner.final_metric, 0.5)
        self.mock_evaluator_init.assert_called_once_with(
            ANY,
            self.mock_model,
            "meteor"  # check for default metric if None is passed
        )

    def test_run_reflective_without_problem_description(self):
        with self.assertRaises(ValueError):
            self.prompt_tuner.run(self.START_PROMPT, method='reflective')

    def test_run_reflective_without_dataset(self):
        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT,
                method='reflective',
                problem_description=''
            )

    def test_run_reflective_empty_dataset(self):
        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT,
                dataset=[],
                target=[],
                method='reflective',
                problem_description=''
            )

    def test_run_reflective(self):
        self.assertEqual(
            self.prompt_tuner.run(
                self.START_PROMPT,
                dataset=['1', '2'],
                target=['1', '2'],
                method='reflective',
                problem_description=''
            ),
            self.FINAL_PROMPT
        )
        self.assertEqual(self.prompt_tuner.init_metric, 0.5)
        self.assertEqual(self.prompt_tuner.final_metric, 0.5)
