import unittest
from unittest.mock import MagicMock, patch, ANY

from coolprompt.assistant import PromptTuner
from coolprompt.utils.prompt_templates.default_templates import (
    CLASSIFICATION_TASK_TEMPLATE,
    GENERATION_TASK_TEMPLATE,
)
from coolprompt.utils.prompt_templates.hype_templates import (
    CLASSIFICATION_TASK_TEMPLATE_HYPE,
    GENERATION_TASK_TEMPLATE_HYPE,
)


class TestPromptTuner(unittest.TestCase):
    START_PROMPT = "start prompt"
    FINAL_PROMPT = "final prompt"

    def setUp(self):
        self.validate_model_patcher = patch(
            "coolprompt.assistant.validate_model"
        )
        self.mock_validate_model = self.validate_model_patcher.start()

        self.mock_evaluator = MagicMock()
        self.mock_evaluator.evaluate.return_value = 0.5
        self.evaluator_patcher = patch(
            "coolprompt.assistant.Evaluator.__new__"
        )
        self.mock_evaluator_init = self.evaluator_patcher.start()
        self.mock_evaluator_init.return_value = self.mock_evaluator

        self.hype_optimizer_patcher = patch(
            "coolprompt.assistant.hype_optimizer"
        )
        self.mock_hype_optimizer = self.hype_optimizer_patcher.start()
        self.mock_hype_optimizer.return_value = self.FINAL_PROMPT

        self.reflective_optimizer_patcher = patch(
            "coolprompt.assistant.reflectiveprompt"
        )
        self.mock_reflective_optimizer = (
            self.reflective_optimizer_patcher.start()
        )
        self.mock_reflective_optimizer.return_value = self.FINAL_PROMPT

        self.mock_model = MagicMock()
        self.prompt_tuner = PromptTuner(self.mock_model)

    def tearDown(self):
        self.validate_model_patcher.stop()
        self.evaluator_patcher.stop()
        self.hype_optimizer_patcher.stop()
        self.reflective_optimizer_patcher.stop()

    def _test_init(self, prompt_tuner):
        self.mock_validate_model.assert_called()
        self.assertIsNone(prompt_tuner.init_metric)
        self.assertIsNone(prompt_tuner.final_metric)
        self.assertEqual(prompt_tuner._model, self.mock_model)

    def test_initialization(self):
        """Testing the initialization of PromptTuner"""

        self._test_init(self.prompt_tuner)

    def test_initialization_without_model(self):
        """
        Testing the initialization of PromptTuner when
        the model is not provided and default model is launching
        """

        patcher = patch("coolprompt.assistant.DefaultLLM.init")
        mock_model_init = patcher.start()
        mock_model_init.return_value = self.mock_model
        prompt_tuner = PromptTuner()

        self._test_init(prompt_tuner)

        patcher.stop()

    def test_get_task_prompt_template_classification(self):
        """Testing that PromptTuner is using proper classification template"""

        self.assertEqual(
            self.prompt_tuner.get_task_prompt_template(
                "classification", "reflective"
            ),
            CLASSIFICATION_TASK_TEMPLATE,
        )
        self.assertEqual(
            self.prompt_tuner.get_task_prompt_template(
                "classification", "hype"
            ),
            CLASSIFICATION_TASK_TEMPLATE_HYPE,
        )

    def test_get_task_prompt_template_generation(self):
        """Testing that PromptTuner is using proper generation template"""

        self.assertEqual(
            self.prompt_tuner.get_task_prompt_template(
                "generation", "reflective"
            ),
            GENERATION_TASK_TEMPLATE,
        )
        self.assertEqual(
            self.prompt_tuner.get_task_prompt_template("generation", "hype"),
            GENERATION_TASK_TEMPLATE_HYPE,
        )

    def test_get_task_prompt_template_unsupported_task(self):
        """
        Testing that using get_task_prompt_template raises
        an exception when the unsupported task type is provided
        """

        with self.assertRaises(ValueError):
            self.prompt_tuner.get_task_prompt_template("unknown task", "hype")

    def test_get_task_prompt_template_unsupported_method(self):
        """
        Testing that using get_task_prompt_template raises
        an exception when the unsupported method is provided
        """

        with self.assertRaises(ValueError):
            self.prompt_tuner.get_task_prompt_template(
                "classification", "unknown method"
            )

    def test_run_unsupported_method(self):
        """
        Testing that run raises an exception when
        the unsupported method name is provided
        """

        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT, method="unsupported method"
            )

    def test_run_dataset_but_no_target(self):
        """
        Testing that run raises an exception when
        the dataset is provided without target
        """

        with self.assertRaises(ValueError):
            self.prompt_tuner.run(self.START_PROMPT, dataset=[])

    def test_run_target_is_smaller_than_dataset(self):
        """
        Testing that run raises an exception when
        the length of dataset doesn't match the length of targets
        """

        with self.assertRaises(ValueError):
            self.prompt_tuner.run(self.START_PROMPT, dataset=["s"], target=[])

    def test_run_incorrect_metric(self):
        """
        Testing that run raises an exception when
        the incorrect or unsupported metric name is provided
        """

        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT,
                dataset=[],
                target=[],
                metric="incorrect metric",
            )

    def test_hype_optimizer_without_dataset(self):
        """
        Testing the work of HyPE optimizer when the dataset is not provided
        """

        final_prompt = self.prompt_tuner.run(self.START_PROMPT)
        self.mock_hype_optimizer.assert_called_once_with(
            self.mock_model, self.START_PROMPT
        )
        self.assertEqual(final_prompt, self.FINAL_PROMPT)

    def test_run_metrics_check(self):
        """
        Testing that the metrics will be evaluated after optimization
        when the dataset is provided

        Also testing that the HyPE optimizer is working when
        the dataset is provided

        Also testing that the default metric will be loaded if
        no metric name is provided
        """

        self.prompt_tuner.run(
            self.START_PROMPT,
            dataset=["sample 1", "sample 2", "sample 3"],
            target=[1, 2, 3],
        )
        self.assertEqual(self.prompt_tuner.init_metric, 0.5)
        self.assertEqual(self.prompt_tuner.final_metric, 0.5)
        self.mock_evaluator_init.assert_called_once_with(
            ANY,
            self.mock_model,
            "meteor",  # check for default metric if None is passed
        )

    def test_run_reflective_without_problem_description(self):
        """
        Testing that run raises an exception when
        the reflective optimizer is called without problem description
        """

        with self.assertRaises(ValueError):
            self.prompt_tuner.run(self.START_PROMPT, method="reflective")

    def test_run_reflective_without_dataset(self):
        """
        Testing that run raises an exception when
        the reflective optimizer is called without dataset
        """

        with self.assertRaises(ValueError):
            self.prompt_tuner.run(
                self.START_PROMPT, method="reflective", problem_description=""
            )

    def test_run_reflective_empty_dataset(self):
        """
        Testing that run raises an exception when
        the reflective optimizer is called with empty or small dataset

        P.S. small dataset will raise an exception when
        is tried to be splitted into train/val split
        """

        with self.assertRaises(Exception):
            self.prompt_tuner.run(
                self.START_PROMPT,
                dataset=[],
                target=[],
                method="reflective",
                problem_description="",
            )

    def test_run_reflective(self):
        """Testing the work of reflective optimizer"""

        self.assertEqual(
            self.prompt_tuner.run(
                self.START_PROMPT,
                dataset=["1", "2"],
                target=["1", "2"],
                method="reflective",
                problem_description="",
            ),
            self.FINAL_PROMPT,
        )
        self.assertEqual(self.prompt_tuner.init_metric, 0.5)
        self.assertEqual(self.prompt_tuner.final_metric, 0.5)
