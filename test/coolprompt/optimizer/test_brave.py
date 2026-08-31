import unittest
from unittest.mock import MagicMock, patch

from coolprompt.optimizer.brave.run import BRAVEMethod, brave
from coolprompt.optimizer.brave.utils import BRAVEConfig
from coolprompt.utils.var_validation import validate_method


class TestBraveIntegration(unittest.TestCase):

    @patch("coolprompt.optimizer.brave.run.BRAVEEvoluter")
    def test_brave_returns_best_validation_prompt(self, evoluter_cls):
        evoluter = evoluter_cls.return_value
        evoluter.optimize.return_value = {
            "best_prompt": "training best",
            "best_val_prompt": "validation best",
        }
        dataset_split = (["train"], ["validation"], ["a"], ["b"])

        result = brave(
            model=MagicMock(),
            dataset_split=dataset_split,
            evaluator=MagicMock(),
            problem_description="Classify text",
            initial_prompt="Initial prompt",
            max_steps=7,
            initial_budget_tokens=1234,
        )

        self.assertEqual(result, "validation best")
        config = evoluter_cls.call_args.kwargs["config"]
        self.assertIsInstance(config, BRAVEConfig)
        self.assertEqual(config.max_steps, 7)
        self.assertEqual(config.initial_budget_tokens, 1234)
        evoluter.optimize.assert_called_once_with(
            initial_prompt="Initial prompt",
            problem_description="Classify text",
            train_data=["train"],
            train_targets=["a"],
            val_data=["validation"],
            val_targets=["b"],
        )

    def test_default_actions_are_executable(self):
        self.assertEqual(
            BRAVEConfig().actions,
            ["crossover", "elitist_mutation"],
        )

    def test_brave_is_a_supported_data_driven_method(self):
        method = validate_method("brave")
        self.assertIsInstance(method, BRAVEMethod)
        self.assertEqual(method.name, "brave")
        self.assertTrue(method.is_data_driven())


if __name__ == "__main__":
    unittest.main()
