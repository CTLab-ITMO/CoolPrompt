import unittest
from unittest.mock import MagicMock

from coolprompt.optimizer.naive import naive_optimizer


class TestNaiveOptimizer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.invoke.return_value = \
            "[PROMPT_START]Final Prompt[PROMPT_END]"

    def test_naive_optimizer(self):
        self.assertEqual(
            "Final Prompt",
            naive_optimizer(self.mock_model, "Input Prompt")
        )
        self.mock_model.invoke.assert_called_once()
