import unittest
from unittest.mock import MagicMock

from coolprompt.optimizer.hype import hype_optimizer


class TestHypeOptimizer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.invoke.return_value = \
            "[PROMPT_START]Final Prompt[PROMPT_END]"

    def test_hype_optimizer(self):
        """Testing the contract of HyPE optimization function"""

        self.assertEqual(
            "Final Prompt",
            hype_optimizer(self.mock_model, "Input Prompt")
        )
        self.mock_model.invoke.assert_called_once()
