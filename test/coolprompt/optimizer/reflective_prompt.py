import sys
import os
import unittest
from unittest.mock import MagicMock, patch, ANY

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)

from coolprompt.optimizer.reflective_prompt import reflectiveprompt


class TestReflectivePrompt(unittest.TestCase):

    def setUp(self):
        self.mock_evoluter = MagicMock()
        self.mock_evoluter.evolution.return_value = "Evoluted prompt"
        self.patcher = patch(
            'coolprompt.optimizer.reflective_prompt' +
            '.run.ReflectiveEvoluter.__new__'
        )
        self.mock_evoluter_init = self.patcher.start()
        self.mock_evoluter_init.return_value = self.mock_evoluter

    def tearDown(self):
        self.patcher.stop()

    def test_reflective_prompt(self):
        final_prompt = reflectiveprompt(
            ANY,
            (ANY, ANY, ANY, ANY),
            ANY,
            ANY,
            ANY
        )
        self.assertEqual(final_prompt, 'Evoluted prompt')
        self.mock_evoluter_init.assert_called_once()
