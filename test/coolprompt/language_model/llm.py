import sys
import os
from langchain_community.llms import VLLM
import unittest
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)

from coolprompt.language_model.llm import DefaultLLM


class TestDefaultLLM(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.patcher = patch.object(
            VLLM,
            '__new__'
        )
        self.mock_vllm_init = self.patcher.start()
        self.mock_vllm_init.return_value = self.mock_model

    def tearDown(self):
        self.patcher.stop()

    def test_init(self):
        self.assertEqual(DefaultLLM.init(), self.mock_model)
