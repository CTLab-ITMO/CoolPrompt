from langchain_community.llms import VLLM
import unittest
from unittest.mock import MagicMock, patch

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
