import unittest
from unittest.mock import patch, ANY

from coolprompt.utils.validation import validate_model


class TestValidateModel(unittest.TestCase):

    def setUp(self):
        self.patcher = patch(
            'coolprompt.utils.validation.isinstance'
        )
        self.isinstance_mock = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_validate_model_correct(self):
        self.isinstance_mock.return_value = True
        validate_model(ANY)

    def test_validate_model_incorrect(self):
        self.isinstance_mock.return_value = False
        with self.assertRaises(TypeError):
            validate_model(ANY)
