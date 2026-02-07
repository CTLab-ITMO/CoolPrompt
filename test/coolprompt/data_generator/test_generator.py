import unittest
from unittest.mock import MagicMock, patch
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.data_generator.pydantic_formatters import (
    ProblemDescriptionStructuredOutputSchema,
    ClassificationTaskExample,
    ClassificationTaskStructuredOutputSchema,
    GenerationTaskExample,
    GenerationTaskStructuredOutputSchema
)
from coolprompt.utils.prompt_templates.data_generator_templates import (
    PROBLEM_DESCRIPTION_TEMPLATE,
    CLASSIFICATION_DATA_GENERATING_TEMPLATE,
    GENERATION_DATA_GENERATING_TEMPLATE
)
from coolprompt.utils.enums import Task


class TestGenerator(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock(spec=BaseLanguageModel)
        self.mock_model.with_structured_output.return_value = self.mock_model

        self.generator = SyntheticDataGenerator(self.mock_model)

    def test_initialization(self):
        """Testing initialization of Generator"""

        self.assertEqual(self.generator.model, self.mock_model)

    def test_inner_generate(self):
        """Testing inner generate function"""

        self.mock_model.invoke.return_value = '{"foo": "bar"}'
        self.assertEqual(
            self.generator._generate("Request", None, "foo"),
            "bar"
        )
        self.mock_model.invoke.assert_called_once_with("Request")

    def test_generate_problem_description(self):
        """Testing problem description generator"""

        with patch(
            "coolprompt.data_generator.generator" +
            ".SyntheticDataGenerator._generate"
        ) as self._generate_mock:
            self._generate_mock.return_value = "problem"
            self.assertEqual(
                self.generator._generate_problem_description("prompt"),
                "problem"
            )
            self._generate_mock.assert_called_once_with(
                PROBLEM_DESCRIPTION_TEMPLATE.format(
                    prompt="prompt"
                ),
                ProblemDescriptionStructuredOutputSchema,
                "problem_description"
            )

    def test_convert_dataset_of_cls_examples(self):
        """Test dataset of classification examples conversion"""

        examples = [ClassificationTaskExample(input="in", output="out")]
        self.assertTupleEqual(
            self.generator._convert_dataset(examples),
            (["in"], ["out"])
        )

    def test_convert_dataset_of_gen_examples(self):
        """Test dataset of generation examples conversion"""

        examples = [GenerationTaskExample(input="in", output="out")]
        self.assertTupleEqual(
            self.generator._convert_dataset(examples),
            (["in"], ["out"])
        )

    def test_convert_dataset_of_dict_examples(self):
        """Test dataset of generation examples conversion"""

        examples = [{"input": "in", "output": "out"}]
        self.assertTupleEqual(
            self.generator._convert_dataset(examples),
            (["in"], ["out"])
        )

    def test_generate_cls_dataset(self):
        """Test generation of classification dataset"""

        self._generate_patcher = patch(
            "coolprompt.data_generator.generator" +
            ".SyntheticDataGenerator._generate"
        )
        self._generate_mock = self._generate_patcher.start()

        problem_description = "problem"
        num_samples = 20
        request = CLASSIFICATION_DATA_GENERATING_TEMPLATE.format(
            problem_description=problem_description,
            num_samples=num_samples
        )
        schema = ClassificationTaskStructuredOutputSchema

        self._generate_mock.return_value = [
            ClassificationTaskExample(input="in", output="out")
        ]
        self.assertTupleEqual(
            self.generator.generate(
                prompt="prompt",
                task=Task.CLASSIFICATION,
                problem_description=problem_description,
                num_samples=num_samples
            ),
            (
                ["in"],
                ["out"],
                problem_description
            )
        )
        self._generate_mock.assert_called_once_with(
            request,
            schema,
            "examples"
        )

        self._generate_patcher.stop()

    def test_generate_gen_dataset(self):
        """Test generation of generation dataset"""

        self._generate_patcher = patch(
            "coolprompt.data_generator.generator" +
            ".SyntheticDataGenerator._generate"
        )
        self._generate_mock = self._generate_patcher.start()

        problem_description = "problem"
        num_samples = 20
        request = GENERATION_DATA_GENERATING_TEMPLATE.format(
            problem_description=problem_description,
            num_samples=num_samples
        )
        schema = GenerationTaskStructuredOutputSchema

        self._generate_mock.return_value = [
            GenerationTaskExample(input="in", output="out")
        ]
        self.assertTupleEqual(
            self.generator.generate(
                prompt="prompt",
                task=Task.GENERATION,
                problem_description=problem_description,
                num_samples=num_samples
            ),
            (
                ["in"],
                ["out"],
                problem_description
            )
        )
        self._generate_mock.assert_called_once_with(
            request,
            schema,
            "examples"
        )

        self._generate_patcher.stop()

    def test_generate_dataset_without_problem_description(self):
        """Test generation of classification dataset"""

        self._generate_patcher = patch(
            "coolprompt.data_generator.generator" +
            ".SyntheticDataGenerator._generate"
        )
        self._generate_mock = self._generate_patcher.start()
        self._generate_problem_description_patcher = patch(
            "coolprompt.data_generator.generator" +
            ".SyntheticDataGenerator._generate_problem_description"
        )
        self._generate_problem_description_mock = (
            self._generate_problem_description_patcher.start()
        )
        self._generate_problem_description_mock.return_value = "problem"

        num_samples = 20
        request = GENERATION_DATA_GENERATING_TEMPLATE.format(
            problem_description="problem",
            num_samples=num_samples
        )
        schema = GenerationTaskStructuredOutputSchema

        self._generate_mock.return_value = [
            GenerationTaskExample(input="in", output="out")
        ]
        self.assertTupleEqual(
            self.generator.generate(
                prompt="prompt",
                task=Task.GENERATION,
                num_samples=num_samples
            ),
            (
                ["in"],
                ["out"],
                "problem"
            )
        )
        self._generate_problem_description_mock.assert_called_once_with(
            "prompt"
        )
        self._generate_mock.assert_called_once_with(
            request,
            schema,
            "examples"
        )

        self._generate_patcher.stop()
        self._generate_problem_description_patcher.stop()
