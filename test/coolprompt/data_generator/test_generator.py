import unittest
from unittest.mock import MagicMock, patch
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.data_generator.pydantic_formatters import (
    ProblemDescriptionStructuredOutputSchema,
    ClassificationTaskExample,
    ClassificationTaskStructuredOutputSchema,
    GenerationTaskExample,
    GenerationTaskStructuredOutputSchema,
)
from coolprompt.utils.prompt_templates.data_generator_templates import (
    PROBLEM_DESCRIPTION_TEMPLATE,
    CLASSIFICATION_DATA_GENERATING_TEMPLATE,
    GENERATION_DATA_GENERATING_TEMPLATE,
    CLASSIFICATION_CORNER_CASE_GENERATING_TEMPLATE,
    GENERATION_CORNER_CASE_GENERATING_TEMPLATE,
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
            self.generator._generate("Request", None, "foo"), "bar"
        )
        self.mock_model.invoke.assert_called_once_with("Request")

    def test_generate_problem_description(self):
        """Testing problem description generator"""

        with patch(
            "coolprompt.data_generator.generator"
            + ".SyntheticDataGenerator._generate"
        ) as self._generate_mock:
            self._generate_mock.return_value = "problem"
            self.assertEqual(
                self.generator._generate_problem_description("prompt"),
                "problem",
            )
            self._generate_mock.assert_called_once_with(
                PROBLEM_DESCRIPTION_TEMPLATE.format(prompt="prompt"),
                ProblemDescriptionStructuredOutputSchema,
                "problem_description",
            )

    def test_convert_dataset_of_cls_examples(self):
        """Test dataset of classification examples conversion"""

        examples = [ClassificationTaskExample(input="in", output="out")]
        self.assertTupleEqual(
            self.generator._convert_dataset(examples), (["in"], ["out"])
        )

    def test_convert_dataset_of_gen_examples(self):
        """Test dataset of generation examples conversion"""

        examples = [GenerationTaskExample(input="in", output="out")]
        self.assertTupleEqual(
            self.generator._convert_dataset(examples), (["in"], ["out"])
        )

    def test_convert_dataset_of_dict_examples(self):
        """Test dataset of generation examples conversion"""

        examples = [{"input": "in", "output": "out"}]
        self.assertTupleEqual(
            self.generator._convert_dataset(examples), (["in"], ["out"])
        )

    def test_generate_cls_dataset(self):
        """Test generation of classification dataset"""

        generate_patcher = patch(
            "coolprompt.data_generator.generator"
            + ".SyntheticDataGenerator._generate"
        )
        generate_mock = generate_patcher.start()
        self.addCleanup(generate_patcher.stop)

        problem_description = "problem"
        num_samples = 20
        regular_request = CLASSIFICATION_DATA_GENERATING_TEMPLATE.format(
            problem_description=problem_description, num_samples=12
        )
        corner_request = CLASSIFICATION_CORNER_CASE_GENERATING_TEMPLATE.format(
            problem_description=problem_description, num_samples=8
        )
        schema = ClassificationTaskStructuredOutputSchema

        generate_mock.side_effect = [
            [ClassificationTaskExample(input="regular in", output="regular out")],
            [ClassificationTaskExample(input="corner in", output="corner out")],
        ]
        self.assertTupleEqual(
            self.generator.generate(
                prompt="prompt",
                task=Task.CLASSIFICATION,
                problem_description=problem_description,
                num_samples=num_samples,
            ),
            (
                ["regular in", "corner in"],
                ["regular out", "corner out"],
                problem_description,
            ),
        )
        self.assertEqual(generate_mock.call_count, 2)
        generate_mock.assert_any_call(regular_request, schema, "examples")
        generate_mock.assert_any_call(corner_request, schema, "examples")

    def test_generate_gen_dataset(self):
        """Test generation of generation dataset"""

        generate_patcher = patch(
            "coolprompt.data_generator.generator"
            + ".SyntheticDataGenerator._generate"
        )
        generate_mock = generate_patcher.start()
        self.addCleanup(generate_patcher.stop)

        problem_description = "problem"
        num_samples = 20
        regular_request = GENERATION_DATA_GENERATING_TEMPLATE.format(
            problem_description=problem_description, num_samples=12
        )
        corner_request = GENERATION_CORNER_CASE_GENERATING_TEMPLATE.format(
            problem_description=problem_description, num_samples=8
        )
        schema = GenerationTaskStructuredOutputSchema

        generate_mock.side_effect = [
            [GenerationTaskExample(input="regular in", output="regular out")],
            [GenerationTaskExample(input="corner in", output="corner out")],
        ]
        self.assertTupleEqual(
            self.generator.generate(
                prompt="prompt",
                task=Task.GENERATION,
                problem_description=problem_description,
                num_samples=num_samples,
            ),
            (
                ["regular in", "corner in"],
                ["regular out", "corner out"],
                problem_description,
            ),
        )
        self.assertEqual(generate_mock.call_count, 2)
        generate_mock.assert_any_call(regular_request, schema, "examples")
        generate_mock.assert_any_call(corner_request, schema, "examples")

    def test_generate_dataset_without_problem_description(self):
        """Test generation of classification dataset"""

        generate_patcher = patch(
            "coolprompt.data_generator.generator"
            + ".SyntheticDataGenerator._generate"
        )
        generate_mock = generate_patcher.start()
        self.addCleanup(generate_patcher.stop)
        problem_description_patcher = patch(
            "coolprompt.data_generator.generator"
            + ".SyntheticDataGenerator._generate_problem_description"
        )
        problem_description_mock = problem_description_patcher.start()
        self.addCleanup(problem_description_patcher.stop)
        problem_description_mock.return_value = "problem"

        num_samples = 20
        regular_request = GENERATION_DATA_GENERATING_TEMPLATE.format(
            problem_description="problem", num_samples=12
        )
        corner_request = GENERATION_CORNER_CASE_GENERATING_TEMPLATE.format(
            problem_description="problem", num_samples=8
        )
        schema = GenerationTaskStructuredOutputSchema

        generate_mock.side_effect = [
            [GenerationTaskExample(input="regular in", output="regular out")],
            [GenerationTaskExample(input="corner in", output="corner out")],
        ]
        self.assertTupleEqual(
            self.generator.generate(
                prompt="prompt", task=Task.GENERATION, num_samples=num_samples
            ),
            (
                ["regular in", "corner in"],
                ["regular out", "corner out"],
                "problem",
            ),
        )
        problem_description_mock.assert_called_once_with("prompt")
        self.assertEqual(generate_mock.call_count, 2)
        generate_mock.assert_any_call(regular_request, schema, "examples")
        generate_mock.assert_any_call(corner_request, schema, "examples")
