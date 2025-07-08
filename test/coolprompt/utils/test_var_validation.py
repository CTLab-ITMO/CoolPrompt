import unittest
from unittest.mock import patch, ANY

from coolprompt.utils.var_validation import (
    Method,
    Task,
    validate_dataset,
    validate_method,
    validate_model,
    validate_problem_description,
    validate_start_prompt,
    validate_target,
    validate_task,
    validate_validation_size,
    validate_verbose,
)


class TestValidateModel(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("coolprompt.utils.var_validation.isinstance")
        self.isinstance_mock = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_validate_model_correct(self):
        """Testing the work of validate_model function"""

        self.isinstance_mock.return_value = True
        validate_model(ANY)

    def test_validate_model_incorrect(self):
        """
        Testing that the validate_model raises an exception when
        the given model is not a BaseLanguageModel instance
        """

        self.isinstance_mock.return_value = False
        with self.assertRaises(TypeError):
            validate_model(ANY)


class TestValidateVerbose(unittest.TestCase):
    def test_valid_verbose(self):
        """Testing the work of validate_verbose function"""

        for value in [0, 1, 2]:
            with self.subTest(value=value):
                validate_verbose(value)

    def test_invalid_verbose(self):
        """Testing that validate_verbose function raises ValueError when
        provided verbose is outside [0.0, 1.0] or is not float"""

        for value in [-1, 3, "1", None, 2.5]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_verbose(value)


class TestValidateStartPrompt(unittest.TestCase):
    def test_valid_start_prompt(self):
        """Testing the work of validate_start_prompt function"""

        value = "start prompt"
        validate_start_prompt(value)

    def test_invalid_start_prompt(self):
        """Testing that validate_start_prompt function raises TypeError
        when provided start prompt is not a string"""

        for value in [1, None, 2.5]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    validate_start_prompt(value)


class TestValidateTask(unittest.TestCase):
    def test_valid_task(self):
        """Testing the work of validate_task function"""

        for value in ["generation", "classification"]:
            with self.subTest(value=value):
                self.assertEqual(validate_task(value),
                                 Task._value2member_map_[value])

    def test_invalid_task_type(self):
        """Testing that validate_task function raises TypeError
        when provided task type is not a string"""

        for value in [1, None, 2.5]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    validate_task(value)

    def test_invalid_task_value(self):
        """Testing that validate_task function raises ValueError when
        provided task type is not one of ["generation", "classification"]"""

        value = "Q/A"
        with self.assertRaises(ValueError):
            validate_task(value)


class TestValidateMethod(unittest.TestCase):
    def test_valid_method(self):
        """Testing the work of validate_method function"""

        for value in ["hype", "reflective", "distill"]:
            with self.subTest(value=value):
                self.assertEqual(validate_method(value),
                                 Method._value2member_map_[value])

    def test_invalid_method_type(self):
        """Testing that validate_method function raises TypeError
        when provided method is not a string"""

        for value in [1, None, 2.5]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    validate_method(value)

    def test_invalid_method_value(self):
        """Testing that validate_method function raises ValueError when
        provided method is not one of ["hype", "reflective", "distill"]"""

        value = "unknown method"
        with self.assertRaises(ValueError):
            validate_method(value)


class TestValidateDataset(unittest.TestCase):
    def test_valid_dataset_provided(self):
        """Testing the work of validate_dataset function
        when dataset is provided"""

        dataset = ["sample 1", "sample 2", "sample 3"]
        target = [1, 2, 3]
        method = Method.HYPE
        validate_dataset(dataset, target, method)

    def test_valid_dataset_not_provied(self):
        """Testing the work of validate_dataset function when
        dataset is not provided"""

        validate_dataset(None, [1, 2, 3], Method.HYPE)

    def test_invalid_dataset_provided_type(self):
        """Testing that validate_dataset function raises TypeError when
        provided dataset is not Iterable"""

        dataset = 1
        target = [1, 2, 3]
        method = Method.HYPE
        with self.assertRaises(TypeError):
            validate_dataset(dataset, target, method)

    def test_invalid_dataset_provided_target_not_provided(self):
        """Testing that validate_dataset function raises ValueError when
        dataset is provided but target is None"""

        with self.assertRaises(ValueError):
            validate_dataset(
                ["sample 1", "sample 2", "sample 3"], None, Method.HYPE
            )

    def test_invalid_dataset_not_provided_method_requires(self):
        """Testing that validate_dataset function raises ValueError when
        dataset is not provided but the ReflectivePrompt method requires it"""

        with self.assertRaises(ValueError):
            validate_dataset(None, None, Method.REFLECTIVE)


class TestValidateTarget(unittest.TestCase):
    def test_valid_target(self):
        """Testing the work of validate_target function"""
        dataset = ["sample 1", "sample 2", "sample 3"]
        target = [1, 2, 3]
        validate_target(target, dataset)

    def test_invalid_target_type(self):
        """Testing that validate_target function raises TypeError when
        targe is not Iterable"""

        dataset = ["sample 1", "sample 2", "sample 3"]
        target = 1
        with self.assertRaises(TypeError):
            validate_target(target, dataset)

    def test_invalid_target_length_mismatch(self):
        """Testing that validate_dataset function raises ValueError when
        the target length does not equal the dataset length"""

        dataset = ["sample 1"]
        target = [1, 2, 3]
        with self.assertRaises(ValueError):
            validate_target(target, dataset)

    def test_invalid_target_provided_dataset_not_provided(self):
        """Testing that validate_dataset function raises ValueError when
        target is provided but dataset is None"""

        with self.assertRaises(ValueError):
            validate_target([1, 2, 3], None)


class TestValidateProblemDescription(unittest.TestCase):
    def test_valid_problem_description_provided(self):
        """Testing the work of validate_problem_description function when
        the problem description is provided"""

        problem_description = "description"
        method = Method.HYPE
        validate_problem_description(problem_description, method)

    def test_valid_problem_description_not_provided(self):
        """Testing the work of validate_problem_description function when
        the problem description is not provided"""

        validate_problem_description(None, Method.HYPE)

    def test_invalid_problem_description_type(self):
        """Testing that validate_problem_description function raises
        TypeError when the problem description is not a string"""

        with self.assertRaises(TypeError):
            validate_problem_description(1, Method.HYPE)

    def test_invalid_problem_description_not_provided_method_requires(self):
        """Testing that validate_problem_description function raises
        TypeError when the problem description is not provided but using
        the ReflectivePrompt method"""

        with self.assertRaises(ValueError):
            validate_problem_description(None, Method.REFLECTIVE)


class TestValidateValidationSize(unittest.TestCase):

    def test_valid_validation_size(self):
        """Testing the work of validate_validation_size function"""

        for value in [0.0, 0.5, 1.0]:
            with self.subTest(value=value):
                validate_validation_size(value)

    def test_invalid_validation_size(self):
        """Testing that validate_validation_size function raises
        ValueError when the provided validation size is outside [0.0, 1.0]
        or not a float"""

        for value in [-1, -0.5, None, "1", [0.0], 1.5, 2]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_validation_size(value)
