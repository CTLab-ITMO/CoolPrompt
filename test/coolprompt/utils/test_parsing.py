import unittest

from coolprompt.utils.parsing import extract_answer, extract_json


class TestExtractAnswer(unittest.TestCase):
    TAGS = ("<a>", "</a>")
    FORMAT_MISMATCH_LABEL = -42

    def test_extract_answer_correct(self):
        """Testing the work of extract_answer method"""

        result = extract_answer("<a>Answer</a>", self.TAGS)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Answer")

    def test_extract_answer_incorrect_tag(self):
        """
        Testing that the extract_answer will
        return format_mismatch_label when
        the tags in output don't match the given tags
        """

        result = extract_answer(
            "<a>Answer</b>", self.TAGS, self.FORMAT_MISMATCH_LABEL
        )
        self.assertIsInstance(result, int)
        self.assertEqual(result, self.FORMAT_MISMATCH_LABEL)

    def test_extract_answer_no_tags(self):
        """
        Testing that the extract_answer will
        return format_mismatch_label when
        there is no tags in model outputs
        """

        self.assertEqual(
            extract_answer("Answer", self.TAGS, self.FORMAT_MISMATCH_LABEL),
            self.FORMAT_MISMATCH_LABEL,
        )

    def test_extract_answer_no_opening_tag(self):
        """
        Testing that the extract_answer will
        return format_mismatch_label when
        there is no opening tag in model outputs
        """

        self.assertEqual(
            extract_answer(
                "Answer</a>", self.TAGS, self.FORMAT_MISMATCH_LABEL
            ),
            self.FORMAT_MISMATCH_LABEL,
        )

    def test_extract_answer_no_closing_tag(self):
        """
        Testing that the extract_answer will
        return format_mismatch_label when
        there is no closing tag in model outputs
        """

        self.assertEqual(
            extract_answer(
                "<a>Answer<", self.TAGS, self.FORMAT_MISMATCH_LABEL
            ),
            self.FORMAT_MISMATCH_LABEL,
        )


class TestExtractJson(unittest.TestCase):
    CORRECT_JSON = {"test": "value"}

    def test_extract_json_correct(self):
        result = extract_json('{{"test": "value"}}')
        self.assertIsInstance(result, dict)
        self.assertEqual(result, self.CORRECT_JSON)

    def test_extract_json_different_quotes(self):
        for value in [
            "{{'test': 'value'}}",
            "{{'test': \"value\"}}",
            "{{\"test\": 'value'}}",
        ]:
            with self.subTest(value=value):
                self.assertEqual(extract_json(value), self.CORRECT_JSON)

    def test_extract_json_whitespaces(self):
        for value in [
            '{{"test":"value"}}',
            '{{"test"   :   "value"}}',
            '{{"test"  :"value"}}',
            '  {{"test": "value"}}    ',
        ]:
            with self.subTest(value=value):
                self.assertEqual(extract_json(value), self.CORRECT_JSON)

    def test_extract_json_dirty(self):
        for value in [
            '{{"test": "value"}} sample text',
            'sample text {{"test": "value"}}',
            'sample text {{"test": "value"}} sample text',
        ]:
            with self.subTest(value=value):
                self.assertEqual(extract_json(value), self.CORRECT_JSON)

    def test_extract_json_other_curvy_brackets(self):
        for value in [
            '{{"test": "value"}} {{"other_test": "value}}',
            '{{{{"test": "value}}' '{{"test": "value"}}}}',
        ]:
            with self.subTest(value=value):
                self.assertEqual(extract_json(value), self.CORRECT_JSON)

    def test_extract_json_invalid(self):
        for value in [
            "{{test: value}}",
            '{{"test": value}}',
            '{{"test": "value"',
            '"test": "value"}}',
            "{{'test': 'value\"}}",
            '{{"test\': "value"}}',
        ]:
            with self.subTest(value=value):
                self.assertEqual(extract_json(value), None)
