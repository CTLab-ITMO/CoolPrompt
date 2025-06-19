import unittest

from coolprompt.utils.parsing import extract_answer


class TestExtractAnswer(unittest.TestCase):
    TAGS = ('<a>', '</a>')
    FORMAT_MISMATCH_LABEL = -42

    def test_extract_answer_correct(self):
        """Testing the work of extract_answer method"""

        result = extract_answer("<a>Answer</a>", self.TAGS)
        self.assertIsInstance(result, str)
        self.assertEqual(result, 'Answer')

    def test_extract_answer_incorrect_tag(self):
        """
        Testing that the extract_answer will
        return format_mismatch_label when
        the tags in output don't match the given tags
        """

        result = extract_answer(
            "<a>Answer</b>",
            self.TAGS,
            self.FORMAT_MISMATCH_LABEL
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
            extract_answer(
                "Answer",
                self.TAGS,
                self.FORMAT_MISMATCH_LABEL
            ),
            self.FORMAT_MISMATCH_LABEL
        )

    def test_extract_answer_no_opening_tag(self):
        """
        Testing that the extract_answer will
        return format_mismatch_label when
        there is no opening tag in model outputs
        """

        self.assertEqual(
            extract_answer(
                "Answer</a>",
                self.TAGS,
                self.FORMAT_MISMATCH_LABEL
            ),
            self.FORMAT_MISMATCH_LABEL
        )

    def test_extract_answer_no_closing_tag(self):
        """
        Testing that the extract_answer will
        return format_mismatch_label when
        there is no closing tag in model outputs
        """

        self.assertEqual(
            extract_answer(
                "<a>Answer<",
                self.TAGS,
                self.FORMAT_MISMATCH_LABEL
            ),
            self.FORMAT_MISMATCH_LABEL
        )
