import unittest

from coolprompt.utils.arithmetics import extract_number_from_text
from coolprompt.evaluator.metrics import ExactMatchMetric


class TestExtractNumber(unittest.TestCase):
    def test_extracts_last_number(self):
        self.assertEqual(
            extract_number_from_text("step one, then 18"), "18"
        )

    def test_no_number_returns_empty(self):
        self.assertEqual(
            extract_number_from_text("I cannot solve this"), ""
        )

    def test_empty_string_returns_empty(self):
        self.assertEqual(extract_number_from_text(""), "")


class TestExactMatchNonNumeric(unittest.TestCase):
    def test_non_numeric_output_counts_wrong_not_crash(self):
        """A non-numeric output must score 0, never raise."""
        m = ExactMatchMetric()
        scores = m._compute_raw(
            outputs=["no number here", "the answer is 18"],
            targets=["42", "18"],
            dataset=None,
        )
        self.assertEqual(scores, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
