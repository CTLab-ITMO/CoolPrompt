import unittest

from coolprompt.evaluator.ifeval_checkers import (
    check_instruction,
    SUPPORTED_INSTRUCTIONS,
)


class TestIFEvalCheckers(unittest.TestCase):
    def test_number_words_at_least(self):
        resp = "one two three four five six"
        ok = check_instruction(
            "length_constraints:number_words",
            resp,
            {"num_words": 5, "relation": "at least"},
        )
        self.assertTrue(ok)

    def test_number_words_at_least_fail(self):
        ok = check_instruction(
            "length_constraints:number_words",
            "too short",
            {"num_words": 5, "relation": "at least"},
        )
        self.assertFalse(ok)

    def test_keyword_existence(self):
        ok = check_instruction(
            "keywords:existence",
            "the quick brown fox",
            {"keywords": ["quick", "fox"]},
        )
        self.assertTrue(ok)

    def test_forbidden_words(self):
        ok = check_instruction(
            "keywords:forbidden_words",
            "a clean sentence",
            {"forbidden_words": ["banned"]},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "keywords:forbidden_words",
            "this is banned",
            {"forbidden_words": ["banned"]},
        )
        self.assertFalse(bad)

    def test_all_lowercase(self):
        ok = check_instruction(
            "change_case:english_lowercase",
            "all lower case here",
            {},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "change_case:english_lowercase",
            "Has Uppercase",
            {},
        )
        self.assertFalse(bad)

    def test_number_bullets(self):
        resp = "* one\n* two\n* three"
        ok = check_instruction(
            "detectable_format:number_bullet_lists",
            resp,
            {"num_bullets": 3},
        )
        self.assertTrue(ok)

    def test_json_format(self):
        ok = check_instruction(
            "detectable_format:json_format",
            '{"a": 1, "b": 2}',
            {},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "detectable_format:json_format",
            "not json",
            {},
        )
        self.assertFalse(bad)

    def test_end_checker(self):
        ok = check_instruction(
            "startend:end_checker",
            "blah blah That is all.",
            {"end_phrase": "That is all."},
        )
        self.assertTrue(ok)

    def test_unknown_instruction_returns_false(self):
        ok = check_instruction("nonexistent:thing", "x", {})
        self.assertFalse(ok)

    def test_supported_set_nonempty(self):
        self.assertIn(
            "keywords:existence", SUPPORTED_INSTRUCTIONS
        )


if __name__ == "__main__":
    unittest.main()
