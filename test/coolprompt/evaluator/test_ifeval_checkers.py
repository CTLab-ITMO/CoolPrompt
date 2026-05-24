import json as _json
import unittest

from coolprompt.evaluator.ifeval_checkers import (
    check_instruction,
    SUPPORTED_INSTRUCTIONS,
)
from coolprompt.evaluator.metrics import IFEvalMetric


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

    def test_number_sentences_at_least(self):
        resp = "One sentence. Two sentence! Three sentence?"
        ok = check_instruction(
            "length_constraints:number_sentences",
            resp,
            {"num_sentences": 3, "relation": "at least"},
        )
        self.assertTrue(ok)

    def test_keyword_frequency_at_least(self):
        resp = "cat cat cat dog"
        ok = check_instruction(
            "keywords:frequency",
            resp,
            {"keyword": "cat", "frequency": 3,
             "relation": "at least"},
        )
        self.assertTrue(ok)

    def test_english_capital(self):
        ok = check_instruction(
            "change_case:english_capital",
            "ALL CAPS HERE",
            {},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "change_case:english_capital",
            "Mixed Case",
            {},
        )
        self.assertFalse(bad)

    def test_json_format_fenced(self):
        resp = '```json\n{"a": 1}\n```'
        ok = check_instruction(
            "detectable_format:json_format",
            resp,
            {},
        )
        self.assertTrue(ok)

    def test_number_highlighted(self):
        ok = check_instruction(
            "detectable_format:number_highlighted_sections",
            "*one* *two*",
            {"num_highlights": 2},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "detectable_format:number_highlighted_sections",
            "**bold** text",
            {"num_highlights": 1},
        )
        self.assertFalse(bad)

    def test_title(self):
        ok = check_instruction(
            "detectable_format:title",
            "Here is <<My Title>> for you",
            {},
        )
        self.assertTrue(ok)

    def test_quotation(self):
        ok = check_instruction(
            "startend:quotation",
            '"quoted"',
            {},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "startend:quotation",
            "not quoted",
            {},
        )
        self.assertFalse(bad)

    def test_postscript(self):
        ok = check_instruction(
            "detectable_content:postscript",
            "Main text.\nP.S. extra note",
            {"postscript_marker": "P.S."},
        )
        self.assertTrue(ok)

    def test_number_placeholders(self):
        ok = check_instruction(
            "detectable_content:number_placeholders",
            "[NAME] [DATE]",
            {"num_placeholders": 2},
        )
        self.assertTrue(ok)
        bad = check_instruction(
            "detectable_content:number_placeholders",
            "[docs](url)",
            {"num_placeholders": 1},
        )
        self.assertFalse(bad)

    def test_unknown_instruction_returns_false(self):
        ok = check_instruction("nonexistent:thing", "x", {})
        self.assertFalse(ok)

    def test_supported_set_nonempty(self):
        self.assertIn(
            "keywords:existence", SUPPORTED_INSTRUCTIONS
        )


class TestIFEvalMetric(unittest.TestCase):
    def _spec(self, ids, kwargs_list):
        return _json.dumps(
            {"instruction_id_list": ids, "kwargs": kwargs_list}
        )

    def test_all_constraints_satisfied_scores_one(self):
        target = self._spec(
            ["keywords:existence",
             "change_case:english_lowercase"],
            [{"keywords": ["fox"]}, {}],
        )
        metric = IFEvalMetric()
        score = metric.compute(
            outputs=["the quick brown fox"],
            targets=[target],
            dataset=["write about a fox"],
        )
        self.assertEqual(score, 1.0)

    def test_one_constraint_failed_scores_zero(self):
        target = self._spec(
            ["keywords:existence",
             "change_case:english_lowercase"],
            [{"keywords": ["fox"]}, {}],
        )
        metric = IFEvalMetric()
        score = metric.compute(
            outputs=["The Quick Brown FOX"],
            targets=[target],
            dataset=["write about a fox"],
        )
        self.assertEqual(score, 0.0)

    def test_mean_over_prompts(self):
        t1 = self._spec(
            ["keywords:existence"], [{"keywords": ["a"]}]
        )
        t2 = self._spec(
            ["keywords:existence"], [{"keywords": ["zzz"]}]
        )
        metric = IFEvalMetric()
        score = metric.compute(
            outputs=["a", "a"],
            targets=[t1, t2],
            dataset=["", ""],
        )
        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
