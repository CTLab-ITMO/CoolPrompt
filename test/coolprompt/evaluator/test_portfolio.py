import unittest

from coolprompt.evaluator.portfolio import select_portfolio


class TestPortfolio(unittest.TestCase):
    def test_picks_higher_val(self):
        r = select_portfolio({
            "pe2": ("PROMPT_A", 0.80),
            "pe2_sgr": ("PROMPT_B", 0.85),
        })
        self.assertEqual(r, ("pe2_sgr", "PROMPT_B", 0.85))

    def test_tie_prefers_first_listed(self):
        r = select_portfolio({
            "pe2": ("A", 0.8), "pe2_sgr": ("B", 0.8),
        })
        self.assertEqual(r[0], "pe2")


if __name__ == "__main__":
    unittest.main()
