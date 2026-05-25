import unittest

from coolprompt.optimizer.pe2_sgr.proposer import SGRProposer
from coolprompt.optimizer.pe2_sgr.schemas import EditDecision


class TestShouldKeep(unittest.TestCase):
    def test_keep_when_no_edit(self):
        ed = EditDecision(
            editing_necessary=False,
            confidence="high", justification="ok",
        )
        self.assertTrue(SGRProposer._should_keep(ed))

    def test_edit_when_needed(self):
        ed = EditDecision(
            editing_necessary=True,
            confidence="high", justification="x",
        )
        self.assertFalse(SGRProposer._should_keep(ed))


if __name__ == "__main__":
    unittest.main()
