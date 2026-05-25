import unittest

from coolprompt.optimizer.pe2.node import Node
from coolprompt.optimizer.pe2.trainer import PE2Trainer


class TestTiebreak(unittest.TestCase):
    def _node(self, nid, prompt, val):
        n = Node(timestamp=0, id=nid, prompt=prompt)
        n.register_score(val, "val")
        return n

    def test_shorter_wins_equal_val(self):
        long_n = self._node(1, "x" * 50, 0.8)
        short_n = self._node(2, "x" * 10, 0.8)
        self.assertGreater(
            PE2Trainer._rank_key(short_n),
            PE2Trainer._rank_key(long_n),
        )

    def test_higher_val_wins_over_length(self):
        hi = self._node(1, "x" * 99, 0.9)
        lo = self._node(2, "x", 0.8)
        self.assertGreater(
            PE2Trainer._rank_key(hi),
            PE2Trainer._rank_key(lo),
        )


if __name__ == "__main__":
    unittest.main()
