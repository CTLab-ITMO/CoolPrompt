import unittest

import numpy as np

from coolprompt.optimizer.brave.actions import ActionResult
from coolprompt.optimizer.brave.population_diversity import (
    PopulationDiversityManager,
)
from coolprompt.optimizer.reflective_prompt.prompt import Prompt


class TestPopulationDiversityManager(unittest.TestCase):

    def setUp(self):
        self.manager = PopulationDiversityManager(
            use_bert=False,
            use_hierarchical=False,
        )

    def test_single_prompt_is_fully_diverse(self):
        self.assertEqual(
            self.manager.compute_diversity([Prompt("only prompt")]),
            1.0,
        )

    def test_identical_prompts_have_zero_diversity(self):
        diversity = self.manager.compute_diversity(
            [
                Prompt("same prompt"),
                Prompt("same prompt"),
            ]
        )

        self.assertAlmostEqual(diversity, 0.0)

    def test_near_duplicate_filter_keeps_highest_ranked_prompt(self):
        population = [
            Prompt("best", score=0.9),
            Prompt("duplicate", score=0.8),
            Prompt("different", score=0.7),
        ]
        similarity = np.array(
            [
                [1.0, 0.99, 0.1],
                [0.99, 1.0, 0.1],
                [0.1, 0.1, 1.0],
            ]
        )

        kept, reduced, indices = self.manager._filter_near_duplicates(
            population,
            similarity,
            duplicate_threshold=0.95,
        )

        self.assertEqual([prompt.text for prompt in kept], ["best", "different"])
        np.testing.assert_array_equal(indices, [0, 2])
        self.assertEqual(reduced.shape, (2, 2))

    def test_maintain_diversity_sorts_population_within_limit(self):
        population = [
            Prompt("low", score=0.1),
            Prompt("high", score=0.9),
        ]

        result = self.manager.maintain_diversity(population, max_size=2)

        self.assertEqual([prompt.text for prompt in result], ["high", "low"])


class TestActionResult(unittest.TestCase):

    def test_optional_fields_have_independent_defaults(self):
        first = ActionResult("a", delta_quality=0.1, cost_tokens=10)
        second = ActionResult("b", delta_quality=0.0, cost_tokens=20)

        first.payload["value"] = 1

        self.assertEqual(second.payload, {})
        self.assertFalse(first.improved)


if __name__ == "__main__":
    unittest.main()
