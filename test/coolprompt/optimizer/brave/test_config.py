import tempfile
import unittest
from pathlib import Path

from coolprompt.optimizer.brave.utils import (
    BRAVEConfig,
    load_brave_config_from_yaml,
    reranking_population,
)
from coolprompt.optimizer.reflective_prompt.prompt import Prompt


class TestBRAVEConfig(unittest.TestCase):

    def test_default_actions_are_supported_by_evoluter(self):
        self.assertEqual(
            BRAVEConfig().actions,
            ["crossover", "elitist_mutation"],
        )

    def test_yaml_profile_overrides_defaults_and_ignores_unknown_fields(self):
        config_text = """
defaults:
  max_steps: 100
  population_size: 8
  unknown_option: ignored
profiles:
  fast:
    max_steps: 5
    initial_budget_tokens: 1200
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brave.yaml"
            path.write_text(config_text, encoding="utf-8")
            config = load_brave_config_from_yaml(str(path), profile="fast")

        self.assertEqual(config.max_steps, 5)
        self.assertEqual(config.population_size, 8)
        self.assertEqual(config.initial_budget_tokens, 1200)
        self.assertFalse(hasattr(config, "unknown_option"))

    def test_yaml_loader_rejects_unknown_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brave.yaml"
            path.write_text("profiles:\n  balanced: {}\n", encoding="utf-8")

            with self.assertRaisesRegex(KeyError, "missing"):
                load_brave_config_from_yaml(str(path), profile="missing")

    def test_reranking_population_orders_by_descending_score(self):
        population = [
            Prompt("low", score=0.1),
            Prompt("high", score=0.9),
            Prompt("middle", score=0.5),
        ]

        result = reranking_population(population)

        self.assertEqual([prompt.text for prompt in result], ["high", "middle", "low"])


if __name__ == "__main__":
    unittest.main()
