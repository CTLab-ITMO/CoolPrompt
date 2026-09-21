import unittest
from unittest.mock import patch

import numpy as np

from coolprompt.optimizer.brave.controller import EVCController


class TestEVCController(unittest.TestCase):

    def _controller(self, **kwargs):
        defaults = {
            "actions": ["cheap", "expensive"],
            "feature_dim": 2,
            "use_neural_bandit": False,
            "uncertainty_penalty_beta": 0.0,
            "max_action_budget_share": 0.5,
            "seed": 7,
        }
        defaults.update(kwargs)
        return EVCController(**defaults)

    def test_select_action_excludes_action_above_budget_share(self):
        controller = self._controller()
        outcomes = {
            "cheap": (1.0, 10.0),
            "expensive": (10.0, 60.0),
        }

        with patch.object(
            controller,
            "_sample_benefit_cost",
            side_effect=lambda action, _: outcomes[action],
        ):
            action, scores = controller.select_action(
                x=np.array([1.0, 0.0]),
                remaining_budget_tokens=100,
            )

        self.assertEqual(action, "cheap")
        self.assertIn("cheap", scores)
        self.assertNotIn("expensive", scores)

    def test_select_action_returns_none_when_every_action_is_unaffordable(self):
        controller = self._controller(max_action_budget_share=1.0)

        with patch.object(
            controller,
            "_sample_benefit_cost",
            return_value=(1.0, 101.0),
        ):
            action, scores = controller.select_action(
                x=np.ones(2),
                remaining_budget_tokens=100,
            )

        self.assertIsNone(action)
        self.assertIsNone(scores)

    def test_update_records_success_and_realized_roi(self):
        controller = self._controller(alpha_roi_ema=0.5)

        controller.update(
            action="cheap",
            x_before=np.array([1.0, 0.0]),
            delta_quality=0.2,
            actual_cost_tokens=100,
            improved=True,
        )

        stats = controller.action_stats["cheap"]
        self.assertEqual(stats["trials"], 1.0)
        self.assertEqual(stats["success_count"], 1.0)
        self.assertAlmostEqual(stats["ema_roi"], 0.001)
        self.assertGreater(
            controller.benefit_models["cheap"].predictive_mean(np.array([1.0, 0.0])),
            0.0,
        )

    def test_kill_switch_disables_action_with_negative_roi(self):
        controller = self._controller(
            kill_switch_min_trials=2,
            kill_switch_roi_threshold=0.0,
            kill_switch_base_cooldown=3,
        )
        stats = controller.action_stats["cheap"]
        stats["trials"] = 2.0
        stats["ema_roi"] = -0.1

        self.assertTrue(controller._should_disable("cheap"))
        self.assertGreater(stats["disabled_until_step"], 0.0)


if __name__ == "__main__":
    unittest.main()
