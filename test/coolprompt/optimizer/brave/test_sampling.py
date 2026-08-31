import unittest

import numpy as np

from coolprompt.optimizer.brave.batch_sampler import (
    CurriculumStratifiedBatchSampler,
    StratifiedBatchSampler,
)
from coolprompt.optimizer.brave.bayesian_sampling import (
    BayesianLinearTS,
    StateFeaturizer,
)
from coolprompt.optimizer.brave.core_states import OptimizerState
from coolprompt.utils.enums import Task


class TestStateFeaturizer(unittest.TestCase):

    def test_transform_includes_interaction_features(self):
        state = OptimizerState(
            val_quality=0.8,
            quality_slope=0.1,
            stagnation=0.6,
            useless_ops_ratio=0.25,
            remaining_budget_ratio=0.5,
            epoch_progress=0.4,
            population_diversity=0.75,
        )

        features = StateFeaturizer().transform(state)

        np.testing.assert_allclose(
            features, [0.8, 0.1, 0.6, 0.25, 0.5, 0.4, 0.3, 0.75, 0.15]
        )
        self.assertEqual(features.shape, (StateFeaturizer().dim,))


class TestBayesianLinearTS(unittest.TestCase):

    def test_update_changes_posterior_and_prediction(self):
        model = BayesianLinearTS(dim=2, alpha=1.0, sigma2=1.0)
        x = np.array([1.0, 0.0])

        model.update(x, y=2.0)

        np.testing.assert_allclose(model.posterior_mean(), [1.0, 0.0])
        self.assertAlmostEqual(model.predictive_mean(x), 1.0)
        self.assertAlmostEqual(model.predictive_std(x), np.sqrt(0.5))


class TestStratifiedBatchSampler(unittest.TestCase):

    def test_classification_sampling_is_balanced_and_deterministic(self):
        dataset = [f"sample-{index}" for index in range(8)]
        targets = ["a"] * 4 + ["b"] * 4
        sampler = StratifiedBatchSampler(
            task=Task.CLASSIFICATION,
            batch_size=4,
            seed=11,
        )

        first = sampler.sample(dataset, targets, epoch=3)
        second = sampler.sample(dataset, targets, epoch=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)
        self.assertEqual([targets[index] for index in first].count("a"), 2)
        self.assertEqual([targets[index] for index in first].count("b"), 2)

    def test_empty_and_small_datasets_do_not_require_sampling(self):
        sampler = StratifiedBatchSampler(Task.GENERATION, batch_size=5)

        self.assertEqual(sampler.sample([], [], epoch=0), [])
        self.assertEqual(
            sampler.sample(["a", "b"], ["x", "y"], epoch=0),
            [0, 1],
        )


class TestCurriculumStratifiedBatchSampler(unittest.TestCase):

    def test_difficulty_and_alpha_are_updated(self):
        sampler = CurriculumStratifiedBatchSampler(
            task=Task.CLASSIFICATION,
            batch_size=2,
            total_steps=10,
            warmup_steps=2,
            max_alpha=0.8,
        )

        sampler.update_difficulties([0, 1], failed_indices=[1])

        self.assertEqual(sampler._curriculum_alpha(epoch=2), 0.0)
        self.assertAlmostEqual(sampler._curriculum_alpha(epoch=6), 0.4)
        self.assertAlmostEqual(sampler._curriculum_alpha(epoch=10), 0.8)
        self.assertEqual(sampler._difficulty(0), 0.0)
        self.assertEqual(sampler._difficulty(1), 1.0)
        self.assertEqual(sampler._difficulty(99), 0.5)


if __name__ == "__main__":
    unittest.main()
