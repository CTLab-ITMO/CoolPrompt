import math
from typing import Dict, List

import numpy as np

from coolprompt.optimizer.brave.core_states import OptimizerState


class StateFeaturizer:
    """Maps OptimizerState to a dense feature vector."""

    def transform(self, s: OptimizerState) -> np.ndarray:
        x = np.array(
            [
                s.val_quality,
                s.quality_slope,
                s.stagnation,
                s.useless_ops_ratio,
                s.remaining_budget_ratio,
                s.epoch_progress,
                # progress under budget pressure
                s.stagnation * s.remaining_budget_ratio,
                s.population_diversity,
                # stagnation is most dangerous when population has converged
                s.stagnation * (1.0 - s.population_diversity),
            ],
            dtype=np.float64,
        )
        return x

    @property
    def dim(self) -> int:
        return 9


class BayesianLinearTS:
    """Bayesian linear regression with Thompson sampling."""

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        sigma2: float = 1.0
    ) -> None:
        self.dim = dim
        self.alpha = alpha
        self.sigma2 = sigma2
        self.A = alpha * np.eye(dim)
        self.b = np.zeros(dim)

    def update(self, x: np.ndarray, y: float) -> None:
        self.A += np.outer(x, x) / self.sigma2
        self.b += (x * y) / self.sigma2

    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        # Numerical guard
        A_inv = np.linalg.pinv(self.A)
        mu = A_inv @ self.b
        cov = self.sigma2 * A_inv
        return rng.multivariate_normal(mu, cov)

    def posterior_mean(self) -> np.ndarray:
        A_inv = np.linalg.pinv(self.A)
        return A_inv @ self.b

    def predictive_mean(self, x: np.ndarray) -> float:
        return float(np.dot(self.posterior_mean(), x))

    def predictive_std(self, x: np.ndarray) -> float:
        A_inv = np.linalg.pinv(self.A)
        var = float(np.dot(x, A_inv @ x)) * self.sigma2
        return float(math.sqrt(max(var, 1e-12)))


class OnlineActionMLP:
    """Tiny shared-trunk neural contextual bandit (numpy-only).

    Heads predict:
    - benefit
    - cost
    - improvement logit (for P(improvement > 0))
    """

    def __init__(
        self,
        actions: List[str],
        input_dim: int,
        hidden_dim: int = 32,
        learning_rate: float = 5e-3,
        seed: int = 123,
    ) -> None:
        self.actions = actions
        self.a2i = {a: i for i, a in enumerate(actions)}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)

        # Shared trunk
        self.W1 = self.rng.normal(0.0, 0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # Per-action heads
        n = len(actions)
        self.W_benefit = self.rng.normal(0.0, 0.1, size=(n, hidden_dim))
        self.b_benefit = np.zeros(n)
        self.W_cost = self.rng.normal(0.0, 0.1, size=(n, hidden_dim))
        self.b_cost = np.zeros(n)
        self.W_impr = self.rng.normal(0.0, 0.1, size=(n, hidden_dim))
        self.b_impr = np.zeros(n)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0)

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -20.0, 20.0))
        return 1.0 / (1.0 + math.exp(-z))

    def _forward_hidden(self, x: np.ndarray) -> np.ndarray:
        return self._relu(x @ self.W1 + self.b1)

    def predict(self, action: str, x: np.ndarray) -> Dict[str, float]:
        idx = self.a2i[action]
        h = self._forward_hidden(x)
        benefit = float(np.dot(self.W_benefit[idx], h) + self.b_benefit[idx])
        # ensure positive-ish cost
        raw_cost = float(np.dot(self.W_cost[idx], h) + self.b_cost[idx])
        cost = float(np.log1p(math.exp(np.clip(raw_cost, -20.0, 20.0))) + 1e-6)
        impr_logit = float(np.dot(self.W_impr[idx], h) + self.b_impr[idx])
        impr_prob = self._sigmoid(impr_logit)
        return {"benefit": benefit, "cost": cost, "impr_prob": impr_prob}

    def update(
        self,
        action: str,
        x: np.ndarray,
        target_benefit: float,
        target_cost: float,
        target_impr: float,
    ) -> None:
        idx = self.a2i[action]
        h_pre = x @ self.W1 + self.b1
        h = self._relu(h_pre)

        # forward
        pred_b = float(np.dot(self.W_benefit[idx], h) + self.b_benefit[idx])
        raw_cost = float(np.dot(self.W_cost[idx], h) + self.b_cost[idx])
        pred_c = float(
            np.log1p(math.exp(np.clip(raw_cost, -20.0, 20.0))) + 1e-6
        )
        pred_l = float(np.dot(self.W_impr[idx], h) + self.b_impr[idx])
        pred_p = self._sigmoid(pred_l)

        # losses:
        # benefit, cost -> mse
        # improvement -> logistic BCE
        db = (pred_b - float(target_benefit))
        dc = (pred_c - float(target_cost))
        dp = (pred_p - float(target_impr))

        # gradients wrt head outputs
        # cost head uses softplus(raw_cost),
        # d pred_c / d raw_cost = sigmoid(raw_cost)
        dsoftplus = self._sigmoid(raw_cost)
        draw_cost = dc * dsoftplus

        # head grads
        gWb = db * h
        gbb = db
        gWc = draw_cost * h
        gbc = draw_cost
        gWi = dp * h
        gbi = dp

        # backprop to hidden
        gh = (
            db * self.W_benefit[idx]
            + draw_cost * self.W_cost[idx]
            + dp * self.W_impr[idx]
        )
        gh = gh * (h_pre > 0.0).astype(float)

        # trunk grads
        gW1 = np.outer(x, gh)
        gb1 = gh

        # SGD updates (only selected action heads)
        self.W_benefit[idx] -= self.lr * gWb
        self.b_benefit[idx] -= self.lr * gbb
        self.W_cost[idx] -= self.lr * gWc
        self.b_cost[idx] -= self.lr * gbc
        self.W_impr[idx] -= self.lr * gWi
        self.b_impr[idx] -= self.lr * gbi
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1
