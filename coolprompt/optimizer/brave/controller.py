import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coolprompt.optimizer.brave.bayesian_sampling import (
    BayesianLinearTS, OnlineActionMLP
)


class EVCController:
    """Action selector maximizing expected epistemic value per token.

    Score(action) = Benefit / Cost
    Benefit ~ ΔQuality + λ_c * ΔCoverage - λ_d * ΔDrift
    """

    def __init__(
        self,
        actions: List[str],
        feature_dim: int,
        min_cost_eps: float = 1e-6,
        max_action_budget_share: float = 0.35,
        uncertainty_penalty_beta: float = 0.35,
        neural_weight: float = 0.5,
        alpha_roi_ema: float = 0.1,
        improve_prob_weight: float = 0.6,
        kill_switch_min_trials: int = 10,
        kill_switch_roi_threshold: float = -0.0002,
        kill_switch_base_cooldown: int = 5,
        kill_switch_scaling_factor: float = 20.0,
        use_neural_bandit: bool = True,
        neural_hidden_dim: int = 32,
        neural_learning_rate: float = 5e-3,
        seed: int = 42,
    ) -> None:
        self.actions = actions
        self.min_cost_eps = min_cost_eps
        self.max_action_budget_share = max_action_budget_share
        self.uncertainty_penalty_beta = uncertainty_penalty_beta
        self.neural_weight = min(max(neural_weight, 0.0), 1.0)
        self.alpha_roi_ema = alpha_roi_ema
        self.improve_prob_weight = improve_prob_weight
        self.kill_switch_min_trials = kill_switch_min_trials
        self.kill_switch_roi_threshold = kill_switch_roi_threshold
        self.kill_switch_base_cooldown = kill_switch_base_cooldown
        self.kill_switch_scaling_factor = kill_switch_scaling_factor
        self.use_neural_bandit = use_neural_bandit
        self.rng = np.random.default_rng(seed)

        self.benefit_models: Dict[str, BayesianLinearTS] = {
            a: BayesianLinearTS(feature_dim, alpha=1.0, sigma2=1.0)
            for a in actions
        }
        self.cost_models: Dict[str, BayesianLinearTS] = {
            a: BayesianLinearTS(feature_dim, alpha=1.0, sigma2=1.0)
            for a in actions
        }
        self.improvement_models: Dict[str, BayesianLinearTS] = {
            a: BayesianLinearTS(feature_dim, alpha=1.0, sigma2=1.0)
            for a in actions
        }
        self.neural_bandit = (
            OnlineActionMLP(
                actions=actions,
                input_dim=feature_dim,
                hidden_dim=neural_hidden_dim,
                learning_rate=neural_learning_rate,
                seed=seed + 17,
            )
            if use_neural_bandit
            else None
        )
        self.action_stats: Dict[str, Dict[str, float]] = {
            a: {
                "trials": 0.0,
                "success_count": 0.0,
                "ema_roi": 0.0,
                "last_selected_step": -1.0,
                "disabled_until_step": -1.0,
            }
            for a in actions
        }
        self.global_step: int = 0

    def _sample_benefit_cost(
        self,
        action: str,
        x: np.ndarray
    ) -> Tuple[float, float]:
        theta_b = self.benefit_models[action].sample_theta(self.rng)
        theta_c = self.cost_models[action].sample_theta(self.rng)

        benefit = float(np.dot(theta_b, x))
        cost = float(np.dot(theta_c, x))
        cost = max(cost, self.min_cost_eps)
        return benefit, cost

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -20.0, 20.0))
        return 1.0 / (1.0 + math.exp(-z))

    def _calculate_usefulness(self, action: str) -> float:
        """Calculate usefulness metric
        (0-1, where 1 = highly useful, 0 = not useful)
        """
        st = self.action_stats[action]
        trials = st["trials"]

        if trials < 1:
            return 1.0  # No data yet, assume useful

        success_rate = st["success_count"] / trials
        return float(success_rate)

    def _calculate_adaptive_cooldown(self, action: str) -> int:
        """Calculate adaptive cooldown duration based on operator usefulness"""
        usefulness = self._calculate_usefulness(action)
        lack_of_usefulness = 1.0 - usefulness
        if lack_of_usefulness < 0.5:
            lack_of_usefulness = self._sigmoid(
                15 * (lack_of_usefulness - 0.5)
            )
        else:
            lack_of_usefulness = self._sigmoid(
                4 * (lack_of_usefulness - 0.5)
            )

        cooldown = int(
            self.kill_switch_base_cooldown +
            self.kill_switch_scaling_factor * lack_of_usefulness
        )

        return cooldown

    def _should_disable(self, action: str) -> bool:
        st = self.action_stats[action]
        if self.global_step < int(st["disabled_until_step"]):
            return True

        if st["trials"] < self.kill_switch_min_trials:
            return False
        # Check if just exiting disability period
        # (give second chance by resetting ema_roi)
        if st["disabled_until_step"] > 0 and \
           self.global_step == int(st["disabled_until_step"]):
            st["ema_roi"] = 0.0  # Reset to give fresh evaluation
            st["disabled_until_step"] = -1.0
            return False

        if st["ema_roi"] < self.kill_switch_roi_threshold:
            st["disabled_until_step"] = float(
                self.global_step + self._calculate_adaptive_cooldown(action)
            )
            return True
        return False

    def select_action(
        self,
        x: np.ndarray,
        remaining_budget_tokens: float,
        candidate_actions: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, float]]:
        self.global_step += 1
        best_action: Optional[str] = None
        best_score = -math.inf
        scores: Dict[str, float] = {}
        if candidate_actions is not None:
            action_pool = candidate_actions
        else:
            action_pool = self.actions

        for action in action_pool:
            if self._should_disable(action):
                continue

            benefit_hat, cost_hat = self._sample_benefit_cost(action, x)
            impr_hat_linear = self._sigmoid(
                self.improvement_models[action].predictive_mean(x)
            )
            b_unc = self.benefit_models[action].predictive_std(x)
            c_unc = self.cost_models[action].predictive_std(x)
            uncertainty = 0.5 * (b_unc + c_unc)

            if self.neural_bandit is not None:
                nn = self.neural_bandit.predict(action, x)
                benefit_hat = (1.0 - self.neural_weight) * benefit_hat +\
                    self.neural_weight * nn["benefit"]
                cost_hat = (1.0 - self.neural_weight) * cost_hat +\
                    self.neural_weight * nn["cost"]
                impr_prob = (1.0 - self.neural_weight) * impr_hat_linear +\
                    self.neural_weight * nn["impr_prob"]
            else:
                impr_prob = impr_hat_linear

            if cost_hat > remaining_budget_tokens:
                continue
            if cost_hat > self.max_action_budget_share *\
                    max(remaining_budget_tokens, 1.0):
                continue

            effective_cost = cost_hat *\
                (1.0 + self.uncertainty_penalty_beta * uncertainty)
            score = (benefit_hat * (1.0 + self.improve_prob_weight * impr_prob))
            score = score / max(effective_cost, self.min_cost_eps)
            scores[action] = score
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            return None, None

        self.action_stats[best_action]["last_selected_step"] =\
            float(self.global_step)
        return best_action, scores

    def update(
        self,
        action: str,
        x_before: np.ndarray,
        delta_quality: float,
        actual_cost_tokens: float,
        improved: bool = False,
    ) -> None:
        benefit = float(delta_quality)
        cost = max(float(actual_cost_tokens), self.min_cost_eps)
        improvement = 1.0 if float(delta_quality) > 0.0 else 0.0

        self.benefit_models[action].update(x_before, benefit)
        self.cost_models[action].update(x_before, cost)
        self.improvement_models[action].update(x_before, improvement)
        if self.neural_bandit is not None:
            self.neural_bandit.update(
                action=action,
                x=x_before,
                target_benefit=benefit,
                target_cost=cost,
                target_impr=improvement,
            )

        st = self.action_stats[action]
        st["trials"] += 1.0
        if improved:
            st["success_count"] += 1.0
        realized_roi = benefit / max(cost, self.min_cost_eps)
        # EMA for kill-switch
        st["ema_roi"] = (1.0 - self.alpha_roi_ema) * st["ema_roi"] +\
            self.alpha_roi_ema * realized_roi

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "action_stats": self.action_stats,
            "uncertainty_penalty_beta": self.uncertainty_penalty_beta,
            "neural_weight": self.neural_weight,
            "use_neural_bandit": self.use_neural_bandit,
        }
