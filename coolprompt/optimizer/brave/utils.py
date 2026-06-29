from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

from coolprompt.optimizer.reflective_prompt.prompt import Prompt

FEEDBACK_TAGS = ("<feedback>", "</feedback>")
HINT_TAGS = ("<hint>", "</hint>")
PROMPT_TAGS = ("<prompt>", "</prompt>")
STYLE_TAGS = ("<style>", "</style>")
ROLE_TAGS = ("<role>", "</role>")


@dataclass
class BEGRAPEConfig:
    actions: List[str] | str = field(
        default_factory=lambda: [
            "crossover",
            "mutation",
        ]
    )
    initial_budget_tokens: float = 200_000.0
    max_steps: int = 1000
    population_size: int = 10
    bad_examples_num: int = 5
    patience_steps: int = 100
    min_improvement: float = 1e-4
    lambda_mean_quality: float = 0.3
    lambda_min_quality: float = 0.3
    max_action_budget_share: float = 0.35
    alpha_roi_ema: float = 0.1
    uncertainty_penalty_beta: float = 0.35
    neural_weight: float = 0.55
    improve_prob_weight: float = 0.6
    kill_switch_min_trials: int = 10
    kill_switch_roi_threshold: float = -0.0002
    kill_switch_base_cooldown: int = 5
    kill_switch_scaling_factor: float = 10.0
    use_neural_bandit: bool = True
    neural_hidden_dim: int = 32
    neural_learning_rate: float = 5e-3
    diversity_similarity_threshold: float = 0.80
    diversity_max_per_cluster: int = 2
    diversity_auto_threshold: bool = True
    diversity_use_hierarchical: bool = True
    diversity_use_bert: bool = True
    diversity_bert_weight: float = 0.6
    random_mutation_probability: float = 0.0
    diversity_duplicate_threshold: float = 0.95
    few_shot_examples_max_num: int = 5
    few_shot_examples_from_data_cnt: int = 7
    population_initializer: str = "begrape"
    population_clusterization: bool = True
    initial_population_size: int = 10
    train_batch_size: int = 0
    train_batch_seed: int = 19
    use_stratified_train_batches: bool = True
    generation_strata_bins: int = 3
    use_curriculum_batches: bool = False
    curriculum_warmup_steps: int = 20
    curriculum_max_alpha: float = 0.6
    val_checkpoint_steps: int = 0
    val_checkpoint_topk: int = 1
    rescore_steps: int = 0
    early_stop: bool = False


def _merge_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    out = dict(base)
    out.update(override)
    return out


def load_begrape_config_from_yaml(
    path: str, profile: str = "balanced"
) -> BEGRAPEConfig:
    """Load BEGRAPEConfig from YAML with merge rule:

    final = defaults overridden by profiles[profile]
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError(
            "PyYAML is required to load YAML configs. " +
            "Install with: pip install pyyaml"
        ) from exc

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.get("defaults", {})
    profiles = raw.get("profiles", {})
    selected = profiles.get(profile)
    if selected is None:
        available = ", ".join(sorted(profiles.keys()))
        raise KeyError(
            f"Profile '{profile}' not found. Available: [{available}]"
        )

    merged = _merge_dicts(defaults, selected)
    allowed = {x.name for x in fields(BEGRAPEConfig)}
    kwargs = {k: v for k, v in merged.items() if k in allowed}
    return BEGRAPEConfig(**kwargs)


@dataclass
class OptimizationLog:
    step: int
    action: str
    score: float
    delta_quality: float
    cost_tokens: float
    cumulative_spent: float
    value_per_token: float
    useful_operation: bool
    controller_diag: Dict[str, float]
    remaining_budget: float
    best_quality: float


def reranking_population(population: List[Prompt]) -> List[Prompt]:
    """
    Sorts given population of prompts by their scores in descending order.

    Args:
        population (List[Prompt]): population to sort.

    Returns:
        List[Prompt]: sorted population.
    """
    return list(
        sorted(population, key=lambda prompt: prompt.score, reverse=True)
    )
