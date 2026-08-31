from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import yaml

from coolprompt.optimizer.reflective_prompt.prompt import Prompt


@dataclass
class ElitistMutationLog:
    """Represent one logged mutation of the elite prompt."""

    iteration: int
    timestamp: str
    elitist_prompt: str
    prev_score: float
    mutated_prompt: str
    mutated_score: float
    new_long_term_reflection: str
    short_term_reflections: List[str]


@dataclass
class GradientStepLog:
    """Represent one logged textual-gradient step."""

    iteration: int
    timestamp: str
    prompt: str
    prev_score: float
    mutated_prompt: str
    mutated_score: float
    textual_gradient: str


@dataclass
class MutationLog:
    """Represent one generic prompt mutation log entry."""

    iteration: int
    timestamp: str
    prompt: str
    prev_score: float
    mutated_prompt: str
    mutated_score: float


@dataclass
class CreativeRoleStyleMutationLog:
    """Represent one creative role-and-style mutation."""

    iteration: int
    timestamp: str
    prompt: str
    prev_score: float
    mutated_prompt: str
    mutated_score: float
    style: str
    role: str


@dataclass
class FewShotExamplesMutationLog:
    """Represent one mutation of a prompt's few-shot examples."""

    iteration: int
    timestamp: str
    prompt: str
    prev_score: float
    mutated_prompt: str
    mutated_score: float
    added_few_shot: List[str]
    removed_few_shot: List[str]


@dataclass
class CrossoverLog:
    """Represent one crossover and its parent feedback."""

    iteration: int
    timestamp: str
    parent1_prompt: str
    parent1_score: float
    parent2_prompt: str
    parent2_score: float
    offspring_prompt: str
    offsprint_score: float
    parent1_textual_gradient: str
    parent2_textual_gradient: str
    short_term_reflection: str


@dataclass
class PopulationLog:
    """Represent a population snapshot at an optimization iteration."""

    iteration: int
    timestamp: str
    population: List[Dict[str, Any]]


@dataclass
class ControllerStateLog:
    """Represent a controller action-selection snapshot."""

    iteration: int
    timestamp: str
    selected_action: str
    action_scores: dict
    is_fallback: bool
    action_stats: dict
    global_step: int


class OperationLogger:
    """Persist BRAVE operation diagnostics as YAML files."""

    def __init__(self, log_dir: str = "operation_logs"):
        """Create a logger that writes beneath ``log_dir``.

        Args:
            log_dir (str): directory in which YAML logs are stored.
        """

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _append_logs(
        self,
        filename: str,
        key: str,
        log_entry: Any
    ) -> None:
        """Append a dataclass log entry beneath a YAML document key.

        Args:
            filename (str): YAML file to update.
            key (str): top-level list key in the YAML document.
            log_entry (Any): dataclass instance to serialize and append.
        """

        existing_logs = []
        if filename.exists():
            with open(filename, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                existing_logs = data.get(key, [])

        existing_logs.append(asdict(log_entry))
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(
                {key: existing_logs},
                f,
                allow_unicode=True,
                sort_keys=False
            )

    def log_elitist_mutation(
        self,
        iteration: int,
        elitist_prompt: str,
        prev_score: float,
        mutated_prompt: str,
        mutated_score: float,
        new_long_term_reflection: str,
        short_term_reflections: List[str]
    ) -> None:
        """Log an elite mutation to a YAML file.

        Args:
            iteration (int): optimization iteration.
            elitist_prompt (str): elite prompt before mutation.
            prev_score (float): elite score before mutation.
            mutated_prompt (str): generated prompt text.
            mutated_score (float): generated prompt score.
            new_long_term_reflection (str): updated long-term reflection.
            short_term_reflections (List[str]): reflections used by mutation.
        """
        log_entry = ElitistMutationLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            elitist_prompt=elitist_prompt,
            prev_score=prev_score,
            mutated_prompt=mutated_prompt,
            mutated_score=mutated_score,
            new_long_term_reflection=new_long_term_reflection,
            short_term_reflections=short_term_reflections
        )

        log_file = self.log_dir / "elitist_mutations.yaml"
        self._append_logs(log_file, "mutations", log_entry)

    def log_mutation(
        self,
        iteration: int,
        prompt: str,
        prev_score: float,
        mutated_prompt: str,
        mutated_score: float,
        file_name: str = "mutations"
    ) -> None:
        """Log a generic mutation to a YAML file.

        Args:
            iteration (int): optimization iteration.
            prompt (str): source prompt text.
            prev_score (float): source prompt score.
            mutated_prompt (str): generated prompt text.
            mutated_score (float): generated prompt score.
            file_name (str): output filename without extension.
        """
        log_entry = MutationLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            prev_score=prev_score,
            mutated_prompt=mutated_prompt,
            mutated_score=mutated_score,
        )

        log_file = self.log_dir / f"{file_name}.yaml"
        self._append_logs(log_file, "mutations", log_entry)

    def log_gradient_step(
        self,
        iteration: int,
        prompt: str,
        prev_score: float,
        mutated_prompt: str,
        mutated_score: float,
        textual_gradient: str,
    ) -> None:
        """Log a textual-gradient mutation.

        Args:
            iteration (int): optimization iteration.
            prompt (str): source prompt text.
            prev_score (float): source prompt score.
            mutated_prompt (str): generated prompt text.
            mutated_score (float): generated prompt score.
            textual_gradient (str): feedback applied by the mutation.
        """

        log_entry = GradientStepLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            prev_score=prev_score,
            mutated_prompt=mutated_prompt,
            mutated_score=mutated_score,
            textual_gradient=textual_gradient
        )

        log_file = self.log_dir / "gradient_steps.yaml"
        self._append_logs(log_file, "mutations", log_entry)

    def log_creative_role_style_mutation(
        self,
        iteration: int,
        prompt: str,
        prev_score: float,
        mutated_prompt: str,
        mutated_score: float,
        style: str,
        role: str
    ) -> None:
        """Log a creative role-and-style mutation.

        Args:
            iteration (int): optimization iteration.
            prompt (str): source prompt text.
            prev_score (float): source prompt score.
            mutated_prompt (str): generated prompt text.
            mutated_score (float): generated prompt score.
            style (str): generated writing style.
            role (str): generated model role.
        """

        log_entry = CreativeRoleStyleMutationLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            prev_score=prev_score,
            mutated_prompt=mutated_prompt,
            mutated_score=mutated_score,
            style=style,
            role=role
        )

        log_file = self.log_dir / "creative_role_style_mutations.yaml"
        self._append_logs(log_file, "mutations", log_entry)

    def log_few_shot_mutation(
        self,
        iteration: int,
        prompt: str,
        prev_score: float,
        mutated_prompt: str,
        mutated_score: float,
        added_few_shot: Tuple[str, str],
        removed_few_shot: Tuple[str, str],
        file_name: str = "few_shot_mutations",
    ) -> None:
        """Log an addition or replacement of a few-shot example.

        Args:
            iteration (int): optimization iteration.
            prompt (str): source prompt text.
            prev_score (float): source prompt score.
            mutated_prompt (str): generated prompt text.
            mutated_score (float): generated prompt score.
            added_few_shot (Tuple[str, str]): inserted input-output example.
            removed_few_shot (Tuple[str, str]): replaced example, if any.
            file_name (str): output filename without extension.
        """

        log_entry = FewShotExamplesMutationLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            prev_score=prev_score,
            mutated_prompt=mutated_prompt,
            mutated_score=mutated_score,
            added_few_shot=list(added_few_shot),
            removed_few_shot=list(removed_few_shot),
        )

        log_file = self.log_dir / f"{file_name}.yaml"
        self._append_logs(log_file, "mutations", log_entry)

    def log_crossover(
        self,
        iteration: int,
        parent1_prompt: str,
        parent1_score: float,
        parent2_prompt: str,
        parent2_score: float,
        parent1_textual_gradient: str,
        parent2_textual_gradient: str,
        offspring_prompt: str,
        offspring_score: float,
        short_term_reflection: str
    ) -> None:
        """Log a crossover operation to a YAML file.

        Args:
            iteration (int): optimization iteration.
            parent1_prompt (str): first parent text.
            parent1_score (float): first parent score.
            parent2_prompt (str): second parent text.
            parent2_score (float): second parent score.
            parent1_textual_gradient (str): first parent feedback.
            parent2_textual_gradient (str): second parent feedback.
            offspring_prompt (str): generated offspring text.
            offspring_score (float): generated offspring score.
            short_term_reflection (str): crossover reflection.
        """
        log_entry = CrossoverLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            parent1_prompt=parent1_prompt,
            parent1_score=parent1_score,
            parent2_prompt=parent2_prompt,
            parent2_score=parent2_score,
            parent1_textual_gradient=parent1_textual_gradient,
            parent2_textual_gradient=parent2_textual_gradient,
            offspring_prompt=offspring_prompt,
            offsprint_score=offspring_score,
            short_term_reflection=short_term_reflection
        )

        log_file = self.log_dir / "crossovers.yaml"
        self._append_logs(log_file, "crossovers", log_entry)

    def log_population(self, iteration: int, population: List[Prompt]) -> None:
        """Write a complete population snapshot for an iteration.

        Args:
            iteration (int): optimization iteration.
            population (List[Prompt]): evaluated prompts to serialize.
        """

        population = [p.to_dict() for p in population]
        log_entry = PopulationLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            population=population
        )

        log_file = self.log_dir / f"{iteration}_population.yaml"
        with open(log_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                asdict(log_entry),
                f,
                allow_unicode=True,
                sort_keys=False
            )

    def log_controller_state(
        self,
        iteration: int,
        selected_action: str,
        action_scores: dict,
        is_fallback: bool,
        action_stats: dict,
        global_step: int
    ) -> None:
        """Log controller state and action selection.

        Args:
            iteration (int): optimization iteration.
            selected_action (str): selected action name.
            action_scores (dict): controller scores by action.
            is_fallback (bool): whether fallback selection was used.
            action_stats (dict): current per-action statistics.
            global_step (int): controller step counter.
        """
        log_entry = ControllerStateLog(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            selected_action=selected_action,
            action_scores=action_scores,
            is_fallback=is_fallback,
            action_stats=action_stats,
            global_step=global_step
        )

        log_file = self.log_dir / "controller_state.yaml"
        existing_logs = []
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                existing_logs = data.get('controller_states', [])

        existing_logs.append(asdict(log_entry))
        with open(log_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                {'controller_states': existing_logs},
                f,
                allow_unicode=True,
                sort_keys=False
            )

    def log_diversity_filter(
        self,
        step: int,
        filter_report: dict
    ) -> None:
        """Log a population-diversity filtering report.

        Args:
            step (int): optimization step.
            filter_report (dict): thresholds, clusters, and removed indices.
        """
        log_file = self.log_dir / "diversity_filter.yaml"
        existing_logs = []
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                existing_logs = data.get('diversity_filters', [])

        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'duplicate_threshold': filter_report.get('threshold'),
            'num_clusters': filter_report.get('num_clusters'),
            'num_removed': filter_report.get('num_removed'),
            'removed_indices': filter_report.get('removed_indices'),
            'deduplication_removed': filter_report.get('deduplication_removed')
        }

        existing_logs.append(log_entry)
        with open(log_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                {'diversity_filters': existing_logs},
                f,
                allow_unicode=True,
                sort_keys=False
            )
