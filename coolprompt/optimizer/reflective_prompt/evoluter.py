import os
import time
import yaml
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import statistics
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.messages.ai import AIMessage
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.reflective_prompt.prompt import Prompt, PromptOrigin
from coolprompt.utils.logging_config import logger

from coolprompt.utils.prompt_templates.reflective_templates_fixed_role import (
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_FIXED_ROLE,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_FIXED_ROLE,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE_FIXED_ROLE,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_FIXED_ROLE,
    REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_FIXED_ROLE,
    REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_FIXED_ROLE,
)
from coolprompt.utils.prompt_templates.reflective_templates_no_role import (
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_NO_ROLE,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_NO_ROLE,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE_NO_ROLE,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_NO_ROLE,
    REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_NO_ROLE,
    REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_NO_ROLE,
)
from coolprompt.utils.prompt_templates.reflective_templates_coevolution import (
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_COEVO,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO,
    REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_COEVO,
    REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO_BASE,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO_BASE,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO_BASE,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO_3F,
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_COEVO_3F,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO_3F,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO_3F,
    REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_COEVO_3F,
    REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_3F,
)
from coolprompt.utils.prompt_templates.reflective_templates_text_only import (
    REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_TEXT_ONLY,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_TEXT_ONLY,
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_TEXT_ONLY,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_TEXT_ONLY,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE_TEXT_ONLY,
    REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_TEXT_ONLY,
)
from coolprompt.utils.prompt_templates.reflective_templates_factorized import (
    REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_ROLE_ONLY,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_ROLE_ONLY,
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_ROLE_ONLY,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_ROLE_ONLY,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE_ROLE_ONLY,
    REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_CONSTRAINTS_ONLY,
    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_CONSTRAINTS_ONLY,
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_CONSTRAINTS_ONLY,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_CONSTRAINTS_ONLY,
    REFLECTIVEPROMPT_MUTATION_TEMPLATE_CONSTRAINTS_ONLY,
)
from coolprompt.utils.parsing import extract_answer, extract_json

_embedding_model = None
_use_embeddings = True


def _get_embedding_model():
    global _embedding_model, _use_embeddings
    if not _use_embeddings:
        return None
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, "
                "falling back to TF-IDF for similarity"
            )
            _use_embeddings = False
            return None
    return _embedding_model


class ReflectiveEvoluter:
    """
    ReflectiveEvoluter class that represents evoluter for ReflectivePrompt

    Attributes:
        model: langchain.BaseLanguageModel class of model to use.
        evaluator: evaluator (Evaluator) to compute metrics.
        train_dataset: a dataset to use while training.
        train_targets: string targets for train dataset.
        validation_dataset: a dataset to use while validating final prompts.
        validation_targets: string targets for validation dataset.
        problem_description: a string that contains
            short description of problem to optimize.
        initial_prompt: initial prompt to start evolution from.
            Will be automatically generated if not provided.
            Defaults to None.
        population_size: an integer fixed size of prompt population.
            Defaults to 10.
        num_epochs: an integer number of epochs to evaluate.
            Defaults to 10.
        use_cache: a boolean variable.
            Either to use caching files or not.
        output_path: a path to store logs of evolution.
        elitist: a prompt with highest score in population.
        best_score_overall: best evaluation score during evolution.
        best_prompt_overall: text of prompt with best score overall.
        iteration: current iteration (epoch) of evolution.
        PROMPT_TAGS: start and end tags for prompt extraction.
        HINT_TAGS: start and end tags for hint extraction.
    """

    PROMPT_TAGS = ("<prompt>", "</prompt>")
    HINT_TAGS = ("<hint>", "</hint>")
    ROLE_LENGTH_ALPHA: float = 0.02
    ROLE_PROMPT_SIM_THRESHOLD: float = 0.72
    ROLE_PROMPT_SIM_ALPHA: float = 0.05
    ELITIST_MAX_FREEZE: int = 3

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        train_dataset: List[str],
        train_targets: List[str],
        validation_dataset: List[str],
        validation_targets: List[str],
        problem_description: str,
        initial_prompt: Optional[str] = None,
        initial_role: Optional[str] = None,
        initial_constraints: Optional[str] = None,
        evolve_role: bool = True,
        evolve_constraints: bool = False,
        population_size: int = 10,
        num_epochs: int = 10,
        output_path: str = "./reflectiveprompt_outputs",
        use_cache: bool = True,
        use_enhancements: bool = True,
        freeze_text: bool = False,
        text_only: bool = False,
        val_evaluator: Optional[Evaluator] = None,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.val_evaluator = val_evaluator or evaluator
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.validation_dataset = validation_dataset
        self.validation_targets = validation_targets
        self.use_cache = use_cache
        self.population_size = population_size
        self.num_epochs = num_epochs
        self.problem_description = problem_description
        self.output_path = output_path
        self.initial_prompt = initial_prompt
        self.initial_role = initial_role
        self.initial_constraints = initial_constraints or ""
        self.evolve_role = evolve_role
        self.evolve_constraints = evolve_constraints
        self.use_enhancements = use_enhancements
        self.freeze_text = freeze_text
        self.text_only = text_only
        self._role_only = (
            self.evolve_role
            and self.freeze_text
            and not self.evolve_constraints
        )
        self._constraints_only = (
            not self.evolve_role
            and bool(self.initial_role)
            and self.evolve_constraints
        )

        self.elitist = None
        self._long_term_reflection_str = ""
        self.best_score_overall = None
        self.best_prompt_overall = None
        self.best_role_overall = None
        self.best_constraints_overall = None
        self.iteration = 0
        self._elitist_freeze_count: int = 0
        self._prev_elitist_role: str = ""
        self._hall_of_fame: List[Prompt] = []
        self._elitist_bad_examples: List[Dict] = []

        self._setup_templates()

    def _setup_templates(self) -> None:
        """Selects prompt templates based on the active evolution mode."""
        if self.text_only:
            self._paraphrasing_template = (
                REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_TEXT_ONLY
            )
            self._crossover_template = (
                REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_TEXT_ONLY
            )
            self._mutation_template = (
                REFLECTIVEPROMPT_MUTATION_TEMPLATE_TEXT_ONLY
            )
            self._short_term_template = (
                REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_TEXT_ONLY
            )
            self._long_term_template = (
                REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_TEXT_ONLY
            )
            self._initial_prompt_template = (
                REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_TEXT_ONLY
            )
        elif self._role_only:
            self._paraphrasing_template = (
                REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_ROLE_ONLY
            )
            self._crossover_template = (
                REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_ROLE_ONLY
            )
            self._mutation_template = (
                REFLECTIVEPROMPT_MUTATION_TEMPLATE_ROLE_ONLY
            )
            self._short_term_template = (
                REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_ROLE_ONLY
            )
            self._long_term_template = (
                REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_ROLE_ONLY
            )
            self._initial_prompt_template = (
                REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO
            )
        elif self._constraints_only:
            self._paraphrasing_template = (
                REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_CONSTRAINTS_ONLY
            )
            self._crossover_template = (
                REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_CONSTRAINTS_ONLY
            )
            self._mutation_template = (
                REFLECTIVEPROMPT_MUTATION_TEMPLATE_CONSTRAINTS_ONLY
            )
            self._short_term_template = (
                REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_CONSTRAINTS_ONLY
            )
            self._long_term_template = (
                REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_CONSTRAINTS_ONLY
            )
            self._initial_prompt_template = (
                REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO
            )
        elif self.evolve_role and self.evolve_constraints:
            self._paraphrasing_template = (
                REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_COEVO_3F
            )
            self._crossover_template = (
                REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO_3F
            )
            self._mutation_template = (
                REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO_3F
            )
            self._short_term_template = (
                REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO_3F
            )
            self._long_term_template = (
                REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_COEVO_3F
            )
            self._initial_prompt_template = (
                REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_3F
            )
        elif self.evolve_role:
            self._paraphrasing_template = (
                REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_COEVO
            )
            if self.use_enhancements:
                self._crossover_template = (
                    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO
                )
                self._mutation_template = (
                    REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO
                )
                self._short_term_template = (
                    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO
                )
            else:
                self._crossover_template = (
                    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_COEVO_BASE
                )
                self._mutation_template = (
                    REFLECTIVEPROMPT_MUTATION_TEMPLATE_COEVO_BASE
                )
                self._short_term_template = (
                    REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_COEVO_BASE
                )
            self._long_term_template = (
                REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_COEVO
            )
            self._initial_prompt_template = (
                REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO
            )
        elif not self.initial_role:
            self._paraphrasing_template = (
                REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_NO_ROLE
            )
            self._crossover_template = (
                REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_NO_ROLE
            )
            self._mutation_template = REFLECTIVEPROMPT_MUTATION_TEMPLATE_NO_ROLE
            self._short_term_template = (
                REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_NO_ROLE
            )
            self._long_term_template = (
                REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_NO_ROLE
            )
            self._initial_prompt_template = (
                REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_NO_ROLE
            )
        else:
            self._paraphrasing_template = (
                REFLECTIVEPROMPT_PARAPHRASING_TEMPLATE_FIXED_ROLE
            )
            self._crossover_template = (
                REFLECTIVEPROMPT_CROSSOVER_TEMPLATE_FIXED_ROLE
            )
            self._mutation_template = (
                REFLECTIVEPROMPT_MUTATION_TEMPLATE_FIXED_ROLE
            )
            self._short_term_template = (
                REFLECTIVEPROMPT_SHORT_TERM_REFLECTION_TEMPLATE_FIXED_ROLE
            )
            self._long_term_template = (
                REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE_FIXED_ROLE
            )
            self._initial_prompt_template = (
                REFLECTIVEPROMPT_PROMPT_BY_DESCRIPTION_TEMPLATE_FIXED_ROLE
            )

    def _reranking(self, population: List[Prompt]) -> List[Prompt]:
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

    @staticmethod
    def _role_prompt_sim(role: str, prompt_text: str) -> float:
        """Cosine similarity between system_behavior and task_description.
        Uses sentence-transformer embeddings when available,
        falls back to TF-IDF otherwise.
        High similarity means the two components are redundant.
        Returns 0.0 if either string is empty.
        """
        if not role or not prompt_text:
            return 0.0
        model = _get_embedding_model()
        if model is not None:
            try:
                embs = model.encode([role, prompt_text])
                return float(
                    cosine_similarity(
                        embs[0].reshape(1, -1), embs[1].reshape(1, -1)
                    )[0][0]
                )
            except Exception:
                pass
        try:
            vec = TfidfVectorizer().fit_transform([role, prompt_text])
            return float(cosine_similarity(vec[0], vec[1])[0][0])
        except Exception:
            return 0.0

    def _update_hall_of_fame(self, population: List[Prompt]) -> None:
        seen = {(p.text, p.role, p.constraints) for p in self._hall_of_fame}
        for p in population:
            if p.score is None:
                continue
            key = (p.text, p.role, p.constraints)
            if key not in seen:
                self._hall_of_fame.append(
                    Prompt(
                        text=p.text,
                        role=p.role,
                        constraints=p.constraints,
                        origin=p.origin,
                        score=p.score,
                    )
                )
                seen.add(key)
        self._hall_of_fame.sort(key=lambda x: x.score, reverse=True)
        max_size = max(self.population_size * 2, 10)
        self._hall_of_fame = self._hall_of_fame[:max_size]

    def _format_bad_examples(self) -> str:
        if not self._elitist_bad_examples:
            return "(none)"
        lines = []
        for i, ex in enumerate(self._elitist_bad_examples, 1):
            inp = ex.input[:120]
            out = ex.output
            correct = ex.correct
            lines.append(
                f"{i}. Input: {inp}\n   Got: {out}  |  Expected: {correct}"
            )
        return "\n".join(lines)

    def _format_top_prompts_history(self, top_k: int = 5) -> str:
        if not self._hall_of_fame:
            return "(none)"
        entries = self._hall_of_fame[:top_k]
        lines = []
        for i, p in enumerate(entries, 1):
            score_str = self._format_score(p.score)
            if self._role_only:
                content = p.role or "(empty)"
                lines.append(f"{i}. [score={score_str}] {content[:120]}")
            elif self._constraints_only:
                content = p.constraints or "(empty)"
                lines.append(f"{i}. [score={score_str}] {content[:120]}")
            elif self.evolve_role and self.evolve_constraints:
                role = (p.role or "(empty)")[:80]
                text = (p.text or "(empty)")[:80]
                constraints = (p.constraints or "(empty)")[:80]
                lines.append(
                    f"{i}. [score={score_str}]\n   system_behavior: {role}\n   task_description: {text}\n   output_constraints: {constraints}"
                )
            elif self.evolve_role:
                role = (p.role or "(empty)")[:80]
                text = (p.text or "(empty)")[:80]
                lines.append(
                    f"{i}. [score={score_str}]\n   system_behavior: {role}\n   task_description: {text}"
                )
            else:
                content = p.text or "(empty)"
                lines.append(f"{i}. [score={score_str}] {content[:120]}")
        return "\n".join(lines)

    def _aggregate_bad_examples(
        self, population: List[Prompt], top_k: int = 3
    ) -> None:
        scored = [
            p for p in population if p.score is not None and p.bad_examples
        ]
        if not scored:
            return
        scored.sort(key=lambda p: p.score, reverse=True)
        top_half = scored[: max(1, len(scored) // 2)]
        counts: Dict[str, Dict] = {}
        for p in top_half:
            for ex in p.bad_examples:
                key = ex.input
                if key not in counts:
                    counts[key] = {"count": 0, "ex": ex}
                counts[key]["count"] += 1
        sorted_examples = sorted(
            counts.values(), key=lambda x: x["count"], reverse=True
        )
        self._elitist_bad_examples = [x["ex"] for x in sorted_examples[:top_k]]

    def _format_score(self, score) -> str:
        if not self.use_enhancements or score is None:
            return "N/A"
        return f"{score:.4f}"

    def _evaluate(self, prompt: Prompt, split="train") -> None:
        """Evaluates given prompt on self.dataset and records the score.
        When evolve_role=True and split=='train':
          - A length penalty proportional to role length is subtracted,
            discouraging bloated roles when scores are close.

        Args:
            prompt (Prompt): a prompt to evaluate.
            split (str, optional): Which split of dataset to use.
                Defaults to 'train'.
        """
        if split == "train":
            dataset, targets = self.train_dataset, self.train_targets
        else:
            dataset, targets = self.validation_dataset, self.validation_targets

        eval_role = prompt.role
        ev = self.evaluator if split == "train" else self.val_evaluator
        result = ev.evaluate(
            prompt=prompt.text,
            dataset=dataset,
            targets=targets,
            system_role=eval_role if eval_role else None,
            constraints=prompt.constraints if self.evolve_constraints else None,
            failed_examples=10 if split == "train" else None,
        )
        if isinstance(result, tuple):
            score, bad_examples = result
            prompt.set_bad_examples(bad_examples)
        else:
            score = result

        if self.evolve_role and split == "train":
            if self.use_enhancements:
                score = score - self.ROLE_LENGTH_ALPHA * len(prompt.role) / 1000

            if prompt.role:
                sim = self._role_prompt_sim(prompt.role, prompt.text)
                if sim > self.ROLE_PROMPT_SIM_THRESHOLD:
                    score -= self.ROLE_PROMPT_SIM_ALPHA * (
                        sim - self.ROLE_PROMPT_SIM_THRESHOLD
                    )

        prompt.set_score(score)

    def _evaluation(
        self, population: List[Prompt], split: str = "train"
    ) -> None:
        """Evaluation operation for prompts population.
        Evaluates every prompt in population and records the results.

        Args:
            population (List[Prompt]): population of prompts to evaluate.
            split (str, optional): Which split of dataset to use.
                Defaults to 'train'.
        """
        logger.info("Evaluating population...")
        for prompt in population:
            self._evaluate(prompt, split=split)
        if split == "train":
            self._aggregate_bad_examples(population)

    def _create_initial_prompt(self) -> Tuple[str, str, str]:
        """Creates an initial prompt according to provided problem description

        Returns:
            Tuple[str, str, str]: initial prompt
        """
        request = self._initial_prompt_template.format(
            PROBLEM_DESCRIPTION=self.problem_description
        )
        answer = self._llm_query([request])[0]
        extracted = extract_json(answer)
        if extracted is None:
            extracted = {}

        if self.evolve_role:
            role = extracted.get("system_behavior", extracted.get("role", ""))
        else:
            role = self.initial_role or ""

        prompt = extracted.get(
            "task_description",
            extracted.get(
                "prompt",
                extract_answer(
                    answer, self.PROMPT_TAGS, format_mismatch_label=""
                ),
            ),
        )
        constraints = (
            extracted.get("output_constraints", "")
            if self.evolve_constraints
            else ""
        )
        return role, prompt, constraints

    def _init_pop(self) -> List[Prompt]:
        """Creates initial population of prompts.

        Returns:
            List[Prompt]: initial population.
        """

        logger.info("Initializing population...")
        if self.initial_prompt is None:
            generated_role, self.initial_prompt, generated_constraints = (
                self._create_initial_prompt()
            )
            if self.evolve_role and not self.initial_role:
                self.initial_role = generated_role
            if self.evolve_constraints and not self.initial_constraints:
                self.initial_constraints = generated_constraints

        if self.initial_role is None:
            self.initial_role = ""

        fmt_kwargs = {
            "ROLE": self.initial_role,
            "PROMPT": self.initial_prompt,
            "NUM_PROMPTS": self.population_size,
            "PROBLEM_DESCRIPTION": self.problem_description,
        }
        if self.evolve_constraints:
            fmt_kwargs["CONSTRAINTS"] = self.initial_constraints
        request = self._paraphrasing_template.format(**fmt_kwargs)
        answer = self._llm_query([request])[0]
        extracted = extract_json(answer)
        if extracted is None or "prompts" not in extracted:
            logger.warning(
                "Failed to extract prompts from LLM response, using fallback"
            )
            prompts_data = [
                {"role": self.initial_role, "prompt": self.initial_prompt}
            ] * self.population_size
        else:
            prompts_data = extracted["prompts"]

        if not isinstance(prompts_data, list) or len(prompts_data) == 0:
            logger.warning("Invalid prompts_data format, using fallback")
            prompts_data = [
                {"role": self.initial_role, "prompt": self.initial_prompt}
            ] * self.population_size

        initial_population = []
        fixed_role = self.initial_role if not self.evolve_role else None

        for p_data in prompts_data:
            if isinstance(p_data, dict):
                if self.evolve_role:
                    role = p_data.get(
                        "system_behavior",
                        p_data.get("role", self.initial_role),
                    )
                else:
                    role = fixed_role or ""
                text = p_data.get(
                    "task_description",
                    p_data.get("prompt", str(p_data)),
                )
                constraints = (
                    p_data.get("output_constraints", "")
                    if self.evolve_constraints
                    else ""
                )
                if self._role_only or self._constraints_only:
                    text = self.initial_prompt
                initial_population.append(
                    Prompt(
                        text=text,
                        role=role,
                        constraints=constraints,
                        origin=PromptOrigin.APE,
                    )
                )
            else:
                role = (
                    fixed_role or self.initial_role
                    if not self.evolve_role
                    else self.initial_role
                )
                initial_population.append(
                    Prompt(text=p_data, role=role, origin=PromptOrigin.APE)
                )

        initial_population[-1] = Prompt(
            text=self.initial_prompt,
            role=(
                fixed_role or self.initial_role
                if not self.evolve_role
                else self.initial_role
            ),
            constraints=(
                self.initial_constraints if self.evolve_constraints else ""
            ),
            origin=PromptOrigin.MANUAL,
        )
        self._evaluation(initial_population)
        initial_population = self._reranking(initial_population)
        return initial_population

    def _cache_data(self, data: Any, savepath: os.PathLike) -> None:
        """Writes the data to the yaml file.

        Args:
            data (Any): data to be cached.
            savepath (os.PathLike): a path to saving file.
        """
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "w") as f:
            yaml.dump(data, f)

    def _cache_population(
        self, population: List[Prompt], savepath: os.PathLike
    ) -> None:
        """Caching a population of prompts to file.
        If self.use_cache is False this function will do nothing.

        Args:
            population (List[Prompt]): prompt population.
            savepath (os.PathLike): a path to saving file.
        """
        if self.use_cache is False:
            return

        best_score = population[0].score
        average_score = statistics.mean([prompt.score for prompt in population])
        data = {
            "best_score": best_score,
            "average_score": average_score,
            "prompts": [prompt.to_dict() for prompt in population],
        }
        self._cache_data(data, savepath)

    def _selection(self, population: List[Prompt]) -> List[Prompt]:
        """Provides selection operation.
        In current implementation we want to select parents
        with different scores.
        But when there is difficult to do so (trial number check),
        it will just sample anyways.

        Probabilities - normalized scores.

        Args:
            population (List[Prompt]): prompt population to select from.

        Returns:
            List[Prompt]: selected prompts.
        """
        selected_population = []

        scores = np.array([prompt.score for prompt in population])
        if np.sum(scores) == 0:
            probas = np.ones(len(scores)) / len(scores)
        else:
            probas = scores / np.sum(scores)

        trial = 0
        anyways = False
        while len(selected_population) < 2 * self.population_size:
            parents = np.random.choice(
                population, size=2, replace=False, p=probas
            )
            if parents[0].score != parents[1].score or anyways:
                selected_population.extend(parents)
            trial += 1
            if trial > 1000:
                anyways = True

        return selected_population

    def _survive(
        self, population: List[Prompt], temperature: float = None
    ) -> List[Prompt]:
        """Final selection before going into new epoch.
        Probabilities are based on softmax function with temperature (if set).

        Args:
            population (List[Prompt]): population to select from.
            temperature (float, optional): temperature parameter for softmax.
                Defaults to None.

        Returns:
            List[Prompt]: selected (survived) prompts.
        """
        scores = np.array([prompt.score for prompt in population])
        if temperature is not None:
            scores /= temperature
        probas = softmax(scores)
        return np.random.choice(
            population, size=self.population_size, replace=False, p=probas
        )

    def _gen_short_term_reflection_prompt(
        self, prompt1: Prompt, prompt2: Prompt
    ) -> Tuple[str, Prompt, Prompt]:
        """Generates short-term reflection request into model.

        Args:
            prompt1 (Prompt): first prompt.
            prompt2 (Prompt): second prompt.

        Returns:
            Tuple[str, Prompt, Prompt]:
                string request, worse prompt, better prompt.
        """
        if prompt1.score > prompt2.score:
            better_prompt, worse_prompt = prompt1, prompt2
        else:
            better_prompt, worse_prompt = prompt2, prompt1

        fmt_kwargs = {
            "PROBLEM_DESCRIPTION": self.problem_description,
            "WORSE_PROMPT_ROLE": worse_prompt.role,
            "WORSE_PROMPT_TEXT": worse_prompt.text,
            "BETTER_PROMPT_ROLE": better_prompt.role,
            "BETTER_PROMPT_TEXT": better_prompt.text,
            "WORSE_SCORE": self._format_score(worse_prompt.score),
            "BETTER_SCORE": self._format_score(better_prompt.score),
        }
        if self.evolve_constraints or self._constraints_only:
            fmt_kwargs["WORSE_PROMPT_CONSTRAINTS"] = worse_prompt.constraints
            fmt_kwargs["BETTER_PROMPT_CONSTRAINTS"] = better_prompt.constraints
        if self._role_only or self._constraints_only:
            fmt_kwargs["FROZEN_PROMPT_TEXT"] = self.initial_prompt
            fmt_kwargs["FROZEN_PROMPT_ROLE"] = self.initial_role or ""
        request = self._short_term_template.format(**fmt_kwargs)

        return request, worse_prompt, better_prompt

    def _make_output_path(self, filename: str) -> os.PathLike:
        """Creates full path for logging based on current iteration.

        Args:
            filename (str): the file name to save.

        Returns:
            os.PathLike: final path to save.
        """
        return os.path.join(
            self.output_path, f"Iteration{self.iteration}", f"{filename}.yaml"
        )

    def _short_term_reflection(
        self,
        population: list[Prompt],
    ) -> Tuple[List[str], List[Prompt], List[Prompt]]:
        """Short-term reflection before crossovering two individuals.

        Args:
            population (list[Prompt]): parenting population.

        Returns:
            Tuple[List[str], List[Prompt], List[Prompt]]:
                generated short-term hints,
                worse prompts,
                better prompts.
        """
        requests = []
        worse_prompts = []
        better_prompts = []
        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i + 1]

            request, worse_p, better_p = self._gen_short_term_reflection_prompt(
                parent_1, parent_2
            )
            requests.append(request)
            worse_prompts.append(worse_p)
            better_prompts.append(better_p)

        responses = self._llm_query(requests)
        responses = [
            extract_answer(response, self.HINT_TAGS, format_mismatch_label="")
            for response in responses
        ]
        return responses, worse_prompts, better_prompts

    def _crossover(
        self,
        short_term_reflection_tuple: Tuple[
            List[str], List[Prompt], List[Prompt]
        ],
    ) -> List[Prompt]:
        """Provides crossover operation.

        Args:
            short_term_reflection_tuple
                (Tuple[List[str], List[Prompt], List[Prompt]]):
                    outputs of short-term reflection.

        Returns:
            List[Prompt]: new crossed prompts population.
        """
        reflection_contents, worse_prompts, better_prompts = (
            short_term_reflection_tuple
        )
        requests = []
        for reflection, worse_p, better_p in zip(
            reflection_contents, worse_prompts, better_prompts
        ):
            fmt_kwargs = {
                "PROBLEM_DESCRIPTION": self.problem_description,
                "WORSE_PROMPT_ROLE": worse_p.role,
                "WORSE_PROMPT_TEXT": worse_p.text,
                "BETTER_PROMPT_ROLE": better_p.role,
                "BETTER_PROMPT_TEXT": better_p.text,
                "SHORT_TERM_REFLECTION": reflection,
                "WORSE_SCORE": self._format_score(worse_p.score),
                "BETTER_SCORE": self._format_score(better_p.score),
            }
            if self.evolve_constraints or self._constraints_only:
                fmt_kwargs["WORSE_PROMPT_CONSTRAINTS"] = worse_p.constraints
                fmt_kwargs["BETTER_PROMPT_CONSTRAINTS"] = better_p.constraints
            if self._role_only or self._constraints_only:
                fmt_kwargs["FROZEN_PROMPT_TEXT"] = self.initial_prompt
                fmt_kwargs["FROZEN_PROMPT_ROLE"] = self.initial_role or ""
            request = self._crossover_template.format(**fmt_kwargs)
            requests.append(request)

        responses = self._llm_query(requests)
        crossed_population = []
        for i, response in enumerate(responses):
            extracted = extract_json(response)
            if extracted is None:
                extracted = {}

            if self._role_only:
                role = extracted.get(
                    "system_behavior", extracted.get("role", "")
                )
                text = self.initial_prompt
                constraints = ""
            elif self._constraints_only:
                role = self.initial_role or ""
                text = self.initial_prompt
                constraints = extracted.get("output_constraints", "")
            else:
                if self.evolve_role:
                    role = extracted.get(
                        "system_behavior", extracted.get("role", "")
                    )
                else:
                    better_p = better_prompts[i]
                    role = (
                        better_p.role
                        if better_p.role
                        else self.initial_role or ""
                    )
                text = extracted.get(
                    "task_description",
                    extracted.get(
                        "prompt",
                        extract_answer(
                            response,
                            self.PROMPT_TAGS,
                            format_mismatch_label="",
                        ),
                    ),
                )
                constraints = (
                    extracted.get("output_constraints", "")
                    if self.evolve_constraints
                    else ""
                )
            crossed_population.append(
                Prompt(text=text, role=role, constraints=constraints)
            )

        assert len(crossed_population) == self.population_size
        return crossed_population

    def _update_elitist(self, population: List[Prompt]) -> None:
        scores = [prompt.score for prompt in population]
        best_score, best_sample_idx = max(scores), np.argmax(np.array(scores))

        if (
            self.best_score_overall is None
            or best_score >= self.best_score_overall
        ):
            self.best_score_overall = best_score
            self.best_prompt_overall = population[best_sample_idx].text
            self.best_constraints_overall = population[
                best_sample_idx
            ].constraints
            self.elitist = population[best_sample_idx]
            logger.info(f"""Iteration {self.iteration}
                Elitist score: {self.best_score_overall}""")
            logger.debug(f"Elitist text:\n{self.elitist.text}")

    def _update_iter(self, population: List[Prompt]) -> None:
        """Updates iteration. Cache current state.
        Also tracks elitist freeze: if the elitist role has not changed
        for ELITIST_MAX_FREEZE consecutive epochs, forces the best
        candidate with a different role to become the new elitist.

        Args:
            population (List[Prompt]): current population.
        """
        logger.info(f"Iteration {self.iteration} finished...")
        logger.info(f"Best score: {self.best_score_overall}")

        if self.use_enhancements:
            current_role = self.elitist.role if self.elitist else ""
            if current_role == self._prev_elitist_role:
                self._elitist_freeze_count += 1
            else:
                self._elitist_freeze_count = 0
            self._prev_elitist_role = current_role

            if self._elitist_freeze_count >= self.ELITIST_MAX_FREEZE:
                diverse = [
                    p
                    for p in population
                    if p.role != current_role and p.score is not None
                ]
                if diverse:
                    best_diverse = max(diverse, key=lambda p: p.score)
                    logger.debug(
                        f"Elitist frozen {self._elitist_freeze_count} epochs, "
                        f"forcing diverse candidate: '{best_diverse.role[:60]}'"
                    )
                    self.elitist = best_diverse
                    self._prev_elitist_role = best_diverse.role
                    self._elitist_freeze_count = 0

        population = self._reranking(population)
        self._cache_population(population, self._make_output_path("population"))

        self.iteration += 1

    def _long_term_reflection(self, short_term_reflections: List[str]) -> None:
        """Long-term reflection before mutation.

        Args:
            short_term_reflections (List[str]): short-term reflections.
        """
        long_term_kwargs = dict(
            PROBLEM_DESCRIPTION=self.problem_description,
            PRIOR_LONG_TERM_REFLECTION=self._long_term_reflection_str,
            NEW_SHORT_TERM_REFLECTIONS="\n".join(short_term_reflections),
        )
        if (
            self._role_only
            or self._constraints_only
            or self.text_only
            or self.evolve_role
        ):
            long_term_kwargs["TOP_PROMPTS_HISTORY"] = (
                self._format_top_prompts_history()
            )
        if self._constraints_only:
            long_term_kwargs["FROZEN_PROMPT_TEXT"] = self.initial_prompt
            long_term_kwargs["FROZEN_PROMPT_ROLE"] = self.initial_role or ""
        request = self._long_term_template.format(**long_term_kwargs)

        response = self._llm_query([request])[0]

        self._long_term_reflection_str = extract_answer(
            response, self.HINT_TAGS, format_mismatch_label=""
        )

    def _llm_query(self, requests: List[str]) -> List[str]:
        """Provides api to query requests to the model.
        Retries up to 3 times with exponential backoff on failure.

        Args:
            requests (List[str]): string requests.

        Returns:
            List[str]: model answers.
        """
        for attempt in range(3):
            try:
                answers = self.model.batch(requests)
                return [
                    a.content if isinstance(a, AIMessage) else a
                    for a in answers
                ]
            except Exception as e:
                if attempt < 2:
                    logger.warning(
                        f"LLM query failed (attempt {attempt + 1}): {e}. Retrying..."
                    )
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

    def _mutate(self) -> List[Prompt]:
        """Elitist-based mutation.

        Returns:
            List[Prompt]: generated population.
        """
        fmt_kwargs = {
            "PROBLEM_DESCRIPTION": self.problem_description,
            "LONG_TERM_REFLECTION": self._long_term_reflection_str,
            "ELITIST_PROMPT_ROLE": self.elitist.role,
            "ELITIST_PROMPT_TEXT": self.elitist.text,
            "ELITIST_SCORE": self._format_score(self.elitist.score),
        }
        if self.evolve_constraints or self._constraints_only:
            fmt_kwargs["ELITIST_PROMPT_CONSTRAINTS"] = self.elitist.constraints
        if (
            self._role_only
            or self._constraints_only
            or self.text_only
            or self.evolve_role
        ):
            fmt_kwargs["BAD_EXAMPLES"] = self._format_bad_examples()
        if self._role_only or self._constraints_only:
            fmt_kwargs["FROZEN_PROMPT_TEXT"] = self.initial_prompt
            fmt_kwargs["FROZEN_PROMPT_ROLE"] = self.initial_role or ""
        request = self._mutation_template.format(**fmt_kwargs)
        responses = self._llm_query([request] * self.population_size)
        mutated_population = []
        fixed_role = (
            self.elitist.role if self.elitist and not self.evolve_role else None
        )
        if fixed_role is None and not self.evolve_role:
            fixed_role = self.initial_role or ""

        for response in responses:
            extracted = extract_json(response)
            if extracted is None:
                extracted = {}

            if self._role_only:
                role = extracted.get(
                    "system_behavior", extracted.get("role", "")
                )
                text = self.initial_prompt
                constraints = ""
            elif self._constraints_only:
                role = self.initial_role or ""
                text = self.initial_prompt
                constraints = extracted.get("output_constraints", "")
            else:
                if self.evolve_role:
                    role = extracted.get(
                        "system_behavior", extracted.get("role", "")
                    )
                else:
                    role = fixed_role
                text = extracted.get(
                    "task_description",
                    extracted.get(
                        "prompt",
                        extract_answer(
                            response,
                            self.PROMPT_TAGS,
                            format_mismatch_label="",
                        ),
                    ),
                )
                constraints = (
                    extracted.get("output_constraints", "")
                    if self.evolve_constraints
                    else ""
                )
            mutated_population.append(
                Prompt(
                    text=text,
                    role=role,
                    constraints=constraints,
                    origin=PromptOrigin.MUTATED,
                )
            )
        return mutated_population

    def evolution(self, skip_validation: bool = False) -> str:
        """Provides evolution operation.

        Selection -> Short-term reflection -> Long-term reflection
            -> Elitist-based mutation -> Survival.

        After all self.num_epochs epochs the best three prompts are selected.
        They will be evaluated on test split of dataset then.
        And based on their test scores,
        the best prompt will be returned.

        Returns:
            str: best evoluted prompt
        """

        population = np.array(self._init_pop())
        self._cache_population(
            population, self._make_output_path("initial_population.yaml")
        )

        while self.iteration < self.num_epochs:
            parent_population = self._selection(population)

            short_term_reflection_tuple = self._short_term_reflection(
                parent_population
            )
            self._cache_data(
                short_term_reflection_tuple[0],
                self._make_output_path("short_term_reflections"),
            )

            crossed_population = self._crossover(short_term_reflection_tuple)

            self._evaluation(crossed_population)
            self._update_elitist(crossed_population)

            self._long_term_reflection(short_term_reflection_tuple[0])
            self._cache_data(
                self._long_term_reflection_str,
                self._make_output_path("long_term_reflection"),
            )

            mutated_population = self._mutate()
            self._evaluation(mutated_population)

            population = np.append(population, np.array(crossed_population))
            population = np.append(population, np.array(mutated_population))
            self._update_elitist(population)
            population = self._survive(population, temperature=1e-1)

            if self.elitist is not None and self.elitist not in population:
                logger.debug("Elitist should always live")
                population = np.append(population, np.array([self.elitist]))

            if self.use_enhancements:
                self._update_hall_of_fame(population)
            self._cache_data(
                self._elitist_bad_examples,
                self._make_output_path("bad_examples"),
            )
            self._cache_data(
                [
                    {
                        "score": self._format_score(p.score),
                        "text": p.text,
                        "role": p.role,
                        "constraints": p.constraints,
                    }
                    for p in self._hall_of_fame[:5]
                ],
                self._make_output_path("top_prompts_history"),
            )
            self._update_iter(population)

        logger.info(f"BEST TRAIN SCORE: {self.best_score_overall}")

        population = self._reranking(population)
        final_candidates = list(population[:3])
        if self.elitist is not None:
            if not any(
                c.text == self.elitist.text
                and c.role == self.elitist.role
                and c.constraints == self.elitist.constraints
                for c in final_candidates
            ):
                final_candidates.append(self.elitist)

        if self.use_enhancements:
            seen = {(c.text, c.role, c.constraints) for c in final_candidates}
            for hof_p in self._hall_of_fame:
                if (hof_p.text, hof_p.role, hof_p.constraints) not in seen:
                    final_candidates.append(hof_p)
                    seen.add((hof_p.text, hof_p.role, hof_p.constraints))
                if len(final_candidates) >= 6:
                    break

        if not skip_validation:
            logger.info(
                f"Final validation: {len(final_candidates)} candidates "
                f"({'with HoF' if self.use_enhancements else 'no HoF'})"
            )
            final_candidates = np.array(final_candidates)
            self._evaluation(final_candidates, split="validation")
            final_candidates = self._reranking(final_candidates)
            self._cache_population(
                final_candidates,
                self._make_output_path("best_prompts_infer.yaml"),
            )
            self.elitist = final_candidates[0]
            self.best_prompt_overall = self.elitist.text
            self.best_role_overall = self.elitist.role
            self.best_constraints_overall = self.elitist.constraints
            self.best_score_overall = self.elitist.score
            logger.info(f"BEST VALIDATION SCORE: {self.best_score_overall}")
            logger.debug(f"BEST ROLE:\n{self.best_role_overall}")
            logger.debug(f"BEST PROMPT:\n{self.best_prompt_overall}")
            if self.best_constraints_overall:
                logger.debug(
                    f"BEST CONSTRAINTS:\n{self.best_constraints_overall}"
                )
        else:
            logger.info("Skipping final validation (intermediate phase).")
            if self.elitist is not None:
                self.best_prompt_overall = self.elitist.text
                self.best_role_overall = self.elitist.role
                self.best_constraints_overall = self.elitist.constraints
            logger.info(f"BEST TRAIN SCORE (kept): {self.best_score_overall}")

        return self.best_prompt_overall
