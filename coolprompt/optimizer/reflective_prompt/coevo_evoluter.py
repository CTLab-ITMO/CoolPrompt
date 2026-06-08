import re
from typing import Dict, List, Optional, Tuple

from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    model_validator,
)
from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.reflective_prompt.coevo_base_evoluter import (
    ReflectiveEvoluter,
)
from coolprompt.optimizer.reflective_prompt.prompt import Prompt, PromptOrigin
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json, extract_answer
from coolprompt.utils.prompt_templates.reflective_templates_coevo_enhanced import (
    PARAPHRASING_TEMPLATE_COEVO_ENH,
    SHORT_TERM_REFLECTION_TEMPLATE_COEVO_ENH,
    LONG_TERM_REFLECTION_TEMPLATE_COEVO_ENH,
    CROSSOVER_TEMPLATE_COEVO_ENH,
    MUTATION_TEMPLATE_COEVO_ENH,
    PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_ENH,
)
from coolprompt.utils.prompt_templates.reflective_templates_coevo_per_field import (
    PARAPHRASING_TEMPLATE_COEVO_PF,
    SHORT_TERM_REFLECTION_TEMPLATE_COEVO_PF,
    LONG_TERM_REFLECTION_TEMPLATE_COEVO_PF,
    CROSSOVER_TEMPLATE_COEVO_PF,
    MUTATION_TEMPLATE_COEVO_PF,
    PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_PF,
)


def _sanitize(value: str) -> str:
    value = value.strip().strip('"').strip("'").strip()
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u2028\u2029]", "", value)
    return value


class _ThreeFieldOutput(BaseModel):
    task_description: str = ""
    system_behavior: str = ""
    output_constraints: str = ""

    @field_validator(
        "task_description",
        "system_behavior",
        "output_constraints",
        mode="before",
    )
    @classmethod
    def clean_field(cls, v):
        return _sanitize(str(v)) if v else ""

    @model_validator(mode="after")
    def check_task_not_empty(self):
        if not self.task_description:
            raise ValueError("task_description is empty")
        return self


class CoevoEvoluter(ReflectiveEvoluter):
    """Evoluter that coevolves all three prompt fields simultaneously.

    Optimizes task_description, system_behavior and output_constraints
    together in each epoch. Uses Pydantic to validate LLM outputs and
    runs field ablation at the end to pick the best field combination.
    """

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
        population_size: int = 10,
        num_epochs: int = 10,
        output_path: str = "./coevo_outputs",
        use_cache: bool = True,
        use_enhancements: bool = True,
        use_bad_examples: Optional[bool] = None,
        val_evaluator: Optional[Evaluator] = None,
    ) -> None:
        super().__init__(
            model=model,
            evaluator=evaluator,
            train_dataset=train_dataset,
            train_targets=train_targets,
            validation_dataset=validation_dataset,
            validation_targets=validation_targets,
            problem_description=problem_description,
            initial_prompt=initial_prompt,
            initial_role=initial_role,
            initial_constraints=initial_constraints,
            evolve_role=True,
            evolve_constraints=True,
            population_size=population_size,
            num_epochs=num_epochs,
            output_path=output_path,
            use_cache=use_cache,
            use_enhancements=use_enhancements,
            use_bad_examples=use_bad_examples,
            freeze_text=False,
            text_only=False,
            val_evaluator=val_evaluator,
        )
        self.candidates: List[Dict] = []

        self._paraphrasing_template = PARAPHRASING_TEMPLATE_COEVO_ENH
        self._crossover_template = CROSSOVER_TEMPLATE_COEVO_ENH
        self._mutation_template = MUTATION_TEMPLATE_COEVO_ENH
        self._short_term_template = SHORT_TERM_REFLECTION_TEMPLATE_COEVO_ENH
        self._long_term_template = LONG_TERM_REFLECTION_TEMPLATE_COEVO_ENH
        self._initial_prompt_template = PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_ENH

    def _llm_query(self, requests: List[str]) -> List[str]:
        results = []
        for req in requests:
            results.extend(super()._llm_query([_sanitize(req)]))
        return results

    def _parse_3f_response(
        self,
        response: str,
        fallback_text: str = "",
        fallback_role: str = "",
        fallback_constraints: str = "",
    ) -> Dict[str, str]:
        raw = extract_json(response) or {}

        try:
            parsed = _ThreeFieldOutput(
                task_description=raw.get("task_description") or fallback_text,
                system_behavior=raw.get("system_behavior") or fallback_role,
                output_constraints=raw.get("output_constraints")
                or fallback_constraints,
            )
        except (ValidationError, ValueError) as e:
            logger.warning(
                f"_parse_3f_response validation failed ({e}), using fallback"
            )
            return {
                "task_description": _sanitize(fallback_text),
                "system_behavior": _sanitize(fallback_role),
                "output_constraints": _sanitize(fallback_constraints),
            }
        return {
            "task_description": parsed.task_description,
            "system_behavior": parsed.system_behavior,
            "output_constraints": parsed.output_constraints,
        }

    def _crossover(
        self,
        short_term_reflection_tuple: Tuple[
            List[str], List[Prompt], List[Prompt]
        ],
    ) -> List[Prompt]:
        reflection_contents, worse_prompts, better_prompts = (
            short_term_reflection_tuple
        )
        requests = []
        for reflection, worse_p, better_p in zip(
            reflection_contents, worse_prompts, better_prompts
        ):
            request = self._crossover_template.format(
                PROBLEM_DESCRIPTION=self.problem_description,
                WORSE_PROMPT_TEXT=worse_p.text,
                WORSE_PROMPT_ROLE=worse_p.role,
                WORSE_PROMPT_CONSTRAINTS=worse_p.constraints,
                BETTER_PROMPT_TEXT=better_p.text,
                BETTER_PROMPT_ROLE=better_p.role,
                BETTER_PROMPT_CONSTRAINTS=better_p.constraints,
                SHORT_TERM_REFLECTION=reflection,
                WORSE_SCORE=self._format_score(worse_p.score),
                BETTER_SCORE=self._format_score(better_p.score),
            )
            requests.append(request)

        responses = self._llm_query(requests)
        crossed_population = []
        for i, response in enumerate(responses):
            fields = self._parse_3f_response(
                response,
                fallback_text=better_prompts[i].text,
                fallback_role=better_prompts[i].role,
                fallback_constraints=better_prompts[i].constraints,
            )
            crossed_population.append(
                Prompt(
                    text=fields["task_description"],
                    role=fields["system_behavior"],
                    constraints=fields["output_constraints"],
                    origin=PromptOrigin.EVOLUTED,
                )
            )

        assert len(crossed_population) == self.population_size
        return crossed_population

    def _mutate(self) -> List[Prompt]:
        request = self._mutation_template.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            LONG_TERM_REFLECTION=self._long_term_reflection_str,
            ELITIST_PROMPT_TEXT=self.elitist.text,
            ELITIST_PROMPT_ROLE=self.elitist.role,
            ELITIST_PROMPT_CONSTRAINTS=self.elitist.constraints,
            ELITIST_SCORE=self._format_score(self.elitist.score),
            BAD_EXAMPLES=self._format_bad_examples(),
        )
        responses = self._llm_query([request] * self.population_size)
        mutated_population = []
        for response in responses:
            fields = self._parse_3f_response(
                response,
                fallback_text=self.elitist.text,
                fallback_role=self.elitist.role,
                fallback_constraints=self.elitist.constraints,
            )
            mutated_population.append(
                Prompt(
                    text=fields["task_description"],
                    role=fields["system_behavior"],
                    constraints=fields["output_constraints"],
                    origin=PromptOrigin.MUTATED,
                )
            )
        return mutated_population

    def _eval_val(self, prompt: str, role: str, constraints: str) -> float:
        result = self.val_evaluator.evaluate(
            prompt=prompt,
            dataset=self.validation_dataset,
            targets=self.validation_targets,
            system_role=role or None,
            constraints=constraints or None,
        )
        assert isinstance(result, float)
        return result

    def _field_ablation(
        self,
        best_text: str,
        best_role: str,
        best_constraints: str,
        score_text_role_constraints: Optional[float] = None,
    ) -> Tuple[List[Dict], Dict]:
        logger.info(
            "[Field Ablation] Evaluating field combinations on validation set..."
        )
        score_a = self._eval_val(best_text, "", "")
        logger.info(f"  text_only:             {score_a:.4f}")
        score_b = self._eval_val(best_text, best_role, "")
        logger.info(f"  text_role:             {score_b:.4f}")

        candidates = [
            {
                "combo": "text_only",
                "prompt": best_text,
                "role": "",
                "constraints": "",
                "val_score": score_a,
            },
            {
                "combo": "text_role",
                "prompt": best_text,
                "role": best_role,
                "constraints": "",
                "val_score": score_b,
            },
        ]

        if best_constraints:
            if score_text_role_constraints is None:
                score_text_role_constraints = self._eval_val(
                    best_text, best_role, best_constraints
                )
            logger.info(
                f"  text_role_constraints: {score_text_role_constraints:.4f}"
            )
            candidates.append(
                {
                    "combo": "text_role_constraints",
                    "prompt": best_text,
                    "role": best_role,
                    "constraints": best_constraints,
                    "val_score": score_text_role_constraints,
                }
            )

        _combo_order = {
            "text_only": 0,
            "text_role": 1,
            "text_role_constraints": 2,
        }
        best_c = max(
            candidates, key=lambda c: (c["val_score"], _combo_order[c["combo"]])
        )
        logger.info(
            f"Best combo: {best_c['combo']} (val={best_c['val_score']:.4f})"
        )
        return candidates, best_c

    def evolution(self) -> Optional[str]:
        super().evolution()

        if self.best_prompt_overall:
            candidates, best_c = self._field_ablation(
                best_text=self.best_prompt_overall,
                best_role=self.best_role_overall or "",
                best_constraints=self.best_constraints_overall or "",
                score_text_role_constraints=self.best_score_overall,
            )
            self.candidates = candidates
            self.best_prompt_overall = best_c["prompt"]
            self.best_role_overall = best_c["role"]
            self.best_constraints_overall = best_c["constraints"]
            self.best_score_overall = best_c["val_score"]
            logger.info(
                f"Field ablation done. Best combo: {best_c['combo']} "
                f"(val={best_c['val_score']:.4f})"
            )

        return self.best_prompt_overall


class PerFieldCoevoEvoluter(CoevoEvoluter):
    """CoevoEvoluter variant that uses per-field reflection hints.

    Each reflection step produces three separate hints — one each for
    task_description, system_behavior, and output_constraints — instead of a
    single combined hint. Crossover and mutation templates consume these
    field-specific hints directly.

    All other enhancements (HoF, role length penalty, similarity penalty,
    field ablation) are inherited from CoevoEvoluter unchanged.
    """

    HINT_TASK_TAGS = ("<hint_task>", "</hint_task>")
    HINT_ROLE_TAGS = ("<hint_role>", "</hint_role>")
    HINT_CONSTRAINTS_TAGS = ("<hint_constraints>", "</hint_constraints>")

    _FALLBACK_HINT = "(no hint)"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paraphrasing_template = PARAPHRASING_TEMPLATE_COEVO_PF
        self._crossover_template = CROSSOVER_TEMPLATE_COEVO_PF
        self._mutation_template = MUTATION_TEMPLATE_COEVO_PF
        self._short_term_template = SHORT_TERM_REFLECTION_TEMPLATE_COEVO_PF
        self._long_term_template = LONG_TERM_REFLECTION_TEMPLATE_COEVO_PF
        self._initial_prompt_template = PROMPT_BY_DESCRIPTION_TEMPLATE_COEVO_PF

        self._per_field_short_hints: List[Dict[str, str]] = []
        self._per_field_long_hints: Dict[str, str] = {
            "task": self._FALLBACK_HINT,
            "role": self._FALLBACK_HINT,
            "constraints": self._FALLBACK_HINT,
        }

    def _parse_per_field_hints(self, response: str) -> Dict[str, str]:
        task = extract_answer(
            response, self.HINT_TASK_TAGS, format_mismatch_label=""
        ).strip()
        role = extract_answer(
            response, self.HINT_ROLE_TAGS, format_mismatch_label=""
        ).strip()
        constraints = extract_answer(
            response, self.HINT_CONSTRAINTS_TAGS, format_mismatch_label=""
        ).strip()
        return {
            "task": task or self._FALLBACK_HINT,
            "role": role or self._FALLBACK_HINT,
            "constraints": constraints or self._FALLBACK_HINT,
        }

    def _short_term_reflection(self, population):
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

        self._per_field_short_hints = []
        combined_strings = []
        for response in responses:
            hints = self._parse_per_field_hints(response)
            self._per_field_short_hints.append(hints)
            combined_strings.append(
                f"task_description: {hints['task']}\n"
                f"system_behavior: {hints['role']}\n"
                f"output_constraints: {hints['constraints']}"
            )

        return combined_strings, worse_prompts, better_prompts

    def _long_term_reflection(self, short_term_reflections: List[str]) -> None:
        request = self._long_term_template.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            TOP_PROMPTS_HISTORY=self._format_top_prompts_history(),
            PRIOR_TASK_HINT=self._per_field_long_hints["task"],
            PRIOR_ROLE_HINT=self._per_field_long_hints["role"],
            PRIOR_CONSTRAINTS_HINT=self._per_field_long_hints["constraints"],
            NEW_SHORT_TERM_REFLECTIONS="\n---\n".join(short_term_reflections),
        )
        response = self._llm_query([request])[0]
        hints = self._parse_per_field_hints(response)
        self._per_field_long_hints = hints
        self._long_term_reflection_str = (
            f"task_description: {hints['task']}\n"
            f"system_behavior: {hints['role']}\n"
            f"output_constraints: {hints['constraints']}"
        )

    def _crossover(
        self,
        short_term_reflection_tuple: Tuple[
            List[str], List[Prompt], List[Prompt]
        ],
    ) -> List[Prompt]:
        _, worse_prompts, better_prompts = short_term_reflection_tuple
        requests = []
        for i, (worse_p, better_p) in enumerate(
            zip(worse_prompts, better_prompts)
        ):
            hints = (
                self._per_field_short_hints[i]
                if i < len(self._per_field_short_hints)
                else {
                    "task": self._FALLBACK_HINT,
                    "role": self._FALLBACK_HINT,
                    "constraints": self._FALLBACK_HINT,
                }
            )
            request = self._crossover_template.format(
                PROBLEM_DESCRIPTION=self.problem_description,
                WORSE_PROMPT_TEXT=worse_p.text,
                WORSE_PROMPT_ROLE=worse_p.role,
                WORSE_PROMPT_CONSTRAINTS=worse_p.constraints,
                BETTER_PROMPT_TEXT=better_p.text,
                BETTER_PROMPT_ROLE=better_p.role,
                BETTER_PROMPT_CONSTRAINTS=better_p.constraints,
                TASK_HINT=hints["task"],
                ROLE_HINT=hints["role"],
                CONSTRAINTS_HINT=hints["constraints"],
                WORSE_SCORE=self._format_score(worse_p.score),
                BETTER_SCORE=self._format_score(better_p.score),
            )
            requests.append(request)

        responses = self._llm_query(requests)
        crossed_population = []
        for i, response in enumerate(responses):
            fields = self._parse_3f_response(
                response,
                fallback_text=better_prompts[i].text,
                fallback_role=better_prompts[i].role,
                fallback_constraints=better_prompts[i].constraints,
            )
            crossed_population.append(
                Prompt(
                    text=fields["task_description"],
                    role=fields["system_behavior"],
                    constraints=fields["output_constraints"],
                    origin=PromptOrigin.EVOLUTED,
                )
            )

        assert len(crossed_population) == self.population_size
        return crossed_population

    def _mutate(self) -> List[Prompt]:
        hints = self._per_field_long_hints
        request = self._mutation_template.format(
            PROBLEM_DESCRIPTION=self.problem_description,
            TASK_HINT=hints["task"],
            ROLE_HINT=hints["role"],
            CONSTRAINTS_HINT=hints["constraints"],
            ELITIST_PROMPT_TEXT=self.elitist.text,
            ELITIST_PROMPT_ROLE=self.elitist.role,
            ELITIST_PROMPT_CONSTRAINTS=self.elitist.constraints,
            ELITIST_SCORE=self._format_score(self.elitist.score),
            BAD_EXAMPLES=self._format_bad_examples(),
        )
        responses = self._llm_query([request] * self.population_size)
        mutated_population = []
        for response in responses:
            fields = self._parse_3f_response(
                response,
                fallback_text=self.elitist.text,
                fallback_role=self.elitist.role,
                fallback_constraints=self.elitist.constraints,
            )
            mutated_population.append(
                Prompt(
                    text=fields["task_description"],
                    role=fields["system_behavior"],
                    constraints=fields["output_constraints"],
                    origin=PromptOrigin.MUTATED,
                )
            )
        return mutated_population
