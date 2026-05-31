import os
from typing import List, Optional, Tuple

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.ai import AIMessage

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.reflective_prompt.evoluter import ReflectiveEvoluter
from coolprompt.utils.logging_config import logger
from coolprompt.utils.parsing import extract_json
from coolprompt.utils.prompt_templates.reflective_templates_factorized import (
    DEDUP_ROLE_TEMPLATE,
    DEDUP_CONSTRAINTS_TEMPLATE,
)


class FactorizedEvoluter:

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
        population_size: int = 5,
        phase_epochs: Tuple[int, int, int] = (4, 3, 3),
        run_constraints_phase: bool = True,
        output_path: str = "./factorized_outputs",
        use_cache: bool = True,
        use_enhancements: bool = True,
        use_dedup: bool = True,
        val_evaluator: Optional[Evaluator] = None,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.val_evaluator = val_evaluator or evaluator
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.validation_dataset = validation_dataset
        self.validation_targets = validation_targets
        self.problem_description = problem_description
        self.initial_prompt = initial_prompt
        self.initial_role = initial_role or ""
        self.initial_constraints = initial_constraints or ""
        self.population_size = population_size
        self.phase_epochs = phase_epochs
        self.run_constraints_phase = run_constraints_phase and bool(
            initial_constraints
        )
        self.output_path = output_path
        self.use_cache = use_cache
        self.use_enhancements = use_enhancements
        self.use_dedup = use_dedup

        self.best_prompt_overall = None
        self.best_role_overall = None
        self.best_constraints_overall = None
        self.best_score_overall = None
        self.candidates: List[dict] = []

    def _make_phase_evoluter(
        self,
        phase_name: str,
        initial_prompt: Optional[str],
        initial_role: Optional[str],
        initial_constraints: Optional[str],
        num_epochs: int,
        evolve_role: bool,
        evolve_constraints: bool,
        freeze_text: bool,
    ) -> ReflectiveEvoluter:
        text_only = (
            not evolve_role and not freeze_text and not evolve_constraints
        )
        return ReflectiveEvoluter(
            model=self.model,
            evaluator=self.evaluator,
            train_dataset=self.train_dataset,
            train_targets=self.train_targets,
            validation_dataset=self.validation_dataset,
            validation_targets=self.validation_targets,
            problem_description=self.problem_description,
            initial_prompt=initial_prompt,
            initial_role=initial_role,
            initial_constraints=initial_constraints,
            evolve_role=evolve_role,
            evolve_constraints=evolve_constraints,
            freeze_text=freeze_text,
            text_only=text_only,
            population_size=self.population_size,
            num_epochs=num_epochs,
            use_cache=self.use_cache,
            output_path=os.path.join(self.output_path, phase_name),
            use_enhancements=self.use_enhancements,
        )

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
        best_text: Optional[str],
        best_role: Optional[str],
        best_constraints: str,
        score_text_role_constraints: Optional[float] = None,
    ) -> Tuple[List[dict], dict]:
        logger.info(
            "[Field Ablation] Evaluating all field combinations on validation set..."
        )
        assert best_text is not None and best_role is not None
        score_a = self._eval_val(best_text, "", "")
        logger.info(f"  text_only:              {score_a:.4f}")
        score_b = self._eval_val(best_text, best_role, "")
        logger.info(f"  text_role:              {score_b:.4f}")
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
                f"  text_role_constraints:  {score_text_role_constraints:.4f}"
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
        best_c = max(candidates, key=lambda c: c["val_score"])
        logger.info(f"Best combo: {best_c['combo']} (val={best_c['val_score']:.4f})")
        return candidates, best_c

    def _llm_call(self, request: str) -> str:
        responses = self.model.batch([request])
        r = responses[0]
        return r.content if isinstance(r, AIMessage) else r

    def _dedup_role(self, task_text: str, role: str) -> str:
        if not role or not self.use_dedup:
            return role
        try:
            parsed = extract_json(
                self._llm_call(
                    DEDUP_ROLE_TEMPLATE.format(TASK=task_text, ROLE=role)
                )
            )
            if parsed and "system_behavior" in parsed:
                cleaned = str(parsed["system_behavior"]).strip()
                if cleaned != role:
                    logger.info(
                        f"[Dedup role] seed cleaned: '{role[:80]}' '{cleaned[:80]}'"
                    )
                return cleaned
        except Exception as e:
            logger.warning(f"Dedup role failed: {e}. Using original.")
        return role

    def _dedup_constraints(
        self, task_text: str, role: str, constraints: str
    ) -> str:
        if not constraints or not self.use_dedup:
            return constraints
        try:
            parsed = extract_json(
                self._llm_call(
                    DEDUP_CONSTRAINTS_TEMPLATE.format(
                        TASK=task_text,
                        ROLE=role or "(none)",
                        CONSTRAINTS=constraints,
                    )
                )
            )
            if parsed and "output_constraints" in parsed:
                cleaned = str(parsed["output_constraints"]).strip()
                if cleaned != constraints:
                    logger.info(
                        f"[Dedup constraints] seed cleaned: '{constraints[:80]}' '{cleaned[:80]}'"
                    )
                return cleaned
        except Exception as e:
            logger.warning(f"Dedup constraints failed: {e}. Using original.")
        return constraints

    def evolution(self) -> str:
        last_phase = 3 if self.run_constraints_phase else 2

        logger.info(
            f"Factorized evolution: {self.phase_epochs[0]} + {self.phase_epochs[1]}"
            + (
                f" + {self.phase_epochs[2]} epochs"
                if self.run_constraints_phase
                else " epochs"
            )
            + f" | phases: text -> role"
            + (" -> constraints" if self.run_constraints_phase else "")
        )

        logger.info(
            f"[Phase 1/{last_phase}] Optimizing task_description ({self.phase_epochs[0]} epochs)"
        )
        p1 = self._make_phase_evoluter(
            phase_name="phase1_text",
            initial_prompt=self.initial_prompt,
            initial_role=None,
            initial_constraints=None,
            num_epochs=self.phase_epochs[0],
            evolve_role=False,
            evolve_constraints=False,
            freeze_text=False,
        )
        p1.evolution(skip_validation=True)
        best_text = p1.best_prompt_overall
        assert best_text is not None
        logger.info(f"Phase 1 best text score (train): {p1.best_score_overall:.4f}")
        logger.info(f"Phase 1 best text: {best_text[:120]}")

        logger.info(
            f"[Phase 2/{last_phase}] Optimizing system_behavior ({self.phase_epochs[1]} epochs)"
        )
        initial_role_for_p2 = self._dedup_role(
            best_text, self.initial_role or ""
        )
        skip_p2_val = self.run_constraints_phase
        p2 = self._make_phase_evoluter(
            phase_name="phase2_role",
            initial_prompt=best_text,
            initial_role=initial_role_for_p2,
            initial_constraints=None,
            num_epochs=self.phase_epochs[1],
            evolve_role=True,
            evolve_constraints=False,
            freeze_text=True,
        )
        p2.evolution(skip_validation=skip_p2_val)
        best_role = p2.best_role_overall
        logger.info(f"Phase 2 best role score (train): {p2.best_score_overall:.4f}")
        logger.info(f"Phase 2 best role: {(best_role or '')[:120]}")

        if not self.run_constraints_phase:
            self.initial_prompt = p1.initial_prompt
            self.initial_role = p2.initial_role or ""
            self.initial_constraints = ""
            candidates, best_c = self._field_ablation(
                best_text=best_text,
                best_role=p2.best_role_overall,
                best_constraints="",
            )
            self.candidates = candidates
            self.best_prompt_overall = best_c["prompt"]
            self.best_role_overall = best_c["role"]
            self.best_constraints_overall = best_c["constraints"]
            self.best_score_overall = best_c["val_score"]
            return self.best_prompt_overall

        val_text_only = self._eval_val(best_text, "", "")
        val_text_role = self._eval_val(best_text, best_role or "", "")
        logger.info(
            f"[Pre-Phase 3 check] text_only val: {val_text_only:.4f}, text_role val: {val_text_role:.4f}"
        )
        if val_text_only >= val_text_role:
            logger.info("Role does not improve on validation. Skipping Phase 3.")
            self.initial_prompt = p1.initial_prompt
            self.initial_role = p2.initial_role or ""
            self.initial_constraints = ""
            candidates, best_c = self._field_ablation(
                best_text=best_text,
                best_role=best_role or "",
                best_constraints="",
            )
            self.candidates = candidates
            self.best_prompt_overall = best_c["prompt"]
            self.best_role_overall = best_c["role"]
            self.best_constraints_overall = best_c["constraints"]
            self.best_score_overall = best_c["val_score"]
            return self.best_prompt_overall

        logger.info(
            f"[Phase 3/{last_phase}] Optimizing output_constraints ({self.phase_epochs[2]} epochs)"
        )
        initial_constraints_for_p3 = self._dedup_constraints(
            best_text, best_role or "", self.initial_constraints
        )
        if not initial_constraints_for_p3 and self.initial_constraints:
            logger.info("[Dedup] constraints redundant with task/role, using fallback seed")
            initial_constraints_for_p3 = "Return only the final answer."
        p3 = self._make_phase_evoluter(
            phase_name="phase3_constraints",
            initial_prompt=best_text,
            initial_role=best_role or "",
            initial_constraints=initial_constraints_for_p3,
            num_epochs=self.phase_epochs[2],
            evolve_role=False,
            evolve_constraints=True,
            freeze_text=False,
        )
        p3.evolution(skip_validation=False)
        logger.info(f"Phase 3 best constraints score: {p3.best_score_overall:.4f}")

        self.initial_prompt = p1.initial_prompt
        self.initial_role = p2.initial_role or ""
        self.initial_constraints = p3.initial_constraints or ""

        candidates, best_c = self._field_ablation(
            best_text=best_text,
            best_role=p2.best_role_overall,
            best_constraints=p3.best_constraints_overall or "",
            score_text_role_constraints=p3.best_score_overall,
        )
        self.candidates = candidates
        self.best_prompt_overall = best_c["prompt"]
        self.best_role_overall = best_c["role"]
        self.best_constraints_overall = best_c["constraints"]
        self.best_score_overall = best_c["val_score"]
        return self.best_prompt_overall
