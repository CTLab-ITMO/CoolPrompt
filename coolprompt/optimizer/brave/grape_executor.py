import json
import re
from typing import Dict, List, Tuple, Callable, Optional, Any, Mapping, Sequence

import numpy as np

from coolprompt.optimizer.begrape.actions import ActionResult
from coolprompt.optimizer.begrape.core_states import (
    OptimizerState,
    EpistemicMemory
)
from coolprompt.utils.prompt_templates.regps_templates import (
    REGPS_TEXTUAL_GRADIENT_TEMPLATE,
    SHORT_TERM_TEXTGRAD_TEMPLATE,
)
from coolprompt.utils.prompt_templates.reflective_templates import (
    REFLECTIVEPROMPT_MUTATION_TEMPLATE,
    REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE,
    REFLECTIVEPROMPT_CROSSOVER_TEMPLATE
)
from coolprompt.utils.parsing import extract_answer


def _safe_template_local(template: str, **kwargs: Any) -> str:
    escaped = {
        k: str(v).replace("{", "{{").replace("}", "}}")
        for k, v in kwargs.items()
    }
    return template.format(**escaped)


class CoolPromptReGPSActionExecutor:
    """ActionExecutor adapter that reuses ReGPS/ReflectivePrompt core logic.

    Notes:
    - No dependency on CoolPrompt evaluator internals.
    - Evaluation is injected via callback.
    - LLM API is injected via callback.

    Required callbacks:
      llm_query_fn(requests: List[str]) -> List[str]

    Optional callbacks:
      evaluate_fn(
        prompts: List[str], split: str, failed_examples: int, **kwargs
      ) -> List[Dict]
        Each dict may contain:
          - score: float
          - bad_examples: List[{"input": str, "output": str, "correct": str}]

      discover_fn(
          population: List[str],
          state: OptimizerState,
          memory: EpistemicMemory,
          train_data: Any
      ) -> Dict[str, Any]
    """

    PROMPT_TAGS = ("<prompt>", "</prompt>")
    HINT_TAGS = ("<hint>", "</hint>")
    FEEDBACK_TAGS = ("<feedback>", "</feedback>")

    def __init__(
        self,
        problem_description: str,
        llm_query_fn: Callable[[List[str]], List[str]],
        evaluate_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        discover_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        population_size: int = 10,
        bad_examples_number: int = 5,
    ) -> None:
        self.problem_description = problem_description
        self.llm_query_fn = llm_query_fn
        self.evaluate_fn = evaluate_fn
        self.discover_fn = discover_fn
        self.population_size = population_size
        self.bad_examples_num = bad_examples_number

        self.prompt_scores: Dict[str, float] = {}
        self.prompt_bad_examples: Dict[str, List[Dict[str, str]]] = {}
        self.textual_gradients: Dict[str, str] = {}
        self.short_term_reflections: List[str] = []
        self.long_term_reflection_str: str = ""
        self.elitist_prompt: Optional[str] = None
        self.elitist_score: float = -1e18
        self.last_population: List[str] = []
        self.last_offspring: List[str] = []
        self.best_quality_seen: float = -1e18
        self.discovered_inputs: List[str] = []

    def _normalize_responses(self, responses: List[Any]) -> List[str]:
        out: List[str] = []
        for r in responses:
            if hasattr(r, "content"):
                out.append(str(r.content))
            else:
                out.append(str(r))
        return out

    def _llm_query(self, requests: List[str]) -> List[str]:
        return self._normalize_responses(self.llm_query_fn(requests))

    def _make_bad_examples(self, bad_examples: List[Dict[str, str]]) -> str:
        if not bad_examples:
            return "Input: N/A\nModel Output: N/A\nCorrect Output: N/A"
        return "\n\n".join(
            [
                "\n".join(
                    (
                        f"Input: {ex.get('input', '')}",
                        f"Model Output: {ex.get('output', '')}",
                        f"Correct Output: {ex.get('correct', '')}",
                    )
                )
                for ex in bad_examples
            ]
        )

    def _gen_textual_gradient(self, prompt: str) -> str:
        bad_examples = self.prompt_bad_examples.get(prompt, [])
        request = _safe_template_local(
            REGPS_TEXTUAL_GRADIENT_TEMPLATE,
            PROBLEM_DESCRIPTION=self.problem_description,
            PROMPT=prompt,
            EXAMPLES=self._make_bad_examples(bad_examples),
        )
        answer = self._llm_query([request])[0]
        return extract_answer(
            answer=answer,
            tags=self.FEEDBACK_TAGS,
            format_mismatch_label=""
        )

    def _sort_by_score(self, population: List[str]) -> List[str]:
        return sorted(
            population,
            key=lambda p: self.prompt_scores.get(p, -1e18),
            reverse=True
        )

    def _selection_pairs(self, population: List[str]) -> List[Tuple[str, str]]:
        ordered = self._sort_by_score(population)
        if len(ordered) < 2:
            return []
        pairs: List[Tuple[str, str]] = []
        idx = 0
        while len(pairs) < self.population_size:
            p1 = ordered[idx % len(ordered)]
            p2 = ordered[(idx + 1) % len(ordered)]
            if p1 != p2:
                pairs.append((p1, p2))
            idx += 2
            if idx > 4 * len(ordered):
                break
        return pairs

    def _evaluate_population(
        self,
        population: List[str],
        split: str,
        train_data: Optional[Any] = None,
    ) -> Tuple[float, Optional[str]]:
        results = self.evaluate_fn(
            population,
            split,
            self.bad_examples_num,
            extra_inputs=self.discovered_inputs,
            train_data=train_data,
        )
        best_score = -1e18
        best_prompt: Optional[str] = None
        for prompt, row in zip(population, results):
            score = float(row.get("score", 0.0))
            self.prompt_scores[prompt] = score
            if "bad_examples" in row:
                self.prompt_bad_examples[prompt] = list(
                    row.get("bad_examples", [])
                )
            if score > best_score:
                best_score = score
                best_prompt = prompt
        return best_score, best_prompt

    def _action_eval(
        self,
        population: List[str],
        split: str = "train",
        cost_tokens: float = 220.0,
        train_data: Optional[Any] = None,
    ) -> ActionResult:
        if self.best_quality_seen > -1e17:
            prev_best = self.best_quality_seen
        else:
            prev_best = 0.0
        best_score, best_prompt = self._evaluate_population(
            population,
            split=split,
            train_data=train_data,
        )

        if best_prompt is not None:
            self.elitist_prompt = best_prompt
            self.elitist_score = best_score
            self.best_quality_seen = max(self.best_quality_seen, best_score)

        delta_q = max(best_score - prev_best, 0.0)
        artifacts = {
            "has_eval": True,
            "has_failures": True,
            "has_best_prompt": best_prompt is not None
        }
        return ActionResult(
            action="eval_high" if split == "validation" else "eval_low",
            delta_quality=delta_q,
            delta_coverage=0.0,
            delta_drift=0.0,
            cost_tokens=cost_tokens,
            payload={
                "best_prompt": best_prompt if best_prompt is not None else "",
                "best_quality": best_score,
                "population": population,
                "artifacts": artifacts,
            },
        )

    @staticmethod
    def _string_hash_embed(text: str, dim: int = 256) -> np.ndarray:
        """Dependency-free hashed character trigram embedding."""
        v = np.zeros(dim, dtype=np.float64)
        s = f"  {text.lower()}  "
        if len(s) < 3:
            idx = abs(hash(s)) % dim
            v[idx] = 1.0
            return v
        for i in range(len(s) - 2):
            tri = s[i:i + 3]
            idx = abs(hash(tri)) % dim
            v[idx] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    @staticmethod
    def _complexity_score(text: str) -> float:
        tokens = text.split()
        length_score = min(len(tokens) / 64.0, 1.0)
        digit_score = min(sum(ch.isdigit() for ch in text) / 10.0, 1.0)
        punct_score = min(sum(ch in ",;:()[]{}" for ch in text) / 12.0, 1.0)
        return 0.5 * length_score + 0.3 * digit_score + 0.2 * punct_score

    def _extract_seen_inputs(
        self,
        population: List[str],
        train_data: List[str]
    ) -> List[str]:
        seen: List[str] = []
        for p in population:
            for ex in self.prompt_bad_examples.get(p, []):
                inp = ex.get("input", "")
                if inp:
                    seen.append(str(inp))
        seen.extend(self.discovered_inputs)
        pool = self._extract_input_pool(train_data)
        seen.extend(pool[: min(len(pool), 128)])

        out: List[str] = []
        seen_set = set()
        for x in seen:
            if x not in seen_set:
                out.append(x)
                seen_set.add(x)
        return out

    def _generate_candidate_unknowns(
        self,
        population: List[str],
        k: int
    ) -> List[str]:
        elitist = self.elitist_prompt or (population[0] if population else "")
        if not elitist:
            return []
        request = (
            "You are a dataset hard-example generator for"
            "prompt optimization.\n"
            f"Task: {self.problem_description}\n"
            f"Current best prompt:\n{elitist}\n\n"
            f"Long-term reflection:\n{self.long_term_reflection_str}\n\n"
            f"Generate {k} NEW hard and diverse input examples likely"
            "to expose unknown failures.\n"
            "Return strict JSON: {\"inputs\": [\"...\", \"...\", ...]}"
        )
        raw = self._llm_query([request])[0]
        try:
            data = json.loads(raw)
            arr = data.get("inputs", []) if isinstance(data, Mapping) else []
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass

        quoted = re.findall(r"\"([^\"]+)\"", raw)
        if quoted:
            return [q.strip() for q in quoted if q.strip()]
        lines = [ln.strip("-* \t") for ln in raw.splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln.split()) > 2][:k]

    def _rank_unknown_candidates(
        self,
        candidates: Sequence[str],
        seen_inputs: Sequence[str],
        max_select: int,
    ) -> List[str]:
        if not candidates:
            return []

        seen_vecs = [
            self._string_hash_embed(x) for x in list(seen_inputs)[:512]
        ]
        if not seen_vecs:
            seen_vecs = [np.zeros(256, dtype=np.float64)]

        def novelty(c: str) -> float:
            cv = self._string_hash_embed(c)
            sims = [float(np.dot(cv, sv)) for sv in seen_vecs]
            return 1.0 - max(sims)

        scored: List[Tuple[float, str]] = []
        for c in candidates:
            n = novelty(c)
            comp = self._complexity_score(c)
            score = 0.75 * n + 0.25 * comp
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        selected: List[str] = []
        used = set()
        for _, cand in scored:
            if cand in used:
                continue
            selected.append(cand)
            used.add(cand)
            if len(selected) >= max_select:
                break
        return selected

    def _action_critic_gradient(self, population: List[str]) -> ActionResult:
        # Generate textual gradients for current ranked population
        ranked = self._sort_by_score(population)
        n = min(len(ranked), max(2, self.population_size))
        generated = 0
        for prompt in ranked[:n]:
            grad = self._gen_textual_gradient(prompt)
            if grad:
                self.textual_gradients[prompt] = grad
                generated += 1

        delta_cov = generated / max(n, 1)
        artifacts = {
            "has_gradients": generated > 0,
            "has_short_term": generated > 0
        }
        return ActionResult(
            action="critic_gradient",
            delta_quality=0.0,
            delta_coverage=delta_cov,
            delta_drift=0.01,
            cost_tokens=850.0,
            payload={"artifacts": artifacts},
        )

    def _action_reflective_crossover(
        self,
        population: List[str]
    ) -> ActionResult:
        pairs = self._selection_pairs(population)

        short_term_requests: List[str] = []
        pair_data: List[Tuple[str, str]] = []
        for p1, p2 in pairs:
            s1 = self.prompt_scores.get(p1, 0.0)
            s2 = self.prompt_scores.get(p2, 0.0)
            better, worse = (p1, p2) if s1 >= s2 else (p2, p1)
            better_fb = self.textual_gradients[better]
            worse_fb = self.textual_gradients[worse]
            req = _safe_template_local(
                SHORT_TERM_TEXTGRAD_TEMPLATE,
                PROBLEM_DESCRIPTION=self.problem_description,
                WORSE_PROMPT=worse,
                WORSE_PROMPT_FEEDBACK=worse_fb,
                BETTER_PROMPT=better,
                BETTER_PROMPT_FEEDBACK=better_fb,
            )
            short_term_requests.append(req)
            pair_data.append((worse, better))

        short_term_raw = self._llm_query(short_term_requests)
        self.short_term_reflections = [
            extract_answer(x, self.HINT_TAGS, format_mismatch_label="")
            for x in short_term_raw
        ]

        crossover_requests: List[str] = []
        for refl, (wrs, btr) in zip(self.short_term_reflections, pair_data):
            req = _safe_template_local(
                REFLECTIVEPROMPT_CROSSOVER_TEMPLATE,
                PROBLEM_DESCRIPTION=self.problem_description,
                WORSE_PROMPT=wrs,
                BETTER_PROMPT=btr,
                SHORT_TERM_REFLECTION=refl,
            )
            crossover_requests.append(req)

        crossover_raw = self._llm_query(crossover_requests)
        offspring = [
            extract_answer(
                answer=x,
                tags=self.PROMPT_TAGS,
                format_mismatch_label=""
            ).strip()
            for x in crossover_raw
        ]
        offspring = [x for x in offspring if x]
        if len(offspring) > self.population_size:
            offspring = offspring[: self.population_size]

        self.last_offspring = offspring
        self.last_population = offspring
        artifacts = {
            "has_short_term": len(self.short_term_reflections) > 0,
            "has_offspring": len(offspring) > 0,
        }
        return ActionResult(
            action="reflective_crossover",
            delta_quality=0.0,
            delta_coverage=min(
                len(offspring) / max(self.population_size, 1),
                1.0
            ),
            delta_drift=0.02,
            cost_tokens=780.0,
            payload={
                "population": offspring if offspring else population,
                "failure_tags": ["short_term_reflection", "crossover"],
                "artifacts": artifacts,
            },
        )

    def _action_memory_update(self) -> ActionResult:
        if not self.short_term_reflections:
            return ActionResult(
                action="memory_update",
                delta_quality=0.0,
                delta_coverage=0.0,
                delta_drift=0.0,
                cost_tokens=50.0,
                payload={"artifacts": {"has_memory_update": False}},
            )
        req = _safe_template_local(
            REFLECTIVEPROMPT_LONG_TERM_REFLECTION_TEMPLATE,
            PROBLEM_DESCRIPTION=self.problem_description,
            PRIOR_LONG_TERM_REFLECTION=self.long_term_reflection_str,
            NEW_SHORT_TERM_REFLECTIONS="\n".join(self.short_term_reflections),
        )
        answer = self._llm_query([req])[0]
        self.long_term_reflection_str = extract_answer(
            answer=answer,
            tags=self.HINT_TAGS,
            format_mismatch_label=""
        )
        return ActionResult(
            action="memory_update",
            delta_quality=0.0,
            delta_coverage=0.05,
            delta_drift=0.01,
            cost_tokens=260.0,
            payload={
                "artifacts": {
                    "has_memory_update": bool(self.long_term_reflection_str)
                }
            },
        )

    def _action_memory_mutation(self, population: List[str]) -> ActionResult:
        elitist = self.elitist_prompt
        req = _safe_template_local(
            REFLECTIVEPROMPT_MUTATION_TEMPLATE,
            PROBLEM_DESCRIPTION=self.problem_description,
            LONG_TERM_REFLECTION=self.long_term_reflection_str,
            ELITIST_PROMPT=elitist,
        )
        answers = self._llm_query([req] * self.population_size)
        mutated = [
            extract_answer(
                answer=x,
                tags=self.PROMPT_TAGS,
                format_mismatch_label=""
            ).strip()
            for x in answers
        ]
        mutated = [x for x in mutated if x]
        if len(mutated) > self.population_size:
            mutated = mutated[: self.population_size]
        self.last_population = mutated if mutated else population
        return ActionResult(
            action="memory_guided_mutation",
            delta_quality=0.0,
            delta_coverage=min(
                len(mutated) / max(self.population_size, 1),
                1.0
            ),
            delta_drift=0.02,
            cost_tokens=690.0,
            payload={
                "population": self.last_population,
                "failure_tags": ["mutation", "long_term_memory"],
                "artifacts": {
                    "has_offspring": len(mutated) > 0,
                    "has_best_prompt": bool(elitist)
                },
            },
        )

    def _action_discover_unknowns(
        self,
        population: List[str],
        state: OptimizerState,
        memory: EpistemicMemory,
        train_data: Any,
    ) -> ActionResult:
        # Default unknown-unknown mining:
        # 1) combine external candidate pool + LLM-generated hard candidates
        # 2) rank by novelty and structural complexity
        # 3) inject selected examples into discovered pool
        #   (and optional train_data)
        pool = train_data
        seen = self._extract_seen_inputs(population, train_data)
        pool_candidates = pool[: min(len(pool), 256)]
        llm_candidates = self._generate_candidate_unknowns(population, k=24)
        merged_candidates = [
            c for c in (pool_candidates + llm_candidates) if c and c not in seen
        ]

        select_k = 16
        selected = self._rank_unknown_candidates(
            candidates=merged_candidates,
            seen_inputs=seen,
            max_select=select_k
        )
        prev_n = len(self.discovered_inputs)
        self.discovered_inputs.extend(selected)
        # de-dup discovered pool
        dedup: List[str] = []
        ds = set()
        for x in self.discovered_inputs:
            if x not in ds:
                dedup.append(x)
                ds.add(x)
        self.discovered_inputs = dedup
        added = len(self.discovered_inputs) - prev_n

        # Make discovered inputs accessible to downstream evaluators.
        if isinstance(train_data, list):
            train_data.extend(selected)
        elif isinstance(train_data, Mapping):
            try:
                current = list(train_data.get("discovered_inputs", []))
                current.extend(selected)
                train_data["discovered_inputs"] = current  # type: ignore[index]
            except Exception:
                pass

        delta_cov = min(added / max(select_k, 1), 1.0)
        return ActionResult(
            action="discover_unknowns",
            delta_quality=0.0,
            delta_coverage=delta_cov,
            delta_drift=0.005,
            cost_tokens=520.0,
            payload={
                "new_inputs": selected,
                "discovered_pool_size": len(self.discovered_inputs),
                "failure_tags": ["unknown_unknowns", "novel_inputs"],
                "artifacts": {"has_eval": True},
            },
        )

    def execute(
        self,
        action: str,
        population: List[str],
        state: OptimizerState,
        memory: EpistemicMemory,
        train_data: Any,
        val_data: Any,
    ) -> ActionResult:
        # Keep latest view
        self.last_population = population

        if action == "eval_low":
            return self._action_eval(
                population,
                split="train",
                cost_tokens=220.0,
                train_data=train_data,
            )
        if action == "eval_high":
            return self._action_eval(
                population,
                split="validation",
                cost_tokens=1200.0,
                train_data=train_data,
            )
        if action == "critic_gradient":
            return self._action_critic_gradient(population)
        if action == "reflective_crossover":
            return self._action_reflective_crossover(population)
        if action == "memory_update":
            return self._action_memory_update()
        if action == "memory_guided_mutation":
            return self._action_memory_mutation(population)
        if action == "discover_unknowns":
            return self._action_discover_unknowns(
                population=population,
                state=state,
                memory=memory,
                train_data=train_data
            )

        return ActionResult(
            action=action,
            delta_quality=0.0,
            delta_coverage=0.0,
            delta_drift=0.0,
            cost_tokens=50.0,
            payload={},
        )
