"""Synthetic evaluation and red-team helpers for RIDER Genesis Ultra."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from coolprompt.utils.prompt_templates.rider_templates import (
    RIDER_CONTRACT_PROMPT,
    RIDER_SAFE_CLASSIFICATION_PROMPT_EN,
    RIDER_SAFE_CLASSIFICATION_PROMPT_RU,
    RIDER_SAFE_CODE_PROMPT_EN,
    RIDER_SAFE_CODE_PROMPT_RU,
    RIDER_SAFE_GENERIC_PROMPT_EN,
    RIDER_SAFE_GENERIC_PROMPT_RU,
    RIDER_SAFE_TRANSLATION_PROMPT_EN,
    RIDER_SAFE_TRANSLATION_PROMPT_RU,
)
from .schemas import (
    _JudgeScoreSchema,
    _PromptContractSchema,
    _RedTeamAdversarialSchema,
    _SyntheticTestsSchema,
)

logger = logging.getLogger(__name__)
_CONTRACT_PROMPT = RIDER_CONTRACT_PROMPT


class RiderSyntheticEvalMixin:
    """Synthetic evaluation and adversarial audit helpers."""

    def _bilingual_adversarial(self, prompt: str, lang: str) -> Optional[str]:
        """Ultra-only: English critique for non-EN prompts to surface universal issues."""
        meta = self._BILINGUAL_ADVERSARIAL_PROMPT.format(prompt=prompt, lang=lang)
        try:
            resp = self._generate(
                prompt=meta, role="critic",
                temperature=0.2, max_tokens=400,
            )
            return resp.strip() if resp and resp.strip() else None
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════════
    # v4 Ultra+ methods
    # ══════════════════════════════════════════════════════════════════════

    def _fallback_synthetic_tests(self, count: int = 3) -> List[str]:
        """Deterministic NEXUS-lite cases when the planner model does not return JSON."""
        archetype = str(self._contract.get('task_archetype') or 'other').lower()
        output_format = self._contract.get('output_format_anchor') or 'requested output'
        domain = self._contract.get('domain') or 'general domain'
        rules = "; ".join(str(x) for x in (self._contract.get('domain_rules') or [])[:3])
        rules = rules or "follow the source task contract"
        contract_text = " ".join(
            str(x)
            for x in (
                list(self._contract.get('must_preserve') or [])
                + list(self._contract.get('domain_rules') or [])
                + list(self._contract.get('failure_modes') or [])
                + list(self._contract.get('quality_dimensions') or [])
            )
        ).lower()
        has_geo_level_rule = any(term in contract_text for term in (
            'russia', 'ukraine', 'russian-side', 'россия', 'украина', 'геополит',
        ))
        has_exclusive_output_rule = any(term in contract_text for term in (
            'exclusive', 'only allowed output', 'output only', 'json only',
            'единственный', 'только json', 'schema drift',
        ))

        if archetype in {'creative_writing', 'analytical_essay', 'persuasion', 'brainstorming', 'other'}:
            tests = [
                f"Input: execute the prompt for a skeptical educated reader in {domain}. Expected invariant checks: distinctive angle, concrete details, clear structure, anti-generic wording, output format = {output_format}.",
                "Input: execute the prompt while avoiding cliches and generic school-essay framing. Expected invariant checks: named thesis, specific evidence/details, no padded meta-commentary.",
                "Input: execute the prompt for a publication-quality result. Expected invariant checks: coherent arc, vivid examples, quality bar satisfied, no irrelevant constraints.",
            ]
        elif archetype in {'debugging', 'code_generation', 'code_review'}:
            tests = [
                f"Input: solve the original code task exactly. Expected invariant checks: preserve identifiers/API, minimal patch, tests included, runtime reasoning correct, output format = {output_format}.",
                "Input: handle the edge case that caused the reported failure. Expected invariant checks: regression test covers the failure and no unrelated refactor is introduced.",
                "Input: explain verification briefly. Expected invariant checks: concrete commands or test cases, no invented dependencies, no renamed fields.",
            ]
        elif archetype == 'classification':
            tests = [
                f"Input: benign placeholder quote with no violation. Expected invariant checks: valid JSON, exact fields, Level 0 stays Level 0, rules: {rules}.",
                f"Input: borderline placeholder quote with ambiguous risk marker. Expected invariant checks: quote-grounded category decision, no invented law/category, schema = {output_format}.",
                "Input: high-risk placeholder quote that must not be missed. Expected invariant checks: false-negative resistance, correct level calibration, no prose outside JSON.",
            ]
            if has_geo_level_rule:
                tests.insert(
                    1,
                    "Input: quote mentions Russia/Ukraine/Russian side in a borderline context. Expected invariant checks: apply the source escalation rule exactly, keep the Level 0 exception, and cap the result at Level 3.",
                )
            if has_exclusive_output_rule:
                tests.insert(
                    0,
                    f"Input: any valid classification case. Expected invariant checks: the only output is {output_format}; no explanations, helper prose, markdown fences, or alternate schemas.",
                )
        elif archetype == 'translation':
            tests = [
                f"Input: short literary paragraph plus glossary entry. Expected invariant checks: glossary compliance, authorial rhythm, no extra commentary, output format = {output_format}.",
                "Input: intentionally rough source sentence with repetition and odd syntax. Expected invariant checks: preserve author texture rather than smoothing into generic Russian.",
                "Input: source paragraph with names/placeholders/tags. Expected invariant checks: placeholders and XML tags preserved exactly.",
            ]
        elif archetype == 'data_extraction':
            tests = [
                f"Input: noisy OCR/table-of-contents snippet. Expected invariant checks: verbatim extraction, no paraphrase, no extra commentary, output format = {output_format}.",
                "Input: partially unreadable line with page numbers and tags. Expected invariant checks: preserve unreadable markers and structure exactly.",
                "Input: mixed headings and list items. Expected invariant checks: completeness, original order, no invented text.",
            ]
        else:
            tests = [
                f"Input: straightforward case in {domain}. Expected invariant checks: fulfills task, preserves constraints, output format = {output_format}.",
                "Input: ambiguous edge case. Expected invariant checks: states assumptions only if allowed and does not drop required fields.",
                "Input: complex case with multiple requirements. Expected invariant checks: complete, structured, and no irrelevant additions.",
            ]
        return tests[:count]

    def _generate_synthetic_tests(self, candidate_prompt: str, count: int = 3) -> List[str]:
        """v4: Generate safe synthetic test inputs via Sonnet. Returns up to `count` inputs."""
        count = self._hyper_int("num_samples", count, min_value=1, max_value=20)
        if not self._contract:
            return []
        meta = self._SYNTHETIC_TEST_PROMPT.format(
            archetype=self._contract.get('task_archetype', 'other'),
            domain=self._contract.get('domain', 'general'),
            audience=self._contract.get('audience', 'general'),
            output_format=self._contract.get('output_format_anchor', 'free-form'),
            domain_rules="\n".join(f"- {x}" for x in (self._contract.get('domain_rules') or ['none'])),
            quality_dimensions="\n".join(f"- {x}" for x in (self._contract.get('quality_dimensions') or ['general task fulfillment'])),
            prompt=(candidate_prompt or '')[:2500],
            count=count,
        )
        obj = self._generate_structured(
            prompt=meta,
            schema=_SyntheticTestsSchema,
            role="planner",
            temperature=0.5,
            max_tokens=1500,
            allowed_starts=("[", "{"),
            max_retries=2,
        )
        if obj is None:
            result = self._fallback_synthetic_tests(count)
            self._synthetic_tests = result
            return result
        clean = list(getattr(obj, "tests", []) or [])
        result = clean[:count] or self._fallback_synthetic_tests(count)
        self._synthetic_tests = result
        return result

    def _evaluate_candidate_on_tests(
        self, candidate: str, tests: List[str]
    ) -> Tuple[float, List[Tuple[str, int]]]:
        """v4.3: Execute candidate on each synthetic test in PARALLEL with ECHO (prompt
        repeated 2x — RIDER ECHO protocol, ~67% eval-quality bump on non-reasoning models),
        then score in PARALLEL. Uses _generate with role-chain refusal fallback.

        v4.3.2: bloat guard — penalize candidates that grew >2.5x vs original (judge
        consistently flags over-engineered monsters as worse than tighter versions).
        """
        if not tests:
            return 0.0, []

        def _run(test_input: str) -> Tuple[str, int]:
            # v4.3: ECHO — repeat the candidate prompt twice to eliminate causal
            # attention asymmetry (Google arxiv.org/pdf/2512.14982, +67% on non-reasoning).
            exec_prompt = f"{candidate}\n\n{candidate}\n\nInput:\n{test_input}"
            response = self._generate(
                prompt=exec_prompt, role="worker", temperature=0.3,
                max_tokens=self._adaptive_max_tokens(test_input, min_tokens=500, ceiling=3000),
            )
            meta = self._EVAL_RESPONSE_PROMPT.format(
                archetype=self._contract.get('task_archetype', 'other'),
                output_format=self._contract.get('output_format_anchor', 'free-form'),
                quality_dimensions="\n".join(
                    f"- {x}" for x in (self._contract.get('quality_dimensions') or ['general task fulfillment'])
                ),
                candidate=(candidate or '')[:1500],
                test_input=(test_input or '')[:800],
                response=(response or '')[:2000],
            )
            score_obj = self._generate_structured(
                prompt=meta,
                schema=_JudgeScoreSchema,
                role="judge",
                temperature=0.0,
                max_tokens=64,
                allowed_starts=("{",),
                max_retries=1,
            )
            return (test_input[:60], int(score_obj.score) if score_obj else 5)

        workers = max(1, min(8, len(tests)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run, t) for t in tests]
            results = [f.result() for f in futures]

        details = results
        scores = [d[1] / 10.0 for d in details]
        base = (sum(scores) / len(scores)) if scores else 0.0

        # Synthetic execution can over-credit a raw original because a strong
        # worker model often infers missing structure by itself. For auto-prompting
        # fitness, the prompt must carry the contract explicitly.
        original_prompt = getattr(self, "_original_prompt", "") or ""
        if candidate.strip() == original_prompt.strip() and self._is_underoptimized_prompt(original_prompt):
            base = min(base, 0.45)

        # Bloat guard: penalize over-engineered monsters relative to _bloat_budget().
        # Short originals: no penalty until candidate exceeds the absolute floor
        # (so ultra-quality expansions are allowed to shine). Long originals:
        # penalty kicks in above 3x — we DO want to curb 10-15x runaways.
        orig = getattr(self, "_original_prompt_len", 0)
        if orig > 0:
            budget = self._bloat_budget(orig)
            if len(candidate) > budget:
                # Excess over budget, normalized by budget itself.
                # 1.0x over budget → 0.08 penalty; 2.5x over → 0.2 (cap).
                excess = (len(candidate) - budget) / max(1, budget)
                penalty = min(0.2, 0.08 * excess)
                base = max(0.0, base - penalty)
        return base, details

    def _external_eval_subset(self) -> Tuple[List[str], List[Any]]:
        context = getattr(self, "_external_eval_context", None) or {}
        dataset = list(context.get("val_dataset") or [])
        targets = list(context.get("val_targets") or [])
        size = min(len(dataset), len(targets))
        if size <= 0:
            return [], []
        limit = context.get("max_examples")
        if limit is None:
            return dataset[:size], targets[:size]
        try:
            limit_int = max(1, int(limit))
        except (TypeError, ValueError):
            return dataset[:size], targets[:size]
        if size <= limit_int:
            return dataset[:size], targets[:size]
        seed = context.get("seed")
        import random
        indices = sorted(random.Random(seed).sample(range(size), limit_int))
        return [dataset[i] for i in indices], [targets[i] for i in indices]

    @staticmethod
    def _normalize_external_score(value: Any) -> float:
        if isinstance(value, tuple):
            value = value[0]
        aggregate = getattr(value, "aggregate_score", value)
        try:
            score = float(aggregate)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    def _rank_by_external_eval(
        self, candidates: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], float]:
        context = getattr(self, "_external_eval_context", None) or {}
        evaluator = context.get("evaluator")
        if evaluator is None:
            return {}
        dataset, targets = self._external_eval_subset()
        if not dataset or not targets:
            return {}
        template = context.get("template")
        scores: Dict[Tuple[str, str], float] = {}
        for candidate in candidates:
            name, text = candidate
            try:
                raw_score = evaluator.evaluate(
                    prompt=text,
                    dataset=dataset,
                    targets=targets,
                    template=template,
                )
            except Exception as exc:
                logger.debug("External eval failed for %s: %s", name, exc)
                continue
            scores[candidate] = self._normalize_external_score(raw_score)
        if scores:
            self._external_rankings.append({
                "num_examples": len(dataset),
                "scores": [
                    {"name": cand[0], "score": score}
                    for cand, score in sorted(
                        scores.items(), key=lambda item: -item[1]
                    )
                ],
            })
        return scores

    def _rank_by_synthetic_eval(
        self, candidates: List[Tuple[str, str]], tests: List[str]
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Score candidates with synthetic tests and optional CoolPrompt val data."""
        if not tests or not candidates:
            return [(c, 0.0) for c in candidates]

        def _rank_one(cand: Tuple[str, str]) -> Tuple[Tuple[str, str], float]:
            avg, _details = self._evaluate_candidate_on_tests(cand[1], tests)
            return (cand, avg)

        workers = max(1, min(4, len(candidates)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_rank_one, c) for c in candidates]
            ranked = [f.result() for f in futures]

        ranked.sort(key=lambda x: -x[1])
        self._synthetic_rankings.append({
            'tests': list(tests),
            'scores': [{'name': cand[0], 'score': score} for cand, score in ranked],
        })

        external_scores = self._rank_by_external_eval(candidates)
        if not external_scores:
            return ranked

        weight = (getattr(self, "_external_eval_context", None) or {}).get(
            "weight", 0.7
        )
        try:
            external_weight = max(0.0, min(1.0, float(weight)))
        except (TypeError, ValueError):
            external_weight = 0.7
        synthetic_scores = dict(ranked)
        combined = []
        for cand in candidates:
            synth = synthetic_scores.get(cand, 0.0)
            external = external_scores.get(cand)
            score = (
                synth
                if external is None
                else (1.0 - external_weight) * synth + external_weight * external
            )
            combined.append((cand, score))
        combined.sort(key=lambda x: -x[1])
        self._external_rankings[-1]["combined_scores"] = [
            {"name": cand[0], "score": score} for cand, score in combined
        ]
        return combined

    def _original_margin_for_beam(self) -> float:
        """How much a safe non-original may trail original and still be worth using."""
        archetype = str(self._contract.get('task_archetype') or '').lower()
        dims = {str(x).lower() for x in (self._contract.get('quality_dimensions') or [])}
        if archetype == 'classification' or {'schema_compliance', 'level_calibration'} & dims:
            return 0.03
        if archetype == 'translation':
            return 0.08
        if archetype == 'data_extraction' or {'verbatim_fidelity', 'markup_compliance'} & dims:
            return 0.05
        if archetype in {'creative_writing', 'analytical_essay', 'persuasion', 'brainstorming'}:
            return 0.15
        return 0.12

    def _safe_non_original_candidates(
        self, ranked: List[Tuple[Tuple[str, str], float]], original: str,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Filter beam candidates that can be selected without final safety rollback."""
        budget = self._bloat_budget(max(1, len(original)))
        safe: List[Tuple[Tuple[str, str], float]] = []
        for cand, score in ranked:
            name, text = cand
            if name == "original":
                continue
            if len(text) > budget:
                continue
            violations, _ = self._check_preservation(original, text)
            if violations:
                continue
            if self._check_completeness_issues(text):
                continue
            safe.append((cand, score))
        return safe

    def _red_team_adversarial(self, prompt: str) -> Optional[Dict[str, Any]]:
        """v4 Ultra-only: adversary finds edge cases + fix directives."""
        meta = self._RED_TEAM_PROMPT.format(
            prompt=(prompt or '')[:3000],
            archetype=self._contract.get('task_archetype', 'other'),
            failure_modes=self._contract.get('failure_modes', []),
        )
        obj = self._generate_structured(
            prompt=meta,
            schema=_RedTeamAdversarialSchema,
            role="critic",
            temperature=0.4,
            max_tokens=800,
            allowed_starts=("{",),
            max_retries=2,
        )
        return self._schema_to_dict(obj) if obj is not None else None

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════
