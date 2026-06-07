"""Light, Blitz, and Standard run modes plus lifecycle helpers for RIDER Genesis Ultra."""

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


class RiderRunModesMixin:
    """Shared setup and execution paths for RIDER run modes."""

    def _setup_run(self, prompt: str, mode: str):
        self._api_calls_start = self.llm_client.total_api_calls
        self._history = []
        self._final_prompt = prompt
        self._original_fitness = 0.0
        self._best_fitness = 0.0
        self._contract = {}
        self._lessons = []
        self._forge = {}
        self._mode = mode
        self._original_prompt = prompt or ""
        self._role_model_chains = self._build_role_model_chains(mode)
        self.model = self._role_model("worker")
        self._synthetic_tests = []
        self._synthetic_rankings = []
        self._llm_attempts = []
        # v4.3 real-quality tracking
        self._synth_best_score: Optional[float] = None
        self._synth_orig_score: Optional[float] = None
        self._phase_temps_used: List[str] = []
        # v4.3.2 bloat guard reference length
        self._original_prompt_len: int = len(prompt or "")
        self._log("=" * 60)
        self._log(f"RIDER Genesis [{mode.upper()}]")
        self._log("=" * 60)
        self._log("Models: " + " | ".join(
            f"{role}={self._role_model(role)}" for role in self._ROLES
        ))
        self._log(f"Original: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    def _finalize_run(self, prompt: str, result: str, mode: str, t0: float) -> str:
        # Preservation check — may add at most 1 call.
        result = self._validate_and_repair(prompt, result)
        final_violations, _ = self._check_preservation(prompt, result)
        completeness_issues = self._check_completeness_issues(result)
        if final_violations or completeness_issues:
            if completeness_issues:
                self._log(f"  [safety] completeness issues: {completeness_issues[:4]}")
            fallback = self._deterministic_safe_enhancement(prompt)
            fb_violations, _ = self._check_preservation(prompt, fallback)
            fb_completeness = self._check_completeness_issues(fallback)
            if not fb_violations and not fb_completeness and fallback.strip() != prompt.strip():
                self._log(
                    f"  [safety] preservation still has {len(final_violations)} violation(s); "
                    "using deterministic safe enhancement"
                )
                result = fallback
            else:
                self._log(
                    f"  [safety] preservation still has {len(final_violations)} violation(s); "
                    "returning original"
                )
                result = prompt

        orig_score, best_score = self._estimate_quality(prompt, result)
        self._original_fitness = orig_score
        self._best_fitness = best_score
        self._final_prompt = result

        # v4: persist accumulated lessons to cross-prompt cache.
        try:
            self._persist_new_lessons()
        except Exception:
            pass

        elapsed = time.time() - t0
        api_calls = self.llm_client.total_api_calls - self._api_calls_start

        self._log(f"\n{'='*60}")
        self._log(f"RIDER Genesis [{mode.upper()}] — Results")
        self._log(f"{'='*60}")
        self._log(f"Contract: archetype={self._contract.get('task_archetype')}, "
                  f"lang={self._contract.get('language')}, "
                  f"strategies={self._contract.get('recommended_strategies')}")
        self._log(f"Quality: {orig_score:.0%} -> {best_score:.0%} ({self.improvement:+.1f}%)")
        self._log(f"Time: {elapsed:.1f}s | API calls: {api_calls}")
        self._log(f"Prompt: {result[:200]}...")
        self._log(f"{'='*60}")
        return result

    # ══════════════════════════════════════════════════════════════════════
    # Mode runners
    # ══════════════════════════════════════════════════════════════════════

    def run_light(self, prompt: str) -> str:
        """RIDER Light — fast optimization (~15-35s, ~5 calls).

        1. Contract extraction (1 Sonnet)
        2. 2 adaptive strategies IN PARALLEL at IGNITION temperature (2 working)
        3. Select best + WHY lesson (1 Sonnet)
        4. Quality estimate (1 Sonnet)

        v4.3: strategies run at IGNITION T=1.15 (exploration).
        """
        t0 = time.time()
        self._setup_run(prompt, "light")

        self._contract = self._extract_contract(prompt)
        self._log(f"Contract: archetype={self._contract.get('task_archetype')}, "
                  f"preserve={self._contract.get('must_preserve')}")

        strategies = self._adaptive_strategies("light")
        self._log(f"Strategies (adaptive, IGNITION T={self._phase_temperature('ignition'):.2f}): {strategies}")
        self._phase_temps_used.append(f"ignition={self._phase_temperature('ignition'):.2f}")
        ign_t = self._phase_temperature('ignition')
        with ThreadPoolExecutor(max_workers=max(1, len(strategies))) as pool:
            futures = {
                pool.submit(self._apply_strategy, prompt, s, None, ign_t): s
                for s in strategies
            }
            variants = []
            for f in futures:
                name = futures[f]
                result = f.result()
                if result:
                    variants.append((name, result))
                    self._log(f"  [{name}] {result[:80]}...")

        if not variants:
            return self._finalize_run(prompt, prompt, "light", t0)

        candidates = [("original", prompt)] + variants
        best_name, best_text = self._select_best(candidates)
        self._log(f"Winner: {best_name}")

        if best_name == "original" and variants:
            best_text = variants[0][1]

        # Light still gets a real independent critic pass: cheap/short, but it
        # prevents the fast mode from being a raw one-shot rewrite.
        if best_text != prompt:
            weaknesses = self._constitutional_audit(best_text)
            best_text = self._refine(best_text, weaknesses, self._phase_temperature('fusion'))
            self._log(f"Light critic-refine: {best_text[:80]}...")

        self._history.append({"generation": 1, "best_fitness": 0.0})
        return self._finalize_run(prompt, best_text, "light", t0)

    def _run_blitz(self, prompt: str) -> str:
        """RIDER Blitz — PARALLEL multi-strategy + diversity-aware merge + fusion refine (~45-120s).

        v4.3: IGNITION strategies at T=1.15, FUSION merge+refine at T=0.85.
        """
        t0 = time.time()
        self._setup_run(prompt, "blitz")

        self._contract = self._extract_contract(prompt)
        self._log(f"Contract: archetype={self._contract.get('task_archetype')}, "
                  f"preserve={self._contract.get('must_preserve')}")

        strategies = self._adaptive_strategies("blitz")
        ign_t = self._phase_temperature('ignition')
        fus_t = self._phase_temperature('fusion')
        self._log(f"Strategies (IGNITION T={ign_t:.2f}): {strategies}")
        with ThreadPoolExecutor(max_workers=max(1, len(strategies))) as pool:
            futures = {
                pool.submit(self._apply_strategy, prompt, s, None, ign_t): s
                for s in strategies
            }
            variants = []
            for f in futures:
                name = futures[f]
                result = f.result()
                if result:
                    variants.append((name, result))
                    self._log(f"  [{name}] {result[:70]}...")

        if not variants:
            return self._finalize_run(prompt, prompt, "blitz", t0)

        candidates = [("original", prompt)] + variants
        top1, partner, collapsed = self._pick_diverse_pair(candidates)
        if top1 is None:
            return self._finalize_run(prompt, prompt, "blitz", t0)
        if partner is None:
            partner = top1
        self._log(f"Top: {top1[0]} | Partner (diversity-aware): {partner[0]} "
                  f"| collapsed={collapsed}")
        self._history.append({"generation": 1, "best_fitness": 0.0, "stage": "tournament"})

        # FUSION phase: merge + audit IN PARALLEL (audit can start from top1 while merge runs),
        # then refine merged with fusion-T.
        self._log(f"FUSION: parallel(audit on top1, merge top1+partner) at T={fus_t:.2f}")
        with ThreadPoolExecutor(max_workers=2) as pool:
            merge_f = pool.submit(self._merge, top1[1], partner[1], fus_t)
            audit_f = pool.submit(self._constitutional_audit, top1[1])
            merged = merge_f.result()
            weaknesses = audit_f.result()
        self._log(f"Merged: {merged[:80]}...")
        refined = self._refine(merged, weaknesses, fus_t)
        self._log(f"Refined: {refined[:80]}...")
        self._history.append({"generation": 2, "best_fitness": 0.0, "stage": "refine"})

        return self._finalize_run(prompt, refined, "blitz", t0)

    def _run_standard(self, prompt: str) -> str:
        """RIDER Standard — PHASE REACTOR 3-phase + v4 synthetic eval + beam k=2 (~120s, ~22 calls)."""
        t0 = time.time()
        self._setup_run(prompt, "standard")

        self._contract = self._extract_contract(prompt)
        self._log(f"Contract: archetype={self._contract.get('task_archetype')}, "
                  f"preserve={self._contract.get('must_preserve')}")

        # v4: bootstrap with lessons from prior runs on same archetype+domain.
        injected = self._prefetch_cached_lessons()
        if injected:
            self._log(f"[lesson cache] injected {injected} cross-prompt lessons "
                      f"(key={self._cache_key()})")

        ign_t = self._phase_temperature('ignition')
        fus_t = self._phase_temperature('fusion')
        cry_t = self._phase_temperature('crystallization')

        # v4.4: adaptive pipeline for standard too.
        tier = self._complexity_tier()
        self._log(f"[adaptive pipeline] complexity tier = {tier.upper()} "
                  f"(len={self._original_prompt_len} chars, "
                  f"archetype={self._contract.get('task_archetype')})")
        # Standard mode: simple tier skips Phase 3 crystal polish (FUSION enough);
        # medium/complex run all 3 phases.
        run_crystal = (tier != 'simple')

        # ── Phase 1: IGNITION (T=1.15 exploration) ──
        self._log(f"\n> Phase 1: IGNITION — adaptive strategies at T={ign_t:.2f}")
        all_strategies = self._adaptive_strategies("standard")
        # On simple, 3 strategies; medium/complex — full 5.
        strategies = all_strategies[:3] if tier == 'simple' else all_strategies
        self._log(f"  strategies ({tier} tier, {len(strategies)}): {strategies}")
        with ThreadPoolExecutor(max_workers=max(1, len(strategies))) as pool:
            futures = {
                pool.submit(self._apply_strategy, prompt, s, None, ign_t): s
                for s in strategies
            }
            variants = []
            for f in futures:
                name = futures[f]
                result = f.result()
                if result:
                    variants.append((name, result))
                    self._log(f"  [{name}] {result[:70]}...")

        if not variants:
            return self._finalize_run(prompt, prompt, "standard", t0)

        candidates = [("original", prompt)] + variants
        # v4.3: N-way batch ranking in ONE Sonnet call (was 3 sequential pairwise).
        top = self._rank_batch(candidates, k=3)
        self._log(f"  -> Top 3 (N-way batch rank): {[n for n, _ in top]}")
        self._history.append({"generation": 1, "best_fitness": 0.0, "stage": "ignition"})

        # ── Phase 2: FUSION (T=0.85 balanced) ──
        self._log(f"\n> Phase 2: FUSION — audit + refine top-3 IN PARALLEL at T={fus_t:.2f}")
        # Audit + refines of 3 top candidates can all run in parallel.
        with ThreadPoolExecutor(max_workers=4) as pool:
            audit_f = pool.submit(self._constitutional_audit, top[0][1])
            weaknesses = audit_f.result()
            self._log(f"  audit: {weaknesses[:160]}...")
            refine_futures = {
                pool.submit(self._refine, t, weaknesses, fus_t): n
                for n, t in top
            }
            refined: List[Tuple[str, str]] = []
            for f in refine_futures:
                name = refine_futures[f]
                result = f.result()
                refined.append((f"{name}+", result))
                self._log(f"  [{name}+] {result[:70]}...")

        top1, partner, collapsed = self._pick_diverse_pair(refined)
        self._log(f"  diverse pair: {top1[0] if top1 else None} + {partner[0] if partner else None} "
                  f"| collapsed={collapsed}")
        merged = self._merge(top1[1], partner[1], fus_t) if (top1 and partner and partner[1] != top1[1]) else (top1[1] if top1 else prompt)
        self._log(f"  Merged: {merged[:80]}...")
        self._history.append({"generation": 2, "best_fitness": 0.0, "stage": "fusion"})

        # ── Phase 3: CRYSTALLIZATION (T=0.55 polish) ──
        # v4.4: skipped on simple tier (FUSION result already well-polished,
        # extra audit+refine accumulates noise).
        if run_crystal:
            self._log(f"\n> Phase 3: CRYSTALLIZATION — adversarial polish at T={cry_t:.2f}")
            final_weaknesses = self._constitutional_audit(merged)
            polished = self._refine(merged, final_weaknesses, cry_t)
            self._log(f"  Polished: {polished[:80]}...")
            self._history.append({"generation": 3, "best_fitness": 0.0, "stage": "crystallization"})
        else:
            self._log("\n> Phase 3 SKIPPED (simple tier) — using merged as polished")
            polished = merged

        # ── Phase 4: SYNTHETIC EVAL — beam k=2 (ECHO-enabled eval) ALWAYS runs ──
        self._log("\n> Phase 4: SYNTHETIC TEST EVAL — beam k=2 with ECHO")
        synth_tests = self._generate_synthetic_tests(polished, count=3)
        self._log(f"  synthetic tests generated: {len(synth_tests)}")
        if synth_tests:
            # v4.3: include original in beam to measure real uplift as fitness.
            beam = [("polished", polished), ("merged", merged), ("original", prompt)]
            ranked = self._rank_by_synthetic_eval(beam, synth_tests)
            if ranked:
                self._synth_best_score = ranked[0][1]
                for c, sc in ranked:
                    if c[0] == "original":
                        self._synth_orig_score = sc
                        break
                orig_len = getattr(self, "_original_prompt_len", 0) or len(prompt)
                safe_non_orig = self._safe_non_original_candidates(ranked, prompt)
                original_score = next((sc for c, sc in ranked if c[0] == "original"), 0.0)
                original_wins = ranked[0][0][0] == "original"
                margin = self._original_margin_for_beam()
                force_evolve = self._is_underoptimized_prompt(prompt)
                if original_wins and orig_len >= 500 and not force_evolve and (
                    not safe_non_orig or original_score - safe_non_orig[0][1] > margin
                ):
                    polished = prompt
                    self._synth_best_score = original_score
                    self._log(f"  [SAFETY] original wins beam {original_score:.0%}; "
                              f"best safe non-original gap="
                              f"{(original_score - safe_non_orig[0][1]) if safe_non_orig else 1.0:.2f} "
                              f"> margin={margin:.2f} — returning original")
                elif safe_non_orig:
                    if ranked[0][0][0] == "original":
                        self._log(f"  [NEAR-MISS OVERRIDE] original won synth "
                                  f"({original_score:.0%}) but "
                                  f"{'prompt is underoptimized' if force_evolve else 'safe non-original is within margin=' + f'{margin:.2f}'}; "
                                  f"selecting evolved prompt")
                    polished = safe_non_orig[0][0][1]
                    self._synth_best_score = safe_non_orig[0][1]
                    self._log(f"  beam winner: {safe_non_orig[0][0][0]} "
                              f"(score={safe_non_orig[0][1]:.0%} vs orig={self._synth_orig_score or 0:.0%})")
                for cand, sc in ranked:
                    self._log(f"       - {cand[0]}: {sc:.0%}")
            self._history.append({"generation": 4, "best_fitness": self._synth_best_score or 0.0,
                                  "stage": "synthetic_eval"})

        return self._finalize_run(prompt, polished, "standard", t0)
