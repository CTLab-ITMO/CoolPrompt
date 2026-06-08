"""Ultra run mode for RIDER Genesis Ultra."""

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


class RiderUltraMixin:
    """RIDER Ultra pipeline implementation."""

    def _run_ultra(self, prompt: str) -> str:
        """RIDER Ultra+ — 5-phase: ignition/fusion/crystal/validation/red-team+synth-beam k=3 (~180s, ~33 calls)."""
        t0 = time.time()
        self._setup_run(prompt, "ultra")

        self._contract = self._extract_contract(prompt)
        self._log(f"Contract: archetype={self._contract.get('task_archetype')}, "
                  f"preserve={self._contract.get('must_preserve')}, "
                  f"failure_modes={self._contract.get('failure_modes')}")

        # v4: bootstrap with lessons from prior runs on same archetype+domain.
        injected = self._prefetch_cached_lessons()
        if injected:
            self._log(f"[lesson cache] injected {injected} cross-prompt lessons "
                      f"(key={self._cache_key()})")

        # v4.4: adaptive pipeline depth per complexity tier.
        tier = self._complexity_tier()
        # Ultra mode quality floor: users who pick ULTRA explicitly asked for
        # ultra-quality output — never run a 3-strategy simple tier even on a
        # 27-char vague prompt. Short prompts need the full RIDER treatment
        # (4 strategies + crystallization) to blow up into a structured ultra
        # instruction; without the floor, "Write an essay about autumn" got a
        # 3-strategy pipeline that then choked on the bloat guard.
        if tier == 'simple':
            self._log("[adaptive pipeline] simple tier UPGRADED → medium "
                      "(ultra mode guarantees crystallization + 4 strategies)")
            tier = 'medium'
        cfg = self._apply_budget_overrides(self._ultra_pipeline_config(tier))
        self._log(f"[adaptive pipeline] complexity tier = {tier.upper()} "
                  f"(len={self._original_prompt_len} chars, "
                  f"archetype={self._contract.get('task_archetype')})")
        self._log(f"  pipeline: crystal={cfg['run_crystallization']}, "
                  f"validation={cfg['run_validation']}, "
                  f"red_team={cfg['run_red_team_harden']}, "
                  f"triple_merge={cfg['run_triple_merge']}, "
                  f"strategies={cfg['num_strategies']}")
        self._tier = tier
        self._cfg = cfg

        ign_t = self._phase_temperature('ignition')
        fus_t = self._phase_temperature('fusion')
        cry_t = self._phase_temperature('crystallization')
        val_t = self._phase_temperature('validation')

        # ── Phase 1: IGNITION (T=1.15) ──
        self._log(f"\n> Phase 1: IGNITION — adaptive strategies at T={ign_t:.2f}")
        all_strategies = self._adaptive_strategies("ultra")
        # v4.4: truncate strategy count per tier (simple=3, medium=4, complex=5).
        strategies = all_strategies[:cfg['num_strategies']]
        self._log(f"  strategies ({tier} tier, top-{cfg['num_strategies']}): {strategies}")
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
            return self._finalize_run(prompt, prompt, "ultra", t0)

        candidates = [("original", prompt)] + variants
        # v4.3: N-way batch ranking in ONE Sonnet call. v4.4: top-k scales per tier.
        k_phase1 = cfg['tournament_k_phase1']
        top = self._rank_batch(candidates, k=k_phase1)
        self._log(f"  -> Top {k_phase1} (N-way batch rank): {[n for n, _ in top]}")
        self._history.append({"generation": 1, "best_fitness": 0.0, "stage": "ignition"})

        # Semantic collapse check across top 3 — VORTEX opportunity.
        collapse_sim = max(
            (self._jaccard(a[1], b[1]) for i, a in enumerate(top) for b in top[i + 1:]),
            default=0.0,
        )
        use_vortex = collapse_sim >= 0.80
        if use_vortex:
            self._log(f"  [collapse detected] pairwise Jaccard={collapse_sim:.2f} -> swap mutation for VORTEX")

        # ── Phase 2: FUSION (T=0.85) — audit + refine + mutate ALL IN PARALLEL ──
        self._log(f"\n> Phase 2: FUSION — parallel audit+refine+mutate at T={fus_t:.2f}")
        with ThreadPoolExecutor(max_workers=6) as pool:
            audit_f = pool.submit(self._constitutional_audit, top[0][1])
            weaknesses = audit_f.result()
            self._log(f"  audit: {weaknesses[:160]}...")

            refine_futures = {
                pool.submit(self._refine, t, weaknesses, fus_t): n for n, t in top
            }

            if use_vortex:
                collapsed_preview = "\n---\n".join(t[1][:220] for t in top)
                vortex_f = pool.submit(
                    self._apply_strategy, top[0][1], 'vortex',
                    {'collapsed_preview': collapsed_preview}, fus_t,
                )
                mut2_f = pool.submit(
                    self._apply_strategy, top[0][1],
                    self._mutation_strategy('creative', 'techniques'), None, fus_t,
                )
            else:
                vortex_f = pool.submit(
                    self._apply_strategy, top[0][1],
                    self._mutation_strategy('creative', 'techniques'), None, fus_t,
                )
                mut2_f = pool.submit(
                    self._apply_strategy, top[0][1],
                    self._mutation_strategy('depth', 'analytical'), None, fus_t,
                )

            arena: List[Tuple[str, str]] = []
            for f in refine_futures:
                name = refine_futures[f]
                result = f.result()
                if result:
                    arena.append((f"{name}+", result))
            r1 = vortex_f.result()
            if r1:
                arena.append(("vortex" if use_vortex else "mutation-creative", r1))
            r2 = mut2_f.result()
            if r2:
                arena.append(("mutation-creative" if use_vortex else "mutation-depth", r2))

        for n, t in arena:
            self._log(f"  [{n}] {t[:70]}...")

        # v4.3: N-way batch ranking for top-2 (1 Sonnet call instead of 2 sequential).
        top2 = self._rank_batch(arena, k=2)
        self._log(f"  -> Top 2 (N-way batch rank): {[n for n, _ in top2]}")
        self._history.append({"generation": 2, "best_fitness": 0.0, "stage": "fusion"})

        # ── Phase 3: CRYSTALLIZATION (T=0.55 polish) — merge + dual polish ──
        # v4.4: only runs on medium/complex tier. On simple prompts, top2 already strong,
        # extra polish just adds noise (confirmed on NARKO3 benchmark).
        if cfg['run_crystallization'] and len(top2) >= 2:
            self._log(f"\n> Phase 3: CRYSTALLIZATION — merge + dual polish at T={cry_t:.2f}")
            merged = self._merge(top2[0][1], top2[1][1], cry_t)

            with ThreadPoolExecutor(max_workers=2) as pool:
                polish_f = pool.submit(self._apply_strategy, merged, 'techniques', None, cry_t)
                depth_f = pool.submit(self._apply_strategy, merged, 'depth', None, cry_t)
                polished = polish_f.result() or merged
                deepened = depth_f.result() or merged

            crystal = self._select_best(
                [("polished", polished), ("deepened", deepened), ("merged", merged)]
            )
            self._log(f"  Crystal winner: {crystal[0]}")
            self._history.append({"generation": 3, "best_fitness": 0.0, "stage": "crystallization"})
        else:
            self._log(f"\n> Phase 3 SKIPPED ({tier} tier) — using top2[0] as crystal")
            crystal = top2[0] if top2 else ("top1", prompt)

        # ── Phase 4: VALIDATION (T=0.3 near-deterministic) ──
        # v4.4: complex tier only. Extra audit+refine on simple prompts tightens
        # language without adding substance; on long prompts it catches real edge cases.
        if cfg['run_validation']:
            self._log(f"\n> Phase 4: VALIDATION — bilingual adversarial + audit IN PARALLEL at T={val_t:.2f}")
            lang = (self._contract.get('language') or '').lower()
            adversarial_directives: Optional[str] = None
            use_bilingual = bool(lang and lang not in ('en', 'english', 'unknown', ''))
            with ThreadPoolExecutor(max_workers=2) as pool:
                audit_f = pool.submit(self._constitutional_audit, crystal[1])
                bilingual_f = pool.submit(self._bilingual_adversarial, crystal[1], lang) if use_bilingual else None
                final_weaknesses = audit_f.result()
                if bilingual_f is not None:
                    adversarial_directives = bilingual_f.result()
                    self._log(f"  bilingual adversarial ({lang}): "
                              f"{adversarial_directives[:140] if adversarial_directives else '-'}...")
            combined_audit = final_weaknesses
            if adversarial_directives:
                combined_audit = f"{final_weaknesses}\n\nBilingual reviewer says:\n{adversarial_directives}"
            validated = self._refine(crystal[1], combined_audit, val_t)

            winner_name, winner_text = self._select_best(
                [("validated", validated), ("crystal", crystal[1])]
            )
            self._log(f"  Final: {winner_name}")
            self._history.append({"generation": 4, "best_fitness": 0.0, "stage": "validation"})
        else:
            self._log(f"\n> Phase 4 SKIPPED ({tier} tier) — using crystal directly")
            winner_name, winner_text = crystal

        # ── Phase 5: SYNTHETIC BEAM k=3 (always on) + optional red-team hardening ──
        # v4.4: synth-eval beam ALWAYS runs (this is the real-fitness signal RIDER
        # depends on). Red-team hardening only on complex tier — on simple prompts
        # it adds rules the LLM can't reliably execute.
        self._log("\n> Phase 5: SYNTHETIC BEAM k=3" + (
            " + RED-TEAM hardening" if cfg['run_red_team_harden'] else
            " (red-team SKIPPED for " + tier + " tier)"
        ))

        if cfg['run_red_team_harden']:
            with ThreadPoolExecutor(max_workers=2) as pool:
                red_team_f = pool.submit(self._red_team_adversarial, winner_text)
                synth_tests_f = pool.submit(self._generate_synthetic_tests, winner_text, 5)
                red_team = red_team_f.result()
                speculative_tests = synth_tests_f.result() or []

            hardened = winner_text
            if red_team and isinstance(red_team, dict):
                edge_cases = red_team.get('edge_cases') or []
                fix_dirs = red_team.get('fix_directives') or []
                severity = red_team.get('severity', 'low')
                self._log(f"  [red-team] severity={severity}, "
                          f"edge_cases={len(edge_cases)}, fixes={len(fix_dirs)}")
                if severity in ('medium', 'high') and fix_dirs:
                    fix_brief = (
                        "RED-TEAM EDGE CASES:\n"
                        + "\n".join(f"- {ec}" for ec in edge_cases[:3])
                        + "\n\nFIXES TO APPLY:\n"
                        + "\n".join(f"- {fd}" for fd in fix_dirs[:3])
                    )
                    hardened = self._refine(winner_text, fix_brief, val_t)
                    self._log(f"  [red-team] hardened: {hardened[:80]}...")
                    synth_tests = self._generate_synthetic_tests(hardened, count=5)
                else:
                    synth_tests = speculative_tests
            else:
                synth_tests = speculative_tests
        else:
            # Simple/medium: skip red-team, just generate synth tests for beam fitness.
            hardened = winner_text
            synth_tests = self._generate_synthetic_tests(winner_text, count=5)
        self._log(f"  synthetic tests generated: {len(synth_tests)}")

        # v4.3: include original in beam to get REAL fitness delta.
        # v4.4: beam composition depends on which phases ran.
        final_pick = hardened
        if synth_tests:
            beam: List[Tuple[str, str]] = []
            if cfg['run_red_team_harden']:
                beam.append(("hardened", hardened))
            if cfg['run_validation']:
                beam.append(("validated", winner_text))
            beam.append(("crystal", crystal[1]))
            if top2:
                beam.append(("top2[0]", top2[0][1]))
            beam.append(("original", prompt))
            # Bloat filter: skip runaway variants. Uses _bloat_budget() so short
            # originals get a generous absolute floor (can reach ultra-quality)
            # while long originals are capped at 3x (no 10-15x runaways).
            orig_len = max(1, len(prompt))
            budget = self._bloat_budget(orig_len)
            filtered_beam: List[Tuple[str, str]] = []
            for name, txt in beam:
                if name == "original" or len(txt) <= budget:
                    filtered_beam.append((name, txt))
                else:
                    ratio = len(txt) / orig_len
                    self._log(
                        f"  [bloat-filter] skipping '{name}' "
                        f"({len(txt)} chars, ratio={ratio:.1f}x, budget={budget})"
                    )
            beam = filtered_beam or [("original", prompt)]
            seen: set = set()
            uniq_beam: List[Tuple[str, str]] = []
            for name, txt in beam:
                key = txt[:200]
                if key not in seen:
                    seen.add(key)
                    uniq_beam.append((name, txt))
            ranked = self._rank_by_synthetic_eval(uniq_beam, synth_tests)
            if ranked:
                for c, sc in ranked:
                    if c[0] == "original":
                        self._synth_orig_score = sc
                        break
                orig_len = getattr(self, "_original_prompt_len", 0) or len(prompt)
                safe_non_orig_ranked = self._safe_non_original_candidates(ranked, prompt)
                original_score = next((sc for c, sc in ranked if c[0] == "original"), 0.0)
                original_wins = ranked[0][0][0] == "original"
                margin = self._original_margin_for_beam()
                force_evolve = self._is_underoptimized_prompt(prompt)
                if original_wins and orig_len >= 500 and not force_evolve and (
                    not safe_non_orig_ranked or original_score - safe_non_orig_ranked[0][1] > margin
                ):
                    final_pick = prompt
                    self._synth_best_score = original_score
                    self._log(f"  [SAFETY] original wins beam {original_score:.0%}; "
                              f"best safe non-original gap="
                              f"{(original_score - safe_non_orig_ranked[0][1]) if safe_non_orig_ranked else 1.0:.2f} "
                              f"> margin={margin:.2f} — returning original")
                elif safe_non_orig_ranked:
                    if ranked[0][0][0] == "original":
                        self._log(f"  [NEAR-MISS OVERRIDE] original won synth "
                                  f"({original_score:.0%}) but "
                                  f"{'prompt is underoptimized' if force_evolve else 'safe non-original is within margin=' + f'{margin:.2f}'}; "
                                  f"selecting evolved prompt")
                    self._synth_best_score = safe_non_orig_ranked[0][1]
                    final_pick = safe_non_orig_ranked[0][0][1]
                    self._log(f"  beam winner: {safe_non_orig_ranked[0][0][0]} "
                              f"(score={safe_non_orig_ranked[0][1]:.0%} vs orig={self._synth_orig_score or 0:.0%})")
                else:
                    # Degenerate case: only original in beam (everything else filtered out).
                    final_pick = prompt
                    self._synth_best_score = ranked[0][1] if ranked else 0.0
                    self._log("  [SAFETY] no non-original candidates in beam — returning original")
                for cand, sc in ranked:
                    self._log(f"       - {cand[0]}: {sc:.0%}")

                # v4.3 TRIPLE-MERGE: if top-2 non-original candidates are close in score
                # (within 15%) AND neither loses to original, merge for super-synthesis.
                # v4.4: complex tier only — on simple prompts there's rarely material
                # content to merge, the super-merge just produces a longer duplicate.
                if (cfg['run_triple_merge']
                        and ranked[0][0][0] != "original"
                        and len(safe_non_orig_ranked) >= 2):
                    top_sc = safe_non_orig_ranked[0][1]
                    second_sc = safe_non_orig_ranked[1][1]
                    if top_sc - second_sc < 0.15 and top_sc >= (self._synth_orig_score or 0):
                        super_merged = self._merge(
                            safe_non_orig_ranked[0][0][1],
                            safe_non_orig_ranked[1][0][1],
                            cry_t,
                        )
                        self._log(f"  [triple-merge] close scores {top_sc:.0%}/{second_sc:.0%} -> super-merge attempt")
                        super_score, _ = self._evaluate_candidate_on_tests(super_merged, synth_tests[:3])
                        self._log(f"  [triple-merge] super score: {super_score:.0%}")
                        if super_score > top_sc:
                            final_pick = super_merged
                            self._synth_best_score = super_score
                            self._log(f"  [triple-merge] WINS — new best {super_score:.0%}")
        self._history.append({"generation": 5, "best_fitness": self._synth_best_score or 0.0,
                              "stage": "red_team_synthetic"})

        # Final safety: if final_pick exceeds _bloat_budget(), downgrade to the
        # smallest candidate still within budget. Uses the same adaptive policy
        # as the 3 other bloat layers (FLOOR=2500, RATIO=3.0) — short originals
        # get a generous absolute floor so ultra-quality expansion is allowed,
        # long originals stay within 3x to avoid 10-15x runaways.
        orig_len = max(1, len(prompt))
        budget = self._bloat_budget(orig_len)
        if len(final_pick) > budget:
            small_candidates: List[Tuple[str, str]] = []
            if cfg['run_validation']:
                small_candidates.append(("validated", winner_text))
            small_candidates.append(("crystal", crystal[1]))
            if top2:
                small_candidates.append(("top2[0]", top2[0][1]))
            small_ok = [c for c in small_candidates if len(c[1]) <= budget]
            if small_ok:
                chosen = min(small_ok, key=lambda c: len(c[1]))
                self._log(f"  [FINAL BLOAT GUARD] final={len(final_pick)} > "
                          f"budget={budget} (orig={orig_len}) → "
                          f"downgrade to {chosen[0]} ({len(chosen[1])} chars)")
                final_pick = chosen[1]
            else:
                # Every candidate is over budget — still prefer the shortest one
                # over reverting to the original. User explicitly wants blow-up
                # on short prompts, so returning the vague input is unacceptable.
                all_non_orig = [
                    ("hardened", hardened),
                    ("crystal", crystal[1]),
                ]
                if top2:
                    all_non_orig.append(("top2[0]", top2[0][1]))
                if cfg['run_validation']:
                    all_non_orig.append(("validated", winner_text))
                chosen = min(all_non_orig, key=lambda c: len(c[1]))
                self._log(f"  [FINAL BLOAT GUARD] all variants over budget={budget}, "
                          f"picking shortest non-original ({chosen[0]}, {len(chosen[1])} chars)")
                final_pick = chosen[1]

        return self._finalize_run(prompt, final_pick, "ultra", t0)
