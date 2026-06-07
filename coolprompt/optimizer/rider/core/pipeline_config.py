"""Mode, phase, and budget configuration helpers for RIDER Genesis Ultra."""

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


class RiderPipelineConfigMixin:
    def _phase_temperature(self, phase: str, default: float = 0.7) -> float:
        return self._PHASE_T.get(phase, default)

    # v4.4: adaptive complexity detection. Multi-phase pipeline is overkill for
    # short classification prompts (every refine layer adds noise → bloat),
    # but essential for long generation prompts (each phase adds structural depth).
    _SIMPLE_ARCHETYPES = frozenset({
        'classification', 'data_extraction', 'extractive', 'qa',
        'yes_no', 'label', 'choice', 'binary',
    })
    _COMPLEX_ARCHETYPES = frozenset({
        'reasoning', 'synthesis', 'generation', 'analysis', 'planning',
        'writing', 'creative_writing', 'analytical_essay',
        'technical_explanation', 'code_generation', 'code_review', 'debugging',
        'summarization', 'translation', 'brainstorming', 'instruction_design',
        'persuasion', 'creative',
    })

    def _complexity_tier(self) -> str:
        """Return 'simple' | 'medium' | 'complex' based on prompt size and archetype.

        v4.4: used by ultra/standard to parametrize pipeline depth. Elite-prompt
        optimization is not one-size-fits-all — a 2K-char classification prompt
        needs tight polishing, a 15K-char reasoning prompt needs the full 5-phase
        beam.
        """
        orig_len = getattr(self, '_original_prompt_len', 0) or 0
        archetype = (self._contract.get('task_archetype') or 'other').lower()

        # Size tier.
        if orig_len < 3000:
            size_tier = 'simple'
        elif orig_len < 8000:
            size_tier = 'medium'
        else:
            size_tier = 'complex'

        # Archetype adjustment: simple archetypes downgrade one level, complex bump up.
        if archetype in self._SIMPLE_ARCHETYPES:
            # Large classifiers (>= 7K chars) still need full pipeline — empirically
            # validated on EXTREM1 (9.7K, classification): full 5-phase gave ideal
            # hierarchy ultra > blitz > standard > light.
            if orig_len >= 7000:
                return 'complex'
            if size_tier == 'medium':
                return 'medium'
            return 'simple'
        if archetype in self._COMPLEX_ARCHETYPES:
            if size_tier == 'simple':
                return 'medium'
            if size_tier == 'medium':
                return 'complex'
            return 'complex'
        return size_tier

    def _ultra_pipeline_config(self, tier: str) -> Dict[str, Any]:
        """v4.4: pipeline shape per complexity tier. Fewer phases on simple prompts
        prevents multi-phase bloat accumulation; all 5 phases fire on complex prompts
        where each phase provides measurable uplift (verified on EXTREM1 9.7K)."""
        return {
            'simple': {
                # 3 phases: IGNITION + FUSION + SYNTH-EVAL beam. No crystal/validation/red-team.
                'num_strategies': 3,
                'tournament_k_phase1': 2,
                'run_crystallization': False,
                'run_validation': False,
                'run_red_team_harden': False,
                'run_triple_merge': False,
                'beam_includes_original': True,
            },
            'medium': {
                # 4 phases: IGNITION + FUSION + CRYSTAL + SYNTH-EVAL beam. Skip validation & red-team hardening.
                'num_strategies': 4,
                'tournament_k_phase1': 3,
                'run_crystallization': True,
                'run_validation': False,
                'run_red_team_harden': False,
                'run_triple_merge': False,
                'beam_includes_original': True,
            },
            'complex': {
                # Full 5 phases + triple-merge. Only fires on long/reasoning/generation prompts.
                'num_strategies': 5,
                'tournament_k_phase1': 3,
                'run_crystallization': True,
                'run_validation': True,
                'run_red_team_harden': True,
                'run_triple_merge': True,
                'beam_includes_original': True,
            },
        }.get(tier, {
            'num_strategies': 5, 'tournament_k_phase1': 3,
            'run_crystallization': True, 'run_validation': True,
            'run_red_team_harden': True, 'run_triple_merge': True,
            'beam_includes_original': True,
        })

    # -----------------------------------------------------------------------
    # Adaptive bloat budget — single source of truth for all 4 bloat-guard
    # layers (strategy/merge/refine hard caps, synth-eval penalty, beam filter).
    # -----------------------------------------------------------------------
    _BLOAT_FLOOR = 4000     # absolute minimum budget — short originals (e.g. 27
                            # chars) must still be able to expand to a full
                            # ultra-quality prompt with role/steps/criteria/
                            # constraints/anti-patterns/examples. 4 KB matches
                            # the typical length of a well-structured ultra
                            # prompt (~500-700 words) so short-prompt blow-up
                            # isn't artificially clipped mid-section.
    _BLOAT_RATIO = 3.0      # multiplicative cap for long originals — a 10 KB
                            # prompt should not balloon past 30 KB (runs of
                            # 10-15x were observed pre-guard on merge pipelines).

    def _bloat_budget(self, orig_len: int) -> int:
        """Maximum allowed length (in characters) for an optimized prompt.

        Policy:
          * Short originals (< FLOOR/RATIO = ~833 chars) → FLOOR applies.
            A vague 27-char prompt is allowed to grow into a 2500-char
            structured instruction.
          * Long originals (>= FLOOR/RATIO) → RATIO × original applies.
            A 4 KB moderation prompt may expand to ~12 KB, a 10 KB
            rubricator to ~30 KB, but no further.

        Used by _apply_strategy, _merge, _refine (hard cap injected into
        the worker model's instructions), by _evaluate_candidate_on_tests
        (synth-eval linear penalty) and by the beam filter before final
        ranking. Keeping a single helper avoids the 4 layers drifting
        apart, which was the root cause of the "ultra returns original
        verbatim" regression on short English prompts.
        """
        archetype = str(self._contract.get('task_archetype') or '').lower()
        if (
            self._is_underoptimized_prompt(getattr(self, "_original_prompt", ""))
            and archetype in {'creative_writing', 'analytical_essay', 'persuasion', 'brainstorming', 'other'}
        ):
            # Short under-specified prompts need room to become real production
            # instructions. Keep long operational prompts strict; loosen only
            # for raw under-specified creative/general asks.
            return max(int(orig_len * 8.0), 6500)
        return max(int(orig_len * self._BLOAT_RATIO), self._BLOAT_FLOOR)

    @classmethod
    def _is_underoptimized_prompt(cls, text: str) -> bool:
        """Detect short/raw prompts where returning original is never impressive."""
        stripped = (text or '').strip()
        if not stripped:
            return False
        words = re.findall(r"\w+", stripped, re.UNICODE)
        if len(stripped) <= 180:
            return True
        if len(words) <= 35 and not re.search(r"\n|```|<[^>]+>|{[A-Za-z_]", stripped):
            return True
        lower = stripped.lower()
        return any(re.match(pat, lower) for pat in cls._GENERIC_PREFIXES)
    # Custom XML-like tags used as structural anchors (e.g. <шкала_нарушений>). Allows
    # Cyrillic tag names used in RU prompt engineering (moderation rubrics).
    # Matches opening/closing tag pair: <name>...</name>, name length ≥ 3 to avoid
    # catching generic HTML stubs like <p>.
    _XML_TAG_RE = re.compile(r"<([\w\u0400-\u04FF_]{3,})>[\s\S]*?</\1>")

    def _adaptive_max_tokens(self, *prompts: str, min_tokens: int = 4000, ceiling: int = 16000) -> int:
        """v4.3: Cyrillic-aware scaling so full rewrites don't get truncated, but
        also don't balloon 10x — cap at input*1.5 + 1024 headroom.

        Cyrillic text tokenizes ~2x worse than English in Anthropic/OpenAI tokenizers.
        Previous v4.2 defaults (min=2000) caused mid-sentence truncation on 4K+ char
        Russian prompts. v4.3.1 raised ceiling to 24K which let ultra balloon to 24K
        chars (7x original). v4.3.2: tighter cap — enough room for 50% growth + structure.
        """
        total_chars = 0
        cyr_chars = 0
        for p in prompts:
            if not p:
                continue
            total_chars += len(p)
            for ch in p:
                code = ord(ch.lower())
                if 0x0400 <= code <= 0x04FF:
                    cyr_chars += 1
        cyr_ratio = (cyr_chars / total_chars) if total_chars else 0.0
        if cyr_ratio > 0.25:
            tokens_per_char = 2.0
        elif cyr_ratio > 0.1:
            tokens_per_char = 1.2
        else:
            tokens_per_char = 0.3
        est_input_tokens = int(total_chars * tokens_per_char)
        # Output: input x1.2 (+20% growth allowance for structural additions) + 1024 headroom.
        budget = int(est_input_tokens * 1.2) + 1024
        return max(min_tokens, min(ceiling, budget))
