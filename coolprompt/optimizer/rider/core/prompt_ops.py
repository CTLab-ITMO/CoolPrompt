"""Prompt mutation, ranking, merge, audit, and quality helpers for RIDER Genesis Ultra."""

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


class RiderPromptOpsMixin:
    def _apply_strategy(
        self, prompt: str, strategy: str,
        extra_vars: Optional[Dict[str, str]] = None,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Apply one improvement strategy. 1 call. Returns improved prompt or None on failure."""
        template = self._STRATEGY_PROMPTS.get(strategy)
        if template is None:
            return None
        vars_ = {
            'prompt': prompt,
            'contract_block': self._contract_block(),
            'preservation_block': self._preservation_block(),
            'forge_block': self._forge_block(strategy),
            'lessons_block': self._lessons_block(),
            'collapsed_preview': '',
        }
        if extra_vars:
            vars_.update(extra_vars)
        meta = template.format(**vars_)
        # Hard char budget so strategies don't balloon.
        orig_ref = getattr(self, "_original_prompt_len", 0) or len(prompt)
        char_budget = self._bloat_budget(orig_ref)
        meta += f"\n\nHARD LIMIT: the improved prompt must not exceed {char_budget} characters."
        try:
            resp = self._generate(
                prompt=meta, role=self._strategy_role(strategy), temperature=temperature,
                max_tokens=self._adaptive_max_tokens(prompt),
            )
            text = self._strip_wrappers(resp)
            if len(text.split()) < 15:
                return None
            # Update FORGE memory with this success.
            self._forge.setdefault(strategy, []).append(text)
            # Keep only top-3 by length proxy (longer usually = more detail); dedup.
            self._forge[strategy] = list(dict.fromkeys(self._forge[strategy]))[-3:]
            return text
        except Exception as exc:
            logger.debug(f"_apply_strategy({strategy}) failed: {exc}")
            return None

    @staticmethod
    def _strip_wrappers(text: str) -> str:
        if text is None:
            return ""
        text = text.strip()
        for tag in ('<prompt>', '</prompt>', '[PROMPT_START]', '[PROMPT_END]',
                    '<START>', '<END>', '```text', '```markdown', '```'):
            text = text.replace(tag, '')
        return text.strip().strip('"').strip("'").strip()

    def _select_best(
        self, candidates: List[Tuple[str, str]], collect_lesson: bool = True,
    ) -> Tuple[str, str]:
        """Pairwise select best + extract GENESIS-lite lesson. 1 Sonnet call."""
        if len(candidates) <= 1:
            return candidates[0]
        parts = []
        for i, (_, text) in enumerate(candidates, 1):
            parts.append(f"--- PROMPT {i} ---\n{text}")
        meta = self._COMPARE_PROMPT.format(
            contract_block=self._contract_block(),
            candidates="\n\n".join(parts),
        )
        try:
            resp = self._generate(
                prompt=meta, role="judge",
                temperature=0.0, max_tokens=180,
                allow_fallback=True,
            )
            winner_idx, why = self._parse_winner_why(resp, len(candidates))
            if winner_idx is not None:
                if collect_lesson and why:
                    self._lessons.append(why.strip())
                return candidates[winner_idx]
        except Exception as exc:
            logger.debug(f"_select_best failed: {exc}")
        for n, t in candidates:
            if n != "original":
                return (n, t)
        return candidates[0]

    def _rank_batch(
        self, candidates: List[Tuple[str, str]], k: int = 3,
        collect_lesson: bool = True,
    ) -> List[Tuple[str, str]]:
        """v4.3: rank N candidates in ONE Sonnet call (N-way tournament).

        Replaces 3 sequential _select_best calls when selecting top-k from N.
        Returns top-k (name, text) tuples in order.
        """
        if len(candidates) <= k:
            return list(candidates)
        parts = []
        for i, (_, text) in enumerate(candidates, 1):
            parts.append(f"--- PROMPT {i} ---\n{text}")
        meta = self._BATCH_RANK_PROMPT.format(
            contract_block=self._contract_block(),
            n=len(candidates),
            candidates="\n\n".join(parts),
        )
        try:
            resp = self._generate(
                prompt=meta, role="judge",
                temperature=0.0, max_tokens=240,
                allow_fallback=True,
            )
            ranked_idx, why_top = self._parse_ranked(resp, len(candidates))
            if ranked_idx and len(ranked_idx) >= k:
                if collect_lesson and why_top:
                    self._lessons.append(why_top.strip())
                return [candidates[i] for i in ranked_idx[:k]]
        except Exception as exc:
            logger.debug(f"_rank_batch failed: {exc}")
        # Fallback: sequential _select_best
        top: List[Tuple[str, str]] = []
        remaining = list(candidates)
        for _ in range(min(k, len(remaining))):
            w = self._select_best(remaining, collect_lesson=collect_lesson and not top)
            top.append(w)
            remaining = [(n, t) for n, t in remaining if t != w[1]]
        return top

    @staticmethod
    def _parse_ranked(resp: str, n_cands: int) -> Tuple[List[int], Optional[str]]:
        """Parse 'RANKED: 3,1,4,2\\nWHY_TOP: ...' response into 0-indexed idx list + why."""
        if not resp:
            return [], None
        m_r = re.search(r'RANKED\s*:\s*([\d,\s]+)', resp, re.IGNORECASE)
        idx_list: List[int] = []
        seen: set = set()
        if m_r:
            for part in re.findall(r'\d+', m_r.group(1)):
                idx = int(part) - 1
                if 0 <= idx < n_cands and idx not in seen:
                    idx_list.append(idx)
                    seen.add(idx)
        # If RANKED parse failed, try to pick all digits from resp.
        if not idx_list:
            for part in re.findall(r'\b(\d+)\b', resp):
                idx = int(part) - 1
                if 0 <= idx < n_cands and idx not in seen:
                    idx_list.append(idx)
                    seen.add(idx)
        m_y = re.search(r'WHY_TOP\s*:\s*(.+?)(?:\n|$)', resp, re.IGNORECASE | re.DOTALL)
        why = None
        if m_y:
            why = re.sub(r'\s+', ' ', m_y.group(1)).strip()
            if len(why) > 240:
                why = why[:240].rstrip() + "..."
        return idx_list, why

    @staticmethod
    def _parse_winner_why(resp: str, n_cands: int) -> Tuple[Optional[int], Optional[str]]:
        """Parse 'WINNER: X\nWHY: ...' response."""
        if not resp:
            return None, None
        winner_idx: Optional[int] = None
        why: Optional[str] = None
        m_w = re.search(r'WINNER\s*:\s*(\d+)', resp, re.IGNORECASE)
        if m_w:
            idx = int(m_w.group(1)) - 1
            if 0 <= idx < n_cands:
                winner_idx = idx
        if winner_idx is None:
            # Fallback: first bare digit in [1..N].
            for num in re.findall(r'\b(\d+)\b', resp):
                idx = int(num) - 1
                if 0 <= idx < n_cands:
                    winner_idx = idx
                    break
        m_y = re.search(r'WHY\s*:\s*(.+?)(?:\n|$)', resp, re.IGNORECASE | re.DOTALL)
        if m_y:
            why = re.sub(r'\s+', ' ', m_y.group(1)).strip()
            if len(why) > 240:
                why = why[:240].rstrip() + "..."
        return winner_idx, why

    def _merge(self, p1: str, p2: str, temperature: float = 0.5) -> str:
        """Merge two prompts. 1 call.

        Hard char budget via _bloat_budget() — short originals get absolute floor
        (can expand to ultra-quality), long originals capped at 3x (no runaway).
        """
        orig_ref = getattr(self, "_original_prompt_len", 0) or max(len(p1), len(p2))
        char_budget = self._bloat_budget(orig_ref)
        meta = self._MERGE_PROMPT.format(
            prompt_a=p1, prompt_b=p2,
            contract_block=self._contract_block(),
            preservation_block=self._preservation_block(),
            lessons_block=self._lessons_block(),
        )
        meta += f"\n\nHARD LIMIT: the merged prompt must not exceed {char_budget} characters."
        try:
            resp = self._generate(
                prompt=meta, role="worker", temperature=temperature,
                max_tokens=self._adaptive_max_tokens(p1, p2),
            )
            text = self._strip_wrappers(resp)
            return text if len(text.split()) >= 15 else p1
        except Exception:
            return p1

    def _constitutional_audit(self, prompt: str) -> str:
        """6-dim rubric audit. 1 Sonnet call. Returns TOP_WEAKNESSES block as directive text."""
        meta = self._CONSTITUTIONAL_AUDIT_PROMPT.format(
            prompt=prompt, contract_block=self._contract_block(),
        )
        try:
            resp = self._generate(
                prompt=meta, role="critic",
                temperature=0.2, max_tokens=700,
            )
            # Extract TOP_WEAKNESSES section if present; else return full response.
            m = re.search(r'TOP_WEAKNESSES\s*:\s*(.+)$', resp, re.IGNORECASE | re.DOTALL)
            directives = m.group(1).strip() if m else resp.strip()
            return directives or "Prompt needs more specificity and tighter constraints."
        except Exception:
            return "Prompt needs more specificity, clearer role, and explicit output format."

    def _refine(self, prompt: str, weaknesses: str, temperature: float = 0.5) -> str:
        """Refine based on audit directives. 1 call.

        Hard char budget via _bloat_budget() — short originals get absolute floor
        (can expand to ultra-quality), long originals capped at 3x (no runaway).
        """
        orig_ref = getattr(self, "_original_prompt_len", 0) or len(prompt)
        char_budget = self._bloat_budget(orig_ref)
        meta = self._REFINE_PROMPT.format(
            prompt=prompt, weaknesses=weaknesses,
            contract_block=self._contract_block(),
            preservation_block=self._preservation_block(),
            lessons_block=self._lessons_block(),
        )
        meta += f"\n\nHARD LIMIT: the refined prompt must not exceed {char_budget} characters."
        try:
            resp = self._generate(
                prompt=meta, role="worker", temperature=temperature,
                max_tokens=self._adaptive_max_tokens(prompt),
            )
            text = self._strip_wrappers(resp)
            return text if len(text.split()) >= 15 else prompt
        except Exception:
            return prompt

    def _estimate_quality(self, original: str, improved: str) -> Tuple[float, float]:
        """Rate original vs improved. 1 Sonnet call. Returns (orig, improved) scores in [0,1].

        v4.3: if a real synthetic-eval winner score was measured during this run
        (self._synth_best_score), use it directly as the improved-side fitness.
        """
        # v4.3: prefer real synthetic-eval score over rubric heuristic.
        synth = getattr(self, "_synth_best_score", None)
        synth_orig = getattr(self, "_synth_orig_score", None)
        if synth is not None and synth > 0 and synth_orig is not None and synth_orig > 0:
            return float(synth_orig), float(synth)
        meta = self._QUALITY_PROMPT.format(
            prompt_a=original, prompt_b=improved,
            contract_block=self._contract_block(),
        )
        try:
            resp = self._generate(
                prompt=meta, role="judge",
                temperature=0.0, max_tokens=30,
            )
            nums = re.findall(r'[AB]\s*:\s*(\d+)', resp)
            if len(nums) >= 2:
                a = min(10, max(1, int(nums[0]))) / 10.0
                b = min(10, max(1, int(nums[1]))) / 10.0
                return a, b
        except Exception:
            pass
        return 0.3, 0.7

    # ══════════════════════════════════════════════════════════════════════
    # Preservation validator (post-hoc, no LLM; repair is 1 conditional call)
    # ══════════════════════════════════════════════════════════════════════

    _CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
    _INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
    _PLACEHOLDER_RE = re.compile(r"\{\{[^{}\s][^{}]*\}\}|\{[A-Za-z_][\w\.]*\}")
    _URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
    _MD_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
    _JSON_FIELD_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:')
    _LEGAL_CITATION_RE = re.compile(
        r"(?:ст\.?\s*\d+(?:\.\d+)?|ФЗ-?\s*№?\s*\d+|№\s*\d+-ФЗ|КоАП\s+РФ|УК\s+РФ)",
        re.IGNORECASE,
    )
    _GENERIC_PREFIXES = (
        r"write\s+(?:a|an|the)?\s*",
        r"create\s+(?:a|an|the)?\s*",
        r"make\s+(?:a|an|the)?\s*",
        r"help\s+me\s+",
        r"напиши\s+",
        r"сделай\s+",
        r"создай\s+",
        r"помоги\s+",
        r"напиши\s+",
        r"сделай\s+",
        r"создай\s+",
        r"помоги\s+",
    )
