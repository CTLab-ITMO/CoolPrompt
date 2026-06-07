"""Operator memory and GENESIS lesson-cache helpers for RIDER Genesis Ultra."""

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


class RiderMemoryMixin:
    """Persistent lesson-cache helpers for RIDER runs."""

    @classmethod
    def _load_lesson_cache(cls) -> Dict[str, List[str]]:
        try:
            if os.path.exists(cls._LESSON_CACHE_PATH):
                with open(cls._LESSON_CACHE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {k: [str(x) for x in v][:cls._LESSON_CACHE_MAX_PER_KEY]
                            for k, v in data.items() if isinstance(v, list)}
        except Exception:
            pass
        return {}

    def _save_lesson_cache(self):
        try:
            os.makedirs(os.path.dirname(self._LESSON_CACHE_PATH), exist_ok=True)
        except Exception:
            pass
        try:
            with open(self._LESSON_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(self._lesson_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _cache_key(self) -> str:
        archetype = self._contract.get('task_archetype', 'other') or 'other'
        domain = self._contract.get('domain', 'general') or 'general'
        # Normalize to lowercase, first word of domain only (to merge related domains).
        d = str(domain).lower().split()[0] if str(domain).strip() else 'general'
        return f"{archetype}::{d}"

    def _prefetch_cached_lessons(self) -> int:
        """Inject cached lessons from prior runs for the same archetype+domain.
        Standard/Ultra use this; light/blitz don't (by design)."""
        if self._mode not in ('standard', 'ultra'):
            return 0
        key = self._cache_key()
        cached = self._lesson_cache.get(key) or []
        if not cached:
            return 0
        # Dedup against current run lessons.
        existing = {str(x)[:120] for x in self._lessons}
        injected = 0
        for lsn in cached[:8]:
            if str(lsn)[:120] not in existing:
                self._lessons.append(lsn)
                existing.add(str(lsn)[:120])
                injected += 1
        return injected

    def _persist_new_lessons(self):
        """After run, merge this run's lessons into the disk cache (standard/ultra only)."""
        if self._mode not in ('standard', 'ultra'):
            return
        if not self._lessons:
            return
        key = self._cache_key()
        existing = list(self._lesson_cache.get(key) or [])
        existing_set = {str(x)[:120] for x in existing}
        for lsn in self._lessons:
            sig = str(lsn)[:120]
            if sig not in existing_set:
                existing.append(lsn)
                existing_set.add(sig)
        # Keep only the most recent N per key.
        self._lesson_cache[key] = existing[-self._LESSON_CACHE_MAX_PER_KEY:]
        self._save_lesson_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Contract extraction + adaptive routing
    # ══════════════════════════════════════════════════════════════════════

    _VALID_STRATEGIES = ('structural', 'analytical', 'creative', 'depth', 'techniques')

    _DEFAULT_CONTRACT: Dict[str, Any] = {
        'task_archetype': 'other',
        'language': 'unknown',
        'domain': 'general',
        'audience': 'general audience',
        'output_format_anchor': 'free-form',
        'must_preserve': [],
        'failure_modes': [],
        'recommended_strategies': ['structural', 'analytical', 'creative', 'depth', 'techniques'],
        'avoid_strategies': [],
        'domain_rules': [],
        'quality_dimensions': [],
    }

    _MODERATION_HINTS = (
        'moderation', 'censor', 'classification', 'json array', 'risk', 'level',
        'модерац', 'цензор', 'классифик', 'запрещ', 'нарушен', 'уровень',
        'экстремизм', 'терроризм', 'наркотик', 'пропаганд', 'дискредитац',
        'госбезопас', 'ненавист', 'вражд', 'нацизм', 'порнограф',
        'модерац', 'цензор', 'классифик', 'запрещ', 'нарушен', 'уровень',
        'экстремизм', 'терроризм', 'наркотик', 'пропаганд', 'дискредитац',
        'госбезопас', 'ненавист', 'вражд', 'нацизм', 'порнограф',
    )
    _TRANSLATION_HINTS = (
        'translation', 'translator', 'translate', '<translation>', '<dictionary>',
        'перевод', 'переводчик', '<перевод>', 'словар',
        'оригинал', 'русский текст', 'английского на русский',
        'перевод', 'переводчик', '<перевод>', '<dictionary>', 'словар',
        'оригинал', 'русский текст', 'английского на русский',
    )
    _GEO_RISK_HINTS = (
        'россия', 'росси', 'украина', 'украин', 'рф', 'российск', 'украинск', 'вооруженных сил',
        'вооружённых сил', 'дискредитац', 'антигосударствен', 'госбезопас',
        'сво', 'армии',
        'россия', 'украина', 'рф', 'российск', 'украинск', 'вооруженных сил',
        'вооружённых сил', 'дискредитац', 'антигосударствен', 'госбезопас',
        'сво', 'армии',
    )

    def _forge_block(self, strategy: str) -> str:
        top = self._forge.get(strategy) or []
        if not top:
            return ""
        bullets = []
        for i, snippet in enumerate(top[:2], 1):
            short = snippet[:180].replace('\n', ' ')
            if len(snippet) > 180:
                short += "..."
            bullets.append(f"  [prev best {i}] {short}")
        return (
            f"Your best '{strategy}' outputs from earlier in this run:\n"
            + "\n".join(bullets)
            + "\nProduce a new prompt that is AT LEAST as strong as these, ideally stronger.\n\n"
        )

    def _lessons_block(self) -> str:
        if not self._lessons:
            return ""
        bullets = "\n".join(f"  - {lsn}" for lsn in self._lessons[-4:])
        return (
            "Lessons learned so far (why previous winners were strong — reuse these insights):\n"
            f"{bullets}\n\n"
        )

    # ══════════════════════════════════════════════════════════════════════
    # v4.3 — Refusal detection + role-chain fallback
    # ══════════════════════════════════════════════════════════════════════

    _REFUSAL_RE: Optional[Any] = None
