"""Preservation, repair, and diversity helpers for RIDER Genesis Ultra."""

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


class RiderPreservationMixin:
    def _extract_required_artifacts(self, original: str) -> Dict[str, List[str]]:
        """Detect artifacts in the original that should be preserved."""
        artifacts: Dict[str, List[str]] = {}
        preserve_items = [str(x) for x in (self._contract.get('must_preserve') or [])]
        flags = {x.lower() for x in preserve_items}
        # Auto-detect common artifacts regardless of contract flags.
        code_fences = self._CODE_FENCE_RE.findall(original)
        if code_fences:
            artifacts['code_blocks'] = code_fences
        inline_code = self._INLINE_CODE_RE.findall(original)
        if inline_code:
            artifacts['inline_code'] = inline_code
        placeholders = self._PLACEHOLDER_RE.findall(original)
        if placeholders:
            artifacts['placeholders'] = placeholders
        urls = self._URL_RE.findall(original)
        if urls:
            artifacts['urls'] = urls
        # Markdown tables — any markdown/table mention in contract or auto-detect.
        if any('table' in f or 'markdown' in f for f in flags) or self._MD_TABLE_RE.search(original):
            lines = [ln for ln in original.splitlines() if self._MD_TABLE_RE.match(ln)]
            if lines:
                artifacts['markdown_table_lines'] = lines
        # Custom XML-like tags (e.g. <шкала_нарушений>, <scale>, etc.) — common anchor
        # in moderation / rubric prompts.
        xml_matches = [m.group(0) for m in self._XML_TAG_RE.finditer(original)]
        if xml_matches:
            artifacts['xml_anchors'] = xml_matches
            # Also require the opening and closing tag names to reappear.
            tag_names: List[str] = []
            for m in self._XML_TAG_RE.finditer(original):
                name = m.group(1)
                if name not in tag_names:
                    tag_names.append(f"<{name}>")
                    tag_names.append(f"</{name}>")
            if tag_names:
                artifacts['xml_tag_names'] = tag_names
        json_fields: List[str] = []
        for field in self._JSON_FIELD_RE.findall(original):
            if field not in json_fields:
                json_fields.append(field)
        if json_fields:
            artifacts['json_field_names'] = json_fields[:80]
        category_labels: List[str] = []
        for pat in (
            r'"name"\s*:\s*"([^"]+)"',
            r'"filter"\s*:\s*"([^"]+)"',
            r"\b(?:name|filter)\s*=\s*['\"]([^'\"]+)['\"]",
            r"\b[A-ZА-ЯЁ0-9]+_[A-ZА-ЯЁ0-9_]+\b",
        ):
            for m in re.finditer(pat, original):
                value = m.group(1) if m.groups() else m.group(0)
                if 2 < len(value) <= 80 and value not in category_labels:
                    category_labels.append(value)
        if category_labels:
            artifacts['category_labels'] = category_labels[:80]
        level_values = []
        for m in re.finditer(r'\b(?:level|уровень)[ \t]*(?:=|:|-)?[ \t]*([0-3])\b', original, re.IGNORECASE):
            value = m.group(1)
            if value not in level_values:
                level_values.append(value)
        if level_values:
            artifacts['level_values'] = level_values
        # Director-feedback invariants from real EKSMO moderation prompts:
        # the optimizer may improve wording, but it must not soften the machine
        # output contract, priority order, level caps, or exception semantics.
        policy_lines = [ln.strip() for ln in original.splitlines() if 4 <= len(ln.strip()) <= 500]
        exclusive_output_lines: List[str] = []
        level_policy_lines: List[str] = []
        priority_policy_lines: List[str] = []
        for line in policy_lines:
            low = line.lower()
            if any(term in low for term in (
                'единственный допустимый вывод',
                'единственный вывод',
                'только json',
                'строго json',
                'без пояснений',
                'без объяснений',
                'only allowed output',
                'output only',
                'json only',
                'no prose',
                'no explanations',
            )):
                if line not in exclusive_output_lines:
                    exclusive_output_lines.append(line)
            has_level_word = ('level' in low) or ('уров' in low)
            if has_level_word and any(term in low for term in (
                'не выше',
                'выше',
                'поднима',
                'повыш',
                'кроме',
                'исключ',
                'нулев',
                'level 0',
                'level 3',
                'cap',
                'maximum',
                'max ',
                'raise',
                'escalat',
                'except',
            )):
                if line not in level_policy_lines:
                    level_policy_lines.append(line)
            if any(term in low for term in ('приоритет', 'порядок', 'сначала', 'затем', 'priority', 'order')) and (
                has_level_word or 'категор' in low or 'category' in low or 'rule' in low
            ):
                if line not in priority_policy_lines:
                    priority_policy_lines.append(line)
        if exclusive_output_lines:
            artifacts['exclusive_output_rules'] = exclusive_output_lines[:12]
        if level_policy_lines:
            artifacts['level_escalation_rules'] = level_policy_lines[:20]
        if priority_policy_lines:
            artifacts['priority_policy_rules'] = priority_policy_lines[:12]
        legal_citations: List[str] = []
        for m in self._LEGAL_CITATION_RE.finditer(original):
            value = m.group(0)
            if value not in legal_citations:
                legal_citations.append(value)
        if legal_citations:
            artifacts['legal_citations'] = legal_citations[:40]
        geo_terms: List[str] = []
        for term in ('Россия', 'Украина', 'РФ', 'СВО', 'ВС РФ', 'Вооруженных сил', 'Вооружённых сил'):
            if term.lower() in original.lower():
                geo_terms.append(term)
        if geo_terms:
            artifacts['geo_escalation_terms'] = geo_terms
        # Literal strings from contract.must_preserve that look like concrete quotes /
        # markers (not generic categories). A simple heuristic: if the item appears
        # verbatim in the original, treat it as a hard preservation requirement.
        literal_hits: List[str] = []
        for item in preserve_items:
            item_clean = item.strip()
            if len(item_clean) < 3 or len(item_clean) > 200:
                continue
            if item_clean in original and item_clean not in literal_hits:
                literal_hits.append(item_clean)
        if literal_hits:
            artifacts['contract_literal_strings'] = literal_hits
        # v4.1 Enumerated structure preservation — the #1 failure mode for legal /
        # moderation rubrics. LLM judges prefer prose flow, but lawyers need
        # numbered criteria for citation in court.
        numbered_items = re.findall(r"(?m)^\s*\d+[\.\)]\s+\S", original)
        if len(numbered_items) >= 3:
            artifacts['enumerated_count'] = [str(len(numbered_items))]
        # Explicit level / step markers (case-insensitive, multi-language).
        # `[ \t]+` (no newlines) to avoid matching word-across-line artifacts.
        marker_patterns = [
            r"\b(?:Уровень|Ступень|Level|Step|ШАГ|Этап|Stage|Tier)[ \t]+\d+(?:\.\d+)?",
            # Category codes like "1.1.а", "2.3.б" (cyrillic or latin letter suffix).
            r"\b\d+\.\d+(?:\.[a-zа-я])?",
            # v4.3: Letter-based markers used in legal/moderation rubrics.
            # Matches "ПРИЗНАК A", "MARKER B", "ITEM C", "CRITERION D", "ПУНКТ А" etc.
            r"\b(?:ПРИЗНАК|Признак|MARKER|Marker|ITEM|Item|CRITERION|Criterion|ПУНКТ|Пункт|КРИТЕРИЙ|Критерий)[ \t]+[A-ZА-Я]\b",
        ]
        markers: List[str] = []
        for pat in marker_patterns:
            for m in re.finditer(pat, original, re.IGNORECASE):
                txt = m.group(0)
                if txt not in markers:
                    markers.append(txt)
        if markers:
            artifacts['enumeration_markers'] = markers[:50]  # cap for safety
        return artifacts

    def _check_preservation(self, original: str, improved: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Return (violations, required_items_by_kind)."""
        artifacts = self._extract_required_artifacts(original)
        violations: List[str] = []
        for kind, items in artifacts.items():
            if kind == 'enumerated_count':
                # Special: numbered list count must not drop by more than 40%.
                try:
                    expected_min = int(int(items[0]) * 0.6)
                except Exception:
                    continue
                actual = len(re.findall(r"(?m)^\s*\d+[\.\)]\s+\S", improved))
                if actual < expected_min:
                    violations.append(
                        f"enumerated_structure_loss: expected at least {expected_min} "
                        f"numbered items, found {actual} (original had {items[0]})"
                    )
                continue
            if kind == 'enumeration_markers':
                # All explicit level/step/category markers must remain present.
                missing = [m for m in items if m not in improved]
                if missing:
                    for mk in missing[:6]:
                        violations.append(f"missing enumeration_marker: {mk}")
                continue
            if kind == 'json_field_names':
                improved_fields = set(self._JSON_FIELD_RE.findall(improved))
                missing = [m for m in items if m not in improved_fields]
                if missing:
                    violations.append(f"missing json_field_names: {', '.join(missing[:12])}")
                continue
            if kind == 'category_labels':
                missing = [m for m in items if m not in improved]
                if missing:
                    violations.append(f"missing category_labels: {', '.join(missing[:8])}")
                continue
            if kind == 'level_values':
                improved_values = set(
                    m.group(1) for m in re.finditer(
                        r'\b(?:level|уровень)[ \t]*(?:=|:|-)?[ \t]*([0-3])\b',
                        improved,
                        re.IGNORECASE,
                    )
                )
                missing = [m for m in items if m not in improved_values]
                if missing:
                    violations.append(f"missing level_values: {', '.join(missing)}")
                continue
            if kind == 'legal_citations':
                missing = [m for m in items if m not in improved]
                if missing:
                    violations.append(f"missing legal_citations: {', '.join(missing[:6])}")
                continue
            if kind == 'geo_escalation_terms':
                missing = [m for m in items if m.lower() not in improved.lower()]
                if missing:
                    violations.append(f"missing geo_escalation_terms: {', '.join(missing[:6])}")
                continue
            for item in items:
                if item and item not in improved:
                    violations.append(f"missing {kind}: {item[:120]}")
        return violations, artifacts

    def _repair_preservation(
        self, original: str, improved: str, violations: List[str],
        required: Dict[str, List[str]],
    ) -> str:
        """One repair call — only fired when violations were detected."""
        if not violations:
            return improved
        flat_items: List[str] = []
        for kind, items in required.items():
            for it in items[:4]:
                flat_items.append(f"[{kind}] {it}")
        required_text = "\n".join(flat_items[:12]) or "(auto-detected artifacts above)"
        meta = self._PRESERVE_REPAIR_PROMPT.format(
            required_items=required_text,
            original=original, broken=improved,
            violations="\n".join(f"- {v}" for v in violations[:12]),
        )
        try:
            resp = self._generate(
                prompt=meta, role="worker", temperature=0.1,
                max_tokens=self._adaptive_max_tokens(original, improved),
            )
            text = self._strip_wrappers(resp)
            return text if len(text.split()) >= 15 else improved
        except Exception:
            return improved

    @staticmethod
    def _check_completeness_issues(text: str) -> List[str]:
        """Conservative truncation/completeness checks for generated prompts."""
        issues: List[str] = []
        if not text or not text.strip():
            return ["empty_prompt"]
        stripped = text.rstrip()
        if stripped.count("```") % 2:
            issues.append("unclosed_code_fence")

        tag_re = re.compile(r"</?([\w\u0400-\u04FF_]{3,})(?:\s[^>]*)?>")
        opens: Dict[str, int] = {}
        closes: Dict[str, int] = {}
        structural_opens: Dict[str, int] = {}
        structural_tag_names = {'перевод', 'translation', 'rules', 'final_check', 'шкала_нарушений'}
        for m in tag_re.finditer(stripped):
            raw = m.group(0)
            if raw.endswith("/>"):
                continue
            name = m.group(1)
            if raw.startswith("</"):
                closes[name] = closes.get(name, 0) + 1
            else:
                opens[name] = opens.get(name, 0) + 1
                line_start = stripped.rfind("\n", 0, m.start()) + 1
                line_end = stripped.find("\n", m.end())
                if line_end == -1:
                    line_end = len(stripped)
                line = stripped[line_start:line_end].strip()
                if line == raw or name in structural_tag_names:
                    structural_opens[name] = structural_opens.get(name, 0) + 1
        for name in opens:
            structural_count = structural_opens.get(name, 0)
            if (name in structural_tag_names or closes.get(name, 0) > 0) and structural_count and closes.get(name, 0) < structural_count:
                issues.append(f"unclosed_xml_tag:{name}")
                if len(issues) >= 4:
                    break

        last_line = next((ln.strip() for ln in reversed(stripped.splitlines()) if ln.strip()), "")
        last_word_match = re.search(r"([\w\u0400-\u04FF-]+)$", last_line)
        last_word = (last_word_match.group(1).lower() if last_word_match else "")
        dangling_words = {
            'if', 'when', 'while', 'because', 'and', 'or', 'with', 'for',
            'если', 'когда', 'пока', 'потому', 'что', 'как', 'и', 'или',
            'эпизод', 'упоминание', 'критер', 'критерий', 'раздел', 'пункт',
            'если', 'когда', 'пока', 'потому', 'что', 'как', 'и', 'или',
            'эпизод', 'упоминание', 'критер', 'критерий', 'раздел', 'пункт',
        }
        if re.match(r"^\s*#{1,6}\s+\S", last_line) and not re.search(r"[.!?:;…)]$", last_line):
            issues.append("ends_with_heading")
        if last_word in dangling_words or re.search(r"[,;({\[-]\s*$", last_line):
            issues.append("dangling_final_line")
        return issues

    def _validate_and_repair(self, original: str, improved: str) -> str:
        violations, required = self._check_preservation(original, improved)
        if not violations:
            return improved
        self._log(f"  [preservation] {len(violations)} violation(s), repairing")
        return self._repair_preservation(original, improved, violations, required)

    def _deterministic_safe_enhancement(self, original: str) -> str:
        """Non-LLM fallback that improves the wrapper while preserving source verbatim."""
        archetype = str(self._contract.get('task_archetype') or '').lower()
        lang = str(self._contract.get('language') or '').lower()
        is_ru = lang == 'ru' or bool(re.search(r'[\u0400-\u04FF]', original or ''))

        if archetype in {'debugging', 'code_generation', 'code_review'}:
            template = RIDER_SAFE_CODE_PROMPT_RU if is_ru else RIDER_SAFE_CODE_PROMPT_EN
            return template.format(original=original)

        if archetype == 'classification':
            template = (
                RIDER_SAFE_CLASSIFICATION_PROMPT_RU
                if is_ru else
                RIDER_SAFE_CLASSIFICATION_PROMPT_EN
            )
            return template.format(original=original)

        if archetype == 'translation':
            template = RIDER_SAFE_TRANSLATION_PROMPT_RU if is_ru else RIDER_SAFE_TRANSLATION_PROMPT_EN
            return template.format(original=original)

        if is_ru:
            return RIDER_SAFE_GENERIC_PROMPT_RU.format(original=original)
        return RIDER_SAFE_GENERIC_PROMPT_EN.format(original=original)

    # ══════════════════════════════════════════════════════════════════════
    # Diversity helpers (Jaccard, no embeddings)
    # ══════════════════════════════════════════════════════════════════════

    _WORD_RE = re.compile(r"\w+", re.UNICODE)
    _STOPWORDS_EN = {
        'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'with', 'as',
        'by', 'is', 'are', 'be', 'been', 'was', 'were', 'this', 'that', 'it', 'at',
        'from', 'you', 'your', 'we', 'our', 'they', 'them', 'not', 'no', 'do', 'does',
        'should', 'must', 'can', 'will', 'would', 'could',
    }

    @classmethod
    def _tokens(cls, text: str) -> set:
        words = [w.lower() for w in cls._WORD_RE.findall(text or '')]
        return {w for w in words if len(w) > 2 and w not in cls._STOPWORDS_EN}

    @classmethod
    def _jaccard(cls, a: str, b: str) -> float:
        ta, tb = cls._tokens(a), cls._tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union else 0.0

    def _pick_diverse_pair(
        self, candidates: List[Tuple[str, str]], similarity_threshold: float = 0.85,
    ) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]], bool]:
        """Pick top-1 + most diverse partner. Returns (top1, partner, is_collapsed)."""
        if not candidates:
            return None, None, True
        top1 = self._select_best(candidates, collect_lesson=True)
        rest = [c for c in candidates if c[1] != top1[1]]
        if not rest:
            return top1, None, True
        # Choose the candidate with LOWEST Jaccard vs top1 AND still high enough quality
        # (we already excluded top1 — rest contains the rest of the arena).
        scored = [(c, self._jaccard(top1[1], c[1])) for c in rest]
        scored.sort(key=lambda x: x[1])  # ascending similarity
        partner, min_sim = scored[0]
        collapsed = min_sim >= similarity_threshold
        return top1, partner, collapsed

    # ══════════════════════════════════════════════════════════════════════
    # Run plumbing
    # ══════════════════════════════════════════════════════════════════════
