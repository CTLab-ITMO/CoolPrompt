"""Prompt-contract analysis and strategy routing for RIDER Genesis Ultra."""

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


class RiderContractMixin:
    @staticmethod
    def _append_unique(items: List[Any], value: str) -> None:
        value = str(value).strip()
        if value and value not in items:
            items.append(value)

    @staticmethod
    def _contains_any(text_lower: str, needles: Tuple[str, ...]) -> bool:
        return any(n in text_lower for n in needles)

    def _enrich_contract_with_static_analysis(
        self, prompt: str, contract: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deterministically enrich LLM contract for production prompts.

        The LLM contract is good at broad intent. This layer catches hard production
        invariants that must never depend on taste: JSON schemas, legal level scales,
        category labels, dictionary tags, and risk-calibration rules.
        """
        c = dict(self._DEFAULT_CONTRACT)
        c.update({k: v for k, v in contract.items() if k in self._DEFAULT_CONTRACT})

        must = list(c.get('must_preserve') or [])
        failures = list(c.get('failure_modes') or [])
        rules = list(c.get('domain_rules') or [])
        dims = list(c.get('quality_dimensions') or [])
        avoid = [s for s in (c.get('avoid_strategies') or []) if s in self._VALID_STRATEGIES]

        prompt_lower = (prompt or '').lower()
        archetype = str(c.get('task_archetype') or '').lower()
        is_moderation = self._contains_any(prompt_lower, self._MODERATION_HINTS)
        is_translation = self._contains_any(prompt_lower, self._TRANSLATION_HINTS)
        is_extraction = archetype == 'data_extraction' or self._contains_any(
            prompt_lower,
            (
                'ocr', 'рукопис', 'расшифр', 'транскрип', 'наборщик', 'оглавлен',
                'table of contents', 'toc', 'извлеч', 'extract',
            ),
        )
        has_json_schema = bool(re.search(r'"(?:quote|context|risk|categories|level|translation|reason)"', prompt or ''))
        has_levels = bool(re.search(r'\b(?:level|уровень)[ \t]*[0-3]\b', prompt or '', re.IGNORECASE))
        has_geo_risk = self._contains_any(prompt_lower, self._GEO_RISK_HINTS)
        is_code = archetype in {'code_generation', 'code_review', 'debugging'} or bool(
            re.search(r"```|def |class |function |typescript|python|javascript|sql|api|stack trace|traceback", prompt_lower)
        )

        if is_moderation or has_json_schema or has_levels:
            c['task_archetype'] = 'classification'
            if c.get('domain') in (None, '', 'general'):
                c['domain'] = 'content moderation / legal compliance'
            if c.get('output_format_anchor') in (None, '', 'free-form'):
                c['output_format_anchor'] = 'strict machine-parseable JSON'

            for item in (
                'exact JSON output schema and allowed field names',
                'category labels / legal taxonomy names',
                'level scale and calibration thresholds',
                'safe-context filters and false-positive/false-negative policy',
                'quote/context/location copying rules',
                'legal references and article numbers',
            ):
                self._append_unique(must, item)
            for failure in (
                'schema drift or extra prose outside JSON',
                'level calibration drift',
                'exclusive-output contract drift',
                'priority order drift',
                'level cap or exception drift',
                'false negatives on level 2/3 risk',
                'generic risk text not grounded in quote',
                'invented legal category or article',
            ):
                self._append_unique(failures, failure)
            for rule in (
                'For moderation/legal prompts, preserve the original legal taxonomy, output schema, and level scale before improving wording.',
                'Prefer bounded legal precision over creative reframing; never invent categories, laws, fields, or thresholds not present in the source prompt.',
                'False negatives on high-risk items are more severe than conservative inclusion, but explicit safe-context filters must remain intact.',
                'Every risk explanation should cite a concrete signal from quote/input text, not a vague topic-level suspicion.',
                'If the source prompt says there is only one allowed output, keep that output contract exclusive: no prose, no alternative formats, no helper text.',
                'Preserve priority order, escalation rules, caps, and exceptions exactly before adding any clarifying wording.',
            ):
                self._append_unique(rules, rule)
            for dim in (
                'schema_compliance',
                'level_calibration',
                'exclusive_output_compliance',
                'priority_order_preservation',
                'level_cap_semantics',
                'false_negative_resistance',
                'false_positive_control',
                'quote_grounding',
            ):
                self._append_unique(dims, dim)
            if 'creative' not in avoid:
                avoid.append('creative')
            c['recommended_strategies'] = [
                s for s in ('analytical', 'structural', 'techniques', 'depth', 'creative')
                if s not in avoid
            ]

        if has_geo_risk:
            for item in (
                'Russia/Ukraine/Russian-side geopolitical escalation rules',
                'maximum level caps and Level 0 exclusion rules',
            ):
                self._append_unique(must, item)
            for failure in (
                'dropping Russia/Ukraine escalation policy',
                'raising Level 0 items when the source prompt forbids it',
                'raising above Level 3',
            ):
                self._append_unique(failures, failure)
            self._append_unique(
                rules,
                'If the source prompt contains Russia/Ukraine/Russian-side escalation logic, preserve its exact cap semantics: raise only as specified, never above Level 3, and do not escalate Level 0 unless the source explicitly says so.',
            )
            for dim in ('geopolitical_escalation_semantics', 'level_cap_semantics'):
                self._append_unique(dims, dim)

        if is_translation:
            c['task_archetype'] = 'translation'
            if c.get('domain') in (None, '', 'general'):
                c['domain'] = 'literary translation'
            if c.get('output_format_anchor') in (None, '', 'free-form'):
                c['output_format_anchor'] = 'translation output specified by source prompt'
            for item in (
                'dictionary/glossary tags and term-handling policy',
                'translation-only or wrapper-tag output format',
                'source-language to Russian literary style contract',
                'author rhythm, syntax, register, and intentional roughness',
            ):
                self._append_unique(must, item)
            for failure in (
                'over-smoothing author syntax and rhythm',
                'ignoring dictionary terms or names',
                'adding commentary when output must be translation-only',
                'forcing verse or genre conventions not requested by source',
            ):
                self._append_unique(failures, failure)
            for rule in (
                'For literary translation prompts, preserve the author-specific rhythm and syntax instead of normalizing everything into generic smooth Russian prose.',
                'Dictionary/glossary instructions are hard constraints; if they conflict with factual or stylistic fidelity, the prompt must specify how to resolve the conflict.',
                'Do not add pre-translation analysis to user-visible output unless the source prompt explicitly requires it.',
            ):
                self._append_unique(rules, rule)
            for dim in (
                'fidelity_to_source',
                'authorial_rhythm',
                'glossary_compliance',
                'no_extra_commentary',
            ):
                self._append_unique(dims, dim)
            # Literary translation can use creative language, but only after structural
            # and analytical constraints have locked the production contract.
            c['recommended_strategies'] = [
                s for s in ('analytical', 'structural', 'depth', 'techniques', 'creative')
                if s not in avoid
            ]

        if is_code:
            for item in (
                'code blocks, inline code, filenames, API names, and placeholders',
                'runtime constraints, error messages, stack traces, and expected behavior',
                'public interfaces and backward-compatibility requirements',
            ):
                self._append_unique(must, item)
            for failure in (
                'renaming public API or config keys',
                'dropping error context or reproduction steps',
                'inventing dependencies not requested',
                'changing behavior outside the requested scope',
            ):
                self._append_unique(failures, failure)
            for rule in (
                'For code prompts, preserve exact identifiers, file paths, stack traces, and public contracts; improve debugging structure without fictional APIs.',
                'Prefer minimal, testable fixes and explicit verification steps over creative rewrites.',
            ):
                self._append_unique(rules, rule)
            for dim in ('identifier_preservation', 'testability', 'scope_control', 'runtime_correctness'):
                self._append_unique(dims, dim)
            if 'creative' not in avoid:
                avoid.append('creative')
            c['recommended_strategies'] = [
                s for s in ('analytical', 'structural', 'techniques', 'depth', 'creative')
                if s not in avoid
            ]

        if is_extraction and not is_moderation:
            for item in (
                'verbatim source text / source structure',
                'markup tags, placeholders, page numbers, and line breaks',
                'no commentary outside the requested extraction output',
            ):
                self._append_unique(must, item)
            for failure in (
                'paraphrasing instead of extraction',
                'omitting source lines or structural elements',
                'adding explanations around extracted output',
                'inventing unreadable or missing text',
            ):
                self._append_unique(failures, failure)
            for rule in (
                'For extraction/OCR prompts, optimize for exact source preservation and completeness, not richer prose.',
                'Do not add creative roleplay, examples, or extra output sections if they make the extraction contract less literal.',
            ):
                self._append_unique(rules, rule)
            for dim in ('verbatim_fidelity', 'completeness', 'markup_compliance', 'no_extra_commentary'):
                self._append_unique(dims, dim)
            if 'creative' not in avoid:
                avoid.append('creative')
            c['recommended_strategies'] = [
                s for s in ('analytical', 'structural', 'techniques', 'depth', 'creative')
                if s not in avoid
            ]

        if archetype in {'creative_writing', 'analytical_essay', 'persuasion', 'brainstorming'}:
            for item in (
                'original user topic, audience, tone, and requested genre',
                'the requested output type (essay, story, argument, idea list, etc.)',
            ):
                self._append_unique(must, item)
            for failure in (
                'generic classroom prompt with no distinctive angle',
                'overly broad structure that produces predictable writing',
                'missing sensory/detail/examples requirements',
                'tone mismatch with the requested genre',
            ):
                self._append_unique(failures, failure)
            for rule in (
                'For short creative or essay prompts, produce a full high-leverage instruction: role, concrete angle, structure, sensory/detail requirements, quality bar, and anti-generic constraints.',
                'Make the prompt immediately usable by a strong model; avoid vague meta-advice and ask for specific rhetorical moves.',
            ):
                self._append_unique(rules, rule)
            for dim in ('distinctive_angle', 'structure_quality', 'specificity', 'anti_genericity'):
                self._append_unique(dims, dim)
            c['recommended_strategies'] = [
                s for s in ('creative', 'structural', 'depth', 'analytical', 'techniques')
                if s not in avoid
            ]

        c['must_preserve'] = must[:40]
        c['failure_modes'] = failures[:24]
        c['domain_rules'] = rules[:20]
        c['quality_dimensions'] = dims[:16]
        c['avoid_strategies'] = avoid
        return c

    def _extract_contract(self, prompt: str) -> Dict[str, Any]:
        """Extract structured prompt contract via Instructor/Pydantic. Falls back to defaults."""
        meta = _CONTRACT_PROMPT.format(prompt=prompt)
        try:
            obj = self._generate_structured(
                prompt=meta,
                schema=_PromptContractSchema,
                role="planner",
                temperature=0.2, max_tokens=600,
                allowed_starts=("{",),
                max_retries=2,
            )
            if obj is None:
                raise ValueError("structured contract unavailable")
            data = self._schema_to_dict(obj)
            # Merge with defaults so missing fields don't break downstream code.
            contract = dict(self._DEFAULT_CONTRACT)
            contract.update({k: v for k, v in data.items() if k in self._DEFAULT_CONTRACT})
            # Sanitize strategy lists.
            rec = [s for s in contract.get('recommended_strategies', []) if s in self._VALID_STRATEGIES]
            if not rec:
                rec = list(self._VALID_STRATEGIES)
            avoid = [s for s in contract.get('avoid_strategies', []) if s in self._VALID_STRATEGIES]
            rec = [s for s in rec if s not in avoid]
            if not rec:
                rec = [s for s in self._VALID_STRATEGIES if s not in avoid] or list(self._VALID_STRATEGIES)
            contract['recommended_strategies'] = rec
            contract['avoid_strategies'] = avoid
            if not isinstance(contract.get('must_preserve'), list):
                contract['must_preserve'] = []
            if not isinstance(contract.get('failure_modes'), list):
                contract['failure_modes'] = []
            if not isinstance(contract.get('domain_rules'), list):
                contract['domain_rules'] = []
            if not isinstance(contract.get('quality_dimensions'), list):
                contract['quality_dimensions'] = []
            return self._enrich_contract_with_static_analysis(prompt, contract)
        except Exception as exc:
            logger.info(f"Contract extraction failed, using defaults: {exc}")
            return self._enrich_contract_with_static_analysis(prompt, dict(self._DEFAULT_CONTRACT))

    @staticmethod
    def _extract_json_value(text: str, allowed_starts: Tuple[str, ...] = ('{', '[')) -> Optional[str]:
        """Extract the first valid JSON object/array text from LLM output."""
        if not text:
            return None
        # Strip common code fences.
        text = re.sub(r"```(?:json)?\s*", "", text.strip())
        text = text.replace("```", "").strip()
        decoder = json.JSONDecoder()
        for start, ch in enumerate(text):
            if ch not in allowed_starts:
                continue
            try:
                _value, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            return text[start:start + end]
        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the first top-level {...} JSON object from LLM output."""
        return RiderGenesis._extract_json_value(text, allowed_starts=('{',))

    def _adaptive_strategies(self, mode: str) -> List[str]:
        """Pick the strategy set for a mode based on contract.recommended_strategies."""
        rec = list(self._contract.get('recommended_strategies', self._VALID_STRATEGIES))
        avoid = set(self._contract.get('avoid_strategies', []))

        # Ordering preference comes from contract.
        ordered = [s for s in rec if s in self._VALID_STRATEGIES and s not in avoid]
        # Fallback: fill with unused valid strategies in canonical order.
        for s in self._VALID_STRATEGIES:
            if s not in ordered and s not in avoid:
                ordered.append(s)
        # Hard floor in case the contract zeroes everything out.
        if not ordered:
            ordered = list(self._VALID_STRATEGIES)

        mode_budget = {'light': 2, 'blitz': 4, 'standard': 5, 'ultra': 5}.get(mode, 5)
        selected = ordered[:mode_budget]

        # Light always keeps analytical+structural in the mix for robustness, but lets
        # the contract reorder: if contract prefers 'depth, techniques' for a code prompt,
        # respect it instead of forcing analytical.
        return selected

    def _strategy_allowed(self, strategy: str) -> bool:
        return strategy in self._VALID_STRATEGIES and strategy not in set(self._contract.get('avoid_strategies') or [])

    def _mutation_strategy(self, preferred: str, fallback: str = 'techniques') -> str:
        if self._strategy_allowed(preferred):
            return preferred
        if self._strategy_allowed(fallback):
            return fallback
        for strategy in ('analytical', 'structural', 'depth', 'techniques', 'creative'):
            if self._strategy_allowed(strategy):
                return strategy
        return fallback

    # ══════════════════════════════════════════════════════════════════════
    # Context block builders (injected into strategy/compare/refine prompts)
    # ══════════════════════════════════════════════════════════════════════

    def _contract_block(self) -> str:
        if not self._contract:
            return ""
        c = self._contract
        lines = [
            "Task contract:",
            f"  archetype: {c.get('task_archetype', 'other')}",
            f"  language: {c.get('language', 'unknown')}",
            f"  domain: {c.get('domain', 'general')}",
            f"  audience: {c.get('audience', 'general')}",
            f"  output anchor: {c.get('output_format_anchor', 'free-form')}",
        ]
        fm = c.get('failure_modes') or []
        if fm:
            lines.append(f"  likely failure modes: {'; '.join(str(x) for x in fm[:5])}")
        rules = c.get('domain_rules') or []
        if rules:
            lines.append("  hard domain rules:")
            for rule in rules[:6]:
                lines.append(f"    - {rule}")
        dims = c.get('quality_dimensions') or []
        if dims:
            lines.append(f"  evaluation dimensions: {'; '.join(str(x) for x in dims[:8])}")
        return "\n".join(lines) + "\n\n"

    def _preservation_block(self) -> str:
        items = self._contract.get('must_preserve') or []
        original = getattr(self, "_original_prompt", "") or ""
        concrete: List[str] = []
        if original:
            artifacts = self._extract_required_artifacts(original)
            for kind, values in artifacts.items():
                if kind == 'code_blocks':
                    concrete.append(f"preserve {len(values)} fenced code block(s) exactly")
                elif kind == 'inline_code':
                    concrete.extend(f"preserve inline code token {v}" for v in values[:8])
                elif kind == 'placeholders':
                    concrete.extend(f"preserve placeholder {v}" for v in values[:12])
                elif kind == 'urls':
                    concrete.extend(f"preserve URL {v}" for v in values[:8])
                elif kind == 'xml_tag_names':
                    concrete.extend(f"preserve XML/tag anchor {v}" for v in values[:12])
                elif kind == 'json_field_names':
                    concrete.append(f"preserve JSON field names: {', '.join(values[:20])}")
                elif kind == 'category_labels':
                    concrete.append(f"preserve category labels/taxonomy: {', '.join(values[:16])}")
                elif kind == 'level_values':
                    concrete.append(f"preserve level scale values: {', '.join(values)}")
                elif kind == 'enumeration_markers':
                    concrete.append(f"preserve explicit level/category markers such as: {', '.join(values[:10])}")
                elif kind == 'enumerated_count':
                    concrete.append(f"preserve numbered-list structure (original has {values[0]} items)")
                elif kind == 'markdown_table_lines':
                    concrete.append("preserve markdown table structure")
                elif kind == 'geo_escalation_terms':
                    concrete.append(f"preserve geopolitical escalation terms: {', '.join(values[:10])}")
        merged_items = list(items)
        for item in concrete:
            self._append_unique(merged_items, item)
        if not merged_items:
            return ""
        bullets = "\n".join(f"  - {it}" for it in merged_items[:16])
        return (
            "PRESERVATION RULES (must-keep-verbatim artifacts from the original):\n"
            f"{bullets}\n"
            "Do NOT rewrite, translate, drop, rename, or paraphrase these artifacts. "
            "You may improve surrounding prose, but the production contract, schema, "
            "taxonomy, thresholds, and tags must survive unchanged unless the original "
            "explicitly asks to change them.\n\n"
        )
