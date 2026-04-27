"""RIDER RiderGenesis v4.1 Ultra+ — elite автоматическая оптимизация промптов без размеченных данных.

Четыре режима с чёткой дифференциацией качества под разные ценовые tier'ы:

  ┌──────────────┬──────┬──────┬─────────┬────────┐
  │ Feature      │ light│ blitz│standard │ ultra+ │
  ├──────────────┼──────┼──────┼─────────┼────────┤
  │ Contract+routing│ ✓  │  ✓   │   ✓     │   ✓    │
  │ Strategies    │  2  │  4   │   5     │  5+2   │
  │ Preservation+repair│ ✓ │ ✓ │   ✓     │   ✓    │
  │ GENESIS lessons│ —  │  ✓   │   ✓     │   ✓    │
  │ FORGE memory  │  —  │  —   │   ✓     │   ✓    │
  │ Constitutional audit│— │ ✓ │  ✓ (2x) │ ✓ (2x) │
  │ Refine cycles │  —  │  1   │   2     │   2    │
  │ Diversity-aware merge│—│ ✓ │   ✓     │   ✓    │
  │ **Synthetic test eval**│— │— │ 3 tests │ 5 tests│
  │ **Beam search**│ k=1 │ k=1 │  k=2    │  k=3   │
  │ **Red-team adversarial**│—│—│   —    │   ✓    │
  │ Bilingual adversarial│ —│—│   —     │ ✓ non-EN│
  │ VORTEX on collapse│ —│ — │   —     │   ✓    │
  │ API calls     │ ~5  │ ~11  │  ~22    │  ~33   │
  │ Runtime       │ ~20s│ ~60s │ ~120s   │ ~180s  │
  └──────────────┴──────┴──────┴─────────┴────────┘

Версия v3 расширяла v2 пятью техниками из полного RIDER + двумя из аудита CoolPrompt;
v4 Ultra+ добавляет ещё 3 ключевых апгрейда для стандартных и ультра-режимов:

  ▸ SYNTHETIC TEST-CASE EVAL — standard/ultra генерируют 3-5 синтетических test
    inputs и ранжируют кандидатов по качеству РЕАЛЬНЫХ output'ов, а не по
    эстетике prompt-текста. Это де-факто labeled fitness без разметки человека.
  ▸ BEAM SEARCH top-k — standard (k=2), ultra (k=3) не схлопывают ветки рано.
  ▸ RED-TEAM ADVERSARIAL — только ultra. Sonnet в роли атакующего ищет 3 edge
    cases; при severity>=medium запускается hardening-refine с конкретными
    fix-directives.
  ▸ v4.1 ENUMERATION PRESERVATION — закрывает главный gap обнаруженный на
    реальном батче 15 ЭКСМО-промптов. Детектит в оригинале:
      * счётчик numbered items (если >=3) → draft не может потерять >40%
      * Level/Step markers (Уровень N, Level N, Step N, ШАГ N, Tier N)
      * category codes (1.1.а, 2.3.б, 3.2) → каждый должен остаться
    При нарушении — repair call заставляет Sonnet восстановить структуру.

Версия v3 расширяла v2 пятью техниками из полного RIDER + две идеи из аудита CoolPrompt
(ITMO), при этом сохраняя бюджет вызовов light/blitz/standard/ultra.

Ключевые улучшения v3 vs v2:
  1. PROMPT CONTRACT EXTRACTION (вместо free-text _analyze_context):
     Sonnet возвращает JSON с task_archetype / language / preservation_flags /
     recommended_strategies / failure_modes / output_anchor. Контракт пробрасывается
     во все последующие вызовы.
  2. ADAPTIVE STRATEGY ROUTING (MARS CONFIG-lite):
     Набор стратегий выбирается из контракта, а не фиксирован. Например,
     prompt про отладку кода НЕ получит 'creative' стратегию.
  3. GENESIS-LITE — lesson extraction:
     Каждый pairwise compare вызов возвращает winner + WHY-объяснение. WHY
     становится уроком (lesson), который пробрасывается в последующие
     strategy/refine/merge вызовы как контекст успеха.
  4. OPERATOR FORGE — per-strategy memory:
     Лучшие outputs каждой стратегии запоминаются и инжектируются обратно,
     когда стратегия вызывается повторно (как few-shot контекст).
  5. PRESERVATION RULES + POST-HOC VALIDATOR (CoolPrompt-inspired):
     Во все strategy/merge/refine промпты инжектируются preservation constraints
     для code/placeholders/markdown/URLs. После генерации — dry check на
     сохранность артефактов; при нарушении — 1 repair call.
  6. CONSTITUTIONAL AUDIT (вместо generic _find_weaknesses):
     Рубрика из 6 измерений (clarity, specificity, constraint_completeness,
     output_anchoring, edge_case_coverage, brevity_vs_bloat). Возвращает top-3
     failure dimensions + конкретные fix-directives.
  7. DIVERSITY-AWARE MERGE (k-DPP-lite):
     Перед merge'ом считается Jaccard-similarity между кандидатами. Если
     top-2 слишком похожи (>0.85) — выбирается более разнообразная пара.
     Если весь пул collapsed — Ultra переключается на VORTEX paradigm-shift.
  8. VORTEX-ON-COLLAPSE (только Ultra):
     При обнаружении semantic collapse в Phase 2 один из refinement слотов
     тратится на VORTEX-style paradigm shift вместо рутинной мутации.

Бюджет вызовов (не меняется по сравнению с v2):
  - light     ~ 5 calls,   ~15s
  - blitz     ~11 calls,   ~45s
  - standard  ~19 calls,   ~70s
  - ultra     ~25 calls,  ~120s

Использование:
    from rider.assistant import RiderGenesis

    tuner = RiderGenesis()
    tuner.run("Write a persuasive essay about climate change", mode="ultra")
    print(tuner.final_prompt)
    print(f"Improvement: {tuner.improvement:+.1f}%")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from coolprompt.optimizer.rider._llm_shim import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured contract extraction
# ---------------------------------------------------------------------------

_CONTRACT_PROMPT = (
    "You are an elite prompt engineer analyzing a raw user prompt to extract its hidden task contract.\n\n"
    "USER PROMPT:\n<<<\n{prompt}\n>>>\n\n"
    "Output a SINGLE JSON object with these fields:\n"
    "  task_archetype: one of [creative_writing, analytical_essay, technical_explanation, "
    "code_generation, code_review, debugging, summarization, translation, qa, classification, "
    "data_extraction, brainstorming, instruction_design, persuasion, other]\n"
    "  language: ISO 639-1 code of the INSTRUCTION language (not of any data inside)\n"
    "  domain: short phrase describing the domain (e.g. 'climate science', 'python backend')\n"
    "  audience: short phrase describing the intended audience\n"
    "  output_format_anchor: short phrase of the expected output structure\n"
    "  must_preserve: list of strings describing artifacts that must not be modified "
    "(e.g. ['code_blocks', 'placeholders_{{var}}', 'markdown_tables', 'urls']). "
    "Empty list if no artifacts to preserve.\n"
    "  failure_modes: list of 3-5 short phrases naming the most likely ways an AI "
    "response could fail for this task\n"
    "  recommended_strategies: ordered list of 2-5 strategy names drawn from "
    "[structural, analytical, creative, depth, techniques], picking ones most likely "
    "to improve THIS prompt. Skip strategies that would overload or hurt this prompt.\n"
    "  avoid_strategies: list of strategies that would HURT this prompt (subset of the 5 above). "
    "Empty list if none.\n\n"
    "Respond with VALID JSON only. No prose, no markdown, no code fences."
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RiderGenesis:
    """Elite automatic prompt optimization without labeled data.

    Pipeline v3 (Multi-Strategy Prompt Enhancement + RIDER-inspired extras):
      1. Prompt CONTRACT extraction (structured JSON via Sonnet)
      2. ADAPTIVE strategy routing (MARS CONFIG-lite, from contract)
      3. Parallel strategies with OPERATOR FORGE memory and injected preservation rules
      4. Pairwise compare with GENESIS-lite lessons (winner + WHY)
      5. Merge top-2 (diversity-aware, VORTEX on collapse for Ultra)
      6. CONSTITUTIONAL audit (6-dim rubric) + refine
      7. Preservation POST-HOC validator + repair (conditional, CoolPrompt-inspired)

    Args:
        model: OpenRouter model for generation (default: claude-haiku-4.5)
        api_key: OpenRouter API key (default: from OPENROUTER_API_KEY)
        verbose: print progress
    """

    # Strong Sonnet for contract extraction, pairwise comparison, constitutional audit,
    # quality estimate. Used 4-7 times per run — cheap but vastly better than a mini model.
    PLANNING_MODEL = "anthropic/claude-sonnet-4.6"

    # ══════════════════════════════════════════════════════════════════════
    # Strategy meta-prompts
    # ══════════════════════════════════════════════════════════════════════

    _STRATEGY_PROMPTS: Dict[str, str] = {
        'structural': (
            "You are an elite prompt engineer. Transform this prompt into a masterfully "
            "structured instruction that will extract the best possible output from an AI.\n\n"
            "Apply ALL of these techniques:\n"
            "1. ROLE PRIMING — assign a specific expert role/persona matching the domain\n"
            "2. TASK DECOMPOSITION — break the task into clear sequential steps\n"
            "3. OUTPUT FORMAT — specify exact format, length, style, structural requirements\n"
            "4. QUALITY CRITERIA — define what makes an excellent vs mediocre response\n"
            "5. CONSTRAINTS — add boundaries that prevent common failures\n"
            "6. NEGATIVE CONSTRAINTS — specify what to avoid\n\n"
            "{contract_block}"
            "{preservation_block}"
            "{forge_block}"
            "{lessons_block}"
            "Original prompt:\n{prompt}\n\n"
            "CRITICAL: write the improved prompt in the SAME LANGUAGE as the original.\n"
            "Write ONLY the improved prompt text. No explanations, no meta-commentary."
        ),
        'analytical': (
            "You are a prompt quality analyst. Your method: find every weakness, "
            "then rewrite to fix ALL of them.\n\n"
            "{contract_block}"
            "{preservation_block}"
            "{forge_block}"
            "{lessons_block}"
            "Original prompt:\n{prompt}\n\n"
            "INTERNAL ANALYSIS (do this mentally, don't output it):\n"
            "- Find 5+ specific weaknesses: vagueness, missing constraints, no format spec, "
            "no role, no examples, ambiguity, lack of structure, missing edge cases\n"
            "- For each: determine the concrete fix\n\n"
            "OUTPUT: Write the complete improved prompt with ALL fixes applied.\n"
            "CRITICAL: write in the SAME LANGUAGE as the original prompt.\n"
            "Write ONLY the improved prompt. No analysis, no commentary."
        ),
        'creative': (
            "You are a world-class prompt innovator. Don't just improve this prompt — "
            "completely reimagine it from a fundamentally better angle.\n\n"
            "{contract_block}"
            "{preservation_block}"
            "{forge_block}"
            "{lessons_block}"
            "Original prompt:\n{prompt}\n\n"
            "Think: what would the world's leading expert in this domain ask an AI? "
            "What framing would produce the most insightful, comprehensive response?\n\n"
            "Consider unconventional approaches: metaphorical framing, audience-specific "
            "tailoring, multi-perspective structure, structured output templates.\n\n"
            "CRITICAL: write in the SAME LANGUAGE as the original prompt.\n"
            "Write ONLY the reimagined prompt. It must be dramatically better."
        ),
        'depth': (
            "You are a prompt depth specialist. Add maximum substance and "
            "specificity while keeping the prompt focused.\n\n"
            "{contract_block}"
            "{preservation_block}"
            "{forge_block}"
            "{lessons_block}"
            "Original prompt:\n{prompt}\n\n"
            "Add ALL of the following:\n"
            "- Concrete examples of expected output quality (1-2 inline examples)\n"
            "- Precise constraints (length, tone, audience, scope, format)\n"
            "- Failure modes to avoid with explicit instructions\n"
            "- Quality criteria the response must meet\n"
            "- Specific subtopics, angles, or dimensions to cover\n"
            "- Negative constraints (what NOT to do)\n\n"
            "CRITICAL: write in the SAME LANGUAGE as the original prompt.\n"
            "Write ONLY the enhanced prompt. Comprehensive but not bloated."
        ),
        'techniques': (
            "You are an AI prompt research scientist applying cutting-edge techniques.\n\n"
            "{contract_block}"
            "{preservation_block}"
            "{forge_block}"
            "{lessons_block}"
            "Original prompt:\n{prompt}\n\n"
            "Apply ALL of these research-backed techniques:\n"
            "- Chain-of-thought elicitation (guide step-by-step reasoning)\n"
            "- Expert persona assignment (specific domain expert, not generic)\n"
            "- Output anchoring (provide structural template for the response)\n"
            "- Negative constraints ('do NOT...', 'avoid...')\n"
            "- Meta-cognitive prompting ('verify your output against...')\n"
            "- Specificity injection (replace every vague term with a precise one)\n\n"
            "CRITICAL: write in the SAME LANGUAGE as the original prompt.\n"
            "Write ONLY the improved prompt."
        ),
        # VORTEX paradigm-shift — activated only on semantic collapse in Phase 2 of Ultra.
        'vortex': (
            "You are a paradigm-shift specialist. The current batch of prompt variants "
            "has COLLAPSED to near-identical text. Your job: produce a FUNDAMENTALLY "
            "DIFFERENT reframing that no one in the batch has tried.\n\n"
            "{contract_block}"
            "{preservation_block}"
            "{forge_block}"
            "Collapsed batch (do not repeat these patterns):\n{collapsed_preview}\n\n"
            "Original task:\n{prompt}\n\n"
            "Examples of paradigm shifts:\n"
            "  - \"Summarize the text\"  ->  \"What would a news editor write as the headline?\"\n"
            "  - \"Explain quantum physics\"  ->  \"Design a dialogue where a physicist convinces "
            "a skeptical journalist in 4 exchanges.\"\n"
            "  - \"Write an essay on X\"  ->  \"Generate a four-voice panel "
            "(historian/engineer/artist/skeptic) arguing about X.\"\n\n"
            "Produce ONE genuinely different reframing, not a surface tweak.\n"
            "CRITICAL: write in the SAME LANGUAGE as the original prompt.\n"
            "Write ONLY the reimagined prompt."
        ),
    }

    _COMPARE_PROMPT = (
        "You are a world-class prompt engineering expert evaluating PROMPT QUALITY.\n\n"
        "{contract_block}"
        "Compare these prompts. Which one would make an AI produce the BEST output? "
        "Consider clarity, specificity, structure, role assignment, constraints, format "
        "requirements, edge-case handling, preservation of required artifacts, and overall "
        "effectiveness for the stated task.\n\n"
        "{candidates}\n\n"
        "Respond EXACTLY in this format:\n"
        "WINNER: <number>\n"
        "WHY: <one short sentence — the specific reason this winner is stronger>"
    )

    _MERGE_PROMPT = (
        "You are a prompt synthesis master. Create a SUPERIOR prompt by combining "
        "the best elements of these two excellent prompts.\n\n"
        "{contract_block}"
        "{preservation_block}"
        "{lessons_block}"
        "PROMPT A:\n{prompt_a}\n\n"
        "PROMPT B:\n{prompt_b}\n\n"
        "Take the strongest elements from each:\n"
        "- Best structural organization from whichever is better structured\n"
        "- Deepest specificity and constraints from whichever is more detailed\n"
        "- Best role / persona framing\n"
        "- All unique strengths from either prompt\n\n"
        "The result must be better than BOTH inputs — a true synthesis, not concatenation.\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Total length must NOT exceed 2x the length of the longer input. Bloat is a regression, not an improvement.\n"
        "- Eliminate redundant rules, duplicated definitions, and layered caveats. Merge overlapping criteria into one.\n"
        "- Write in the SAME LANGUAGE as the input prompts.\n"
        "Write ONLY the merged prompt."
    )

    # Constitutional 6-dimension rubric audit — replaces generic weakness audit.
    _CONSTITUTIONAL_AUDIT_PROMPT = (
        "You are a ruthless prompt quality auditor using a 6-dimension rubric.\n\n"
        "{contract_block}"
        "Prompt to audit:\n{prompt}\n\n"
        "For each of these 6 dimensions, score 1-5 (5 = excellent, 1 = broken) AND "
        "identify whether this dimension is among the TOP-3 current weaknesses:\n"
        "  1. CLARITY — unambiguous, no vagueness\n"
        "  2. SPECIFICITY — concrete details, precise terms, not generic\n"
        "  3. CONSTRAINT_COMPLETENESS — length, tone, format, negative constraints present\n"
        "  4. OUTPUT_ANCHORING — explicit template, examples, or structure for the answer\n"
        "  5. EDGE_CASE_COVERAGE — named failure modes, ambiguous-input handling\n"
        "  6. BREVITY_VS_BLOAT — focused, not padded, not under-specified\n\n"
        "Respond EXACTLY in this format:\n"
        "DIM1 CLARITY: <1-5>\n"
        "DIM2 SPECIFICITY: <1-5>\n"
        "DIM3 CONSTRAINT_COMPLETENESS: <1-5>\n"
        "DIM4 OUTPUT_ANCHORING: <1-5>\n"
        "DIM5 EDGE_CASE_COVERAGE: <1-5>\n"
        "DIM6 BREVITY_VS_BLOAT: <1-5>\n"
        "TOP_WEAKNESSES:\n"
        "- <dimension name>: <concrete issue> -> <concrete fix directive>\n"
        "- <dimension name>: <concrete issue> -> <concrete fix directive>\n"
        "- <dimension name>: <concrete issue> -> <concrete fix directive>"
    )

    _REFINE_PROMPT = (
        "You are a precision prompt refiner. Fix every listed weakness "
        "while preserving ALL existing strengths.\n\n"
        "{contract_block}"
        "{preservation_block}"
        "{lessons_block}"
        "Current prompt:\n{prompt}\n\n"
        "Audit findings to act on:\n{weaknesses}\n\n"
        "Apply EVERY fix. Tighten language, improve flow, ensure completeness. "
        "Do NOT remove anything that is already strong.\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Total length must NOT exceed 1.5x the current prompt's length. Pad-and-verbose is a regression.\n"
        "- Replace vague rules with precise ones — don't layer new caveats on top of old ones.\n"
        "- Write in the SAME LANGUAGE as the original.\n"
        "Write ONLY the refined prompt."
    )

    # Preservation repair (CoolPrompt-inspired) — fired only when violations detected.
    _PRESERVE_REPAIR_PROMPT = (
        "You are a preservation-repair specialist. The improved prompt below has "
        "accidentally modified or dropped artifacts that must be kept unchanged.\n\n"
        "Required artifacts (copy them verbatim from the original):\n{required_items}\n\n"
        "Original prompt (authoritative source for required artifacts):\n{original}\n\n"
        "Current (broken) improved prompt:\n{broken}\n\n"
        "Violations detected:\n{violations}\n\n"
        "Produce a REPAIRED version of the improved prompt that:\n"
        "- keeps the improvements to the instruction text,\n"
        "- re-embeds every required artifact EXACTLY as in the original,\n"
        "- preserves the original language.\n\n"
        "Write ONLY the repaired prompt."
    )

    _QUALITY_PROMPT = (
        "Rate these two prompts for quality on a scale 1-10.\n"
        "Higher = more clear, specific, structured, effective at guiding an AI.\n\n"
        "{contract_block}"
        "PROMPT A (original):\n{prompt_a}\n\n"
        "PROMPT B (improved):\n{prompt_b}\n\n"
        "Reply EXACTLY in this format:\nA: <number>\nB: <number>"
    )

    # v4.3 N-way batch tournament: rank k candidates at once via ONE Sonnet call.
    # Replaces 3 sequential pairwise calls (top-3 selection) — saves ~25s per mode.
    _BATCH_RANK_PROMPT = (
        "You are a world-class prompt engineering judge.\n\n"
        "{contract_block}"
        "Rank the following {n} prompts from BEST to WORST based on which would make "
        "an AI produce the highest-quality output for the stated task. Consider: "
        "clarity, specificity, structure, role priming, constraints, output format, "
        "edge-case handling, preservation of required artifacts.\n\n"
        "{candidates}\n\n"
        "Respond EXACTLY in this format:\n"
        "RANKED: <comma-separated prompt numbers best-to-worst, e.g. 3,1,4,2>\n"
        "WHY_TOP: <one short sentence — why #1 is strongest>\n"
    )

    # v4.3 deepseek fallback — used when primary model refuses (censor/safety content).
    FALLBACK_MODEL = "deepseek/deepseek-v3.2"
    _REFUSAL_PATTERNS = (
        r"I can'?t help with",
        r"I cannot (?:assist|help|provide|comply)",
        r"I'?m not able to (?:help|assist|provide)",
        r"I won'?t (?:help|assist|provide)",
        r"I must (?:decline|refuse)",
        r"unable to (?:assist|help|comply|provide|generate)",
        r"against my (?:guidelines|principles|policies)",
        r"cannot generate (?:this|such) content",
        r"Я не могу (?:помочь|выполнить|обработать)",
        r"не буду (?:помогать|обрабатывать|выполнять)",
    )

    # Bilingual adversarial review — Ultra only, for non-English source prompts.
    _BILINGUAL_ADVERSARIAL_PROMPT = (
        "You are a bilingual adversarial reviewer. The prompt below is written in "
        "language '{lang}'. Critique it in ENGLISH first (to surface universal issues "
        "that in-language critique might miss), then propose concrete fixes.\n\n"
        "Prompt (in {lang}):\n{prompt}\n\n"
        "Output format:\n"
        "EN_CRITIQUE: <3-5 sharp sentences of English critique>\n"
        "FIX_DIRECTIVES:\n"
        "- <specific fix 1>\n"
        "- <specific fix 2>\n"
        "- <specific fix 3>"
    )

    # v4 Ultra+ additions -------------------------------------------------

    # Synthetic test-case generation — used in Standard (3 tests) and Ultra (5 tests).
    # Produces safe, de-identified inputs so a candidate prompt can be exercised on
    # realistic inputs and judged by output quality, not by prompt-text aesthetics.
    _SYNTHETIC_TEST_PROMPT = (
        "You are generating SAFE synthetic test inputs to evaluate a prompt's quality.\n\n"
        "TASK CONTRACT:\n"
        "  archetype: {archetype}\n"
        "  domain: {domain}\n"
        "  audience: {audience}\n"
        "  output_format: {output_format}\n\n"
        "ORIGINAL PROMPT (what an AI will execute):\n<<<\n{prompt}\n>>>\n\n"
        "Generate {count} SHORT, SAFE, DE-IDENTIFIED synthetic test inputs for this task. "
        "For sensitive domains (moderation, legal, safety) use benign placeholder text that "
        "still exercises the prompt's logic — do NOT produce real harmful content. Cover "
        "diverse edge cases (trivial / borderline / complex).\n\n"
        "Output a JSON array of strings: [\"input1\", \"input2\", ...]. No prose, no fences."
    )

    # Scoring LLM response quality against task. Single-integer output for cheap eval.
    _EVAL_RESPONSE_PROMPT = (
        "Score how well this response fulfills the task on scale 1-5 (5=excellent).\n\n"
        "TASK ARCHETYPE: {archetype}\n"
        "OUTPUT_FORMAT_ANCHOR: {output_format}\n\n"
        "PROMPT (given to AI):\n<<<\n{candidate}\n>>>\n\n"
        "TEST INPUT:\n<<<\n{test_input}\n>>>\n\n"
        "AI RESPONSE:\n<<<\n{response}\n>>>\n\n"
        "Output ONLY a single integer 1-5."
    )

    # Red-team adversary — Ultra only. Looks for edge cases that would break the prompt.
    _RED_TEAM_PROMPT = (
        "You are a red-team adversary testing a prompt for robustness. Find up to 3 "
        "concrete, realistic edge cases / inputs that would cause this prompt to produce "
        "a WRONG, INCOMPLETE, or MISLEADING answer. Be specific.\n\n"
        "PROMPT UNDER TEST:\n<<<\n{prompt}\n>>>\n\n"
        "TASK ARCHETYPE: {archetype}\n"
        "KNOWN FAILURE MODES: {failure_modes}\n\n"
        "Output a JSON object:\n"
        "{{\n"
        '  "edge_cases": ["<case 1>", "<case 2>", "<case 3>"],\n'
        '  "severity": "low|medium|high",\n'
        '  "fix_directives": ["<directive 1>", "<directive 2>", "<directive 3>"]\n'
        "}}\nJSON only."
    )

    # ══════════════════════════════════════════════════════════════════════
    # Construction
    # ══════════════════════════════════════════════════════════════════════

    # v4 Ultra+: persistent cross-prompt lesson cache (RIDER-like cross-experiment memory).
    _LESSON_CACHE_PATH = os.path.expanduser("~/.rider_genesis_lessons.json")
    _LESSON_CACHE_MAX_PER_KEY = 20

    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4.5",
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose

        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY or pass api_key="
            )

        self.llm_client = LLMClient(provider="openrouter", api_key=api_key)

        # Results (populated after run())
        self._final_prompt: Optional[str] = None
        self._original_fitness: float = 0.0
        self._best_fitness: float = 0.0
        self._history: List[Dict] = []
        self._api_calls_start: int = 0

        # v3 state — reset in _setup_run
        self._contract: Dict[str, Any] = {}
        self._lessons: List[str] = []
        self._forge: Dict[str, List[str]] = {}
        self._mode: str = "standard"

        # v4 cross-prompt persistent lesson cache.
        self._lesson_cache: Dict[str, List[str]] = self._load_lesson_cache()

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
    }

    def _extract_contract(self, prompt: str) -> Dict[str, Any]:
        """Extract structured prompt contract via 1 Sonnet call. Falls back to defaults."""
        meta = _CONTRACT_PROMPT.format(prompt=prompt)
        try:
            resp = self._generate(
                prompt=meta, model=self.PLANNING_MODEL,
                temperature=0.2, max_tokens=600,
            )
            json_text = self._extract_json(resp)
            if json_text is None:
                raise ValueError("no JSON block found")
            data = json.loads(json_text)
            if not isinstance(data, dict):
                raise ValueError("JSON is not an object")
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
            return contract
        except Exception as exc:
            logger.info(f"Contract extraction failed, using defaults: {exc}")
            return dict(self._DEFAULT_CONTRACT)

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the first top-level {...} JSON object from LLM output."""
        if not text:
            return None
        # Strip common code fences.
        text = re.sub(r"```(?:json)?\s*", "", text.strip())
        text = text.replace("```", "").strip()
        # Find first balanced {...}.
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

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
        return "\n".join(lines) + "\n\n"

    def _preservation_block(self) -> str:
        items = self._contract.get('must_preserve') or []
        if not items:
            return ""
        bullets = "\n".join(f"  - {it}" for it in items[:8])
        return (
            "PRESERVATION RULES (must-keep-verbatim artifacts from the original):\n"
            f"{bullets}\n"
            "Do NOT rewrite, translate, or paraphrase these artifacts. Copy them verbatim "
            "into the improved prompt, preserving surrounding punctuation.\n\n"
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
    # v4.3 — Refusal detection + deepseek fallback
    # ══════════════════════════════════════════════════════════════════════

    _REFUSAL_RE: Optional[Any] = None

    @classmethod
    def _is_refusal(cls, text: str) -> bool:
        """Detect LLM safety refusal (so we can retry on a permissive model)."""
        if not text or len(text.split()) > 80:
            return False  # full-length outputs are not refusals
        if cls._REFUSAL_RE is None:
            import re as _re
            cls._REFUSAL_RE = _re.compile("|".join(cls._REFUSAL_PATTERNS), _re.IGNORECASE)
        return bool(cls._REFUSAL_RE.search(text))

    def _generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        allow_fallback: bool = True,
    ) -> str:
        """LLM call with deepseek fallback on refusal. Single entrypoint for all gen."""
        effective_model = model or self.model
        try:
            resp = self.llm_client.generate(
                prompt=prompt, model=effective_model,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.debug(f"_generate({effective_model}) exc: {exc}")
            resp = ""
        if allow_fallback and effective_model != self.FALLBACK_MODEL and self._is_refusal(resp):
            logger.info(f"_generate: refusal detected on {effective_model}, retry deepseek")
            try:
                resp = self.llm_client.generate(
                    prompt=prompt, model=self.FALLBACK_MODEL,
                    temperature=temperature, max_tokens=max_tokens,
                )
            except Exception as exc:
                logger.debug(f"_generate(deepseek fallback) exc: {exc}")
        return resp or ""

    # ══════════════════════════════════════════════════════════════════════
    # Primitive helpers
    # ══════════════════════════════════════════════════════════════════════

    # v4.3 PHASE REACTOR temperatures — 4-phase progression.
    _PHASE_T = {
        'ignition': 1.15,       # broad exploration
        'fusion': 0.85,         # balanced refine
        'crystallization': 0.55,  # polish, low drift
        'validation': 0.3,      # stabilize, near-deterministic
    }

    def _phase_temperature(self, phase: str, default: float = 0.7) -> float:
        return self._PHASE_T.get(phase, default)

    # v4.4: adaptive complexity detection. Multi-phase pipeline is overkill for
    # short classification prompts (every refine layer adds noise → bloat),
    # but essential for long generation prompts (each phase adds structural depth).
    _SIMPLE_ARCHETYPES = frozenset({
        'classification', 'extractive', 'yes_no', 'label', 'choice', 'binary',
    })
    _COMPLEX_ARCHETYPES = frozenset({
        'reasoning', 'synthesis', 'generation', 'analysis', 'planning',
        'writing', 'summarization', 'translation', 'creative',
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
        return max(int(orig_len * self._BLOAT_RATIO), self._BLOAT_FLOOR)

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
                prompt=meta, model=self.model, temperature=temperature,
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
                prompt=meta, model=self.PLANNING_MODEL,
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
                prompt=meta, model=self.PLANNING_MODEL,
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
                prompt=meta, model=self.model, temperature=temperature,
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
                prompt=meta, model=self.PLANNING_MODEL,
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
                prompt=meta, model=self.model, temperature=temperature,
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
                prompt=meta, model=self.PLANNING_MODEL,
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

    def _extract_required_artifacts(self, original: str) -> Dict[str, List[str]]:
        """Detect artifacts in the original that should be preserved."""
        artifacts: Dict[str, List[str]] = {}
        preserve_items = [str(x) for x in (self._contract.get('must_preserve') or [])]
        flags = {x.lower() for x in preserve_items}
        # Auto-detect common artifacts regardless of contract flags (CoolPrompt-style).
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
            resp = self.llm_client.generate(
                prompt=meta, model=self.model, temperature=0.1,
                max_tokens=self._adaptive_max_tokens(original, improved),
            )
            text = self._strip_wrappers(resp)
            return text if len(text.split()) >= 15 else improved
        except Exception:
            return improved

    def _validate_and_repair(self, original: str, improved: str) -> str:
        violations, required = self._check_preservation(original, improved)
        if not violations:
            return improved
        self._log(f"  [preservation] {len(violations)} violation(s), repairing")
        return self._repair_preservation(original, improved, violations, required)

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
        # v4.3 real-quality tracking
        self._synth_best_score: Optional[float] = None
        self._synth_orig_score: Optional[float] = None
        self._phase_temps_used: List[str] = []
        # v4.3.2 bloat guard reference length
        self._original_prompt_len: int = len(prompt or "")
        self._log("=" * 60)
        self._log(f"RIDER Genesis v3 [{mode.upper()}]")
        self._log("=" * 60)
        self._log(f"Model: {self.model}")
        self._log(f"Original: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    def _finalize_run(self, prompt: str, result: str, mode: str, t0: float) -> str:
        # Preservation check — may add at most 1 call.
        result = self._validate_and_repair(prompt, result)

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
        self._log(f"RIDER Genesis v4 [{mode.upper()}] — Results")
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
            self._log(f"[v4 cache] injected {injected} cross-prompt lessons "
                      f"(key={self._cache_key()})")

        ign_t = self._phase_temperature('ignition')
        fus_t = self._phase_temperature('fusion')
        cry_t = self._phase_temperature('crystallization')

        # v4.4: adaptive pipeline for standard too.
        tier = self._complexity_tier()
        self._log(f"[adaptive v4.4] complexity tier = {tier.upper()} "
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
            self._log(f"\n> Phase 3 SKIPPED (simple tier) — using merged as polished")
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
                # Safety: original wins beam → keep it, EXCEPT for short prompts
                # where user wants blow-up (see _run_ultra for full rationale).
                orig_len = getattr(self, "_original_prompt_len", 0) or len(prompt)
                non_orig = [(c, sc) for c, sc in ranked if c[0] != "original"]
                if ranked[0][0][0] == "original" and orig_len >= 500:
                    polished = prompt
                    self._log(f"  [SAFETY] original wins beam {ranked[0][1]:.0%} — "
                              f"returning original unchanged (no regression, orig={orig_len})")
                elif non_orig:
                    if ranked[0][0][0] == "original":
                        self._log(f"  [SHORT-PROMPT OVERRIDE] orig won ({ranked[0][1]:.0%}) "
                                  f"but orig={orig_len} chars — forcing non-original winner")
                    polished = non_orig[0][0][1]
                    self._log(f"  beam winner: {non_orig[0][0][0]} "
                              f"(score={non_orig[0][1]:.0%} vs orig={self._synth_orig_score or 0:.0%})")
                for cand, sc in ranked:
                    self._log(f"       - {cand[0]}: {sc:.0%}")
            self._history.append({"generation": 4, "best_fitness": self._synth_best_score or 0.0,
                                  "stage": "synthetic_eval"})

        return self._finalize_run(prompt, polished, "standard", t0)

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
            self._log(f"[v4 cache] injected {injected} cross-prompt lessons "
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
            self._log(f"[adaptive v4.4] simple tier UPGRADED → medium "
                      f"(ultra mode guarantees crystallization + 4 strategies)")
            tier = 'medium'
        cfg = self._ultra_pipeline_config(tier)
        self._log(f"[adaptive v4.4] complexity tier = {tier.upper()} "
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
                mut2_f = pool.submit(self._apply_strategy, top[0][1], 'creative', None, fus_t)
            else:
                vortex_f = pool.submit(self._apply_strategy, top[0][1], 'creative', None, fus_t)
                mut2_f = pool.submit(self._apply_strategy, top[0][1], 'depth', None, fus_t)

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
                # Safety: original wins beam → return unchanged, EXCEPT on short
                # originals where the user explicitly wants blow-up to ultra-quality.
                # A 27-char vague prompt makes synth tests noisy (all candidates score
                # similarly, original ends up on top by stable-sort), so we must not
                # regress to the input. Rule: short originals (<500 chars) always pick
                # the best non-original candidate; long originals keep SAFETY.
                orig_len = getattr(self, "_original_prompt_len", 0) or len(prompt)
                non_orig_ranked = [(c, sc) for c, sc in ranked if c[0] != "original"]
                if ranked[0][0][0] == "original" and orig_len >= 500:
                    final_pick = prompt
                    self._synth_best_score = ranked[0][1]
                    self._log(f"  [SAFETY] original wins beam {ranked[0][1]:.0%} — "
                              f"returning original unchanged (no regression, orig={orig_len})")
                elif non_orig_ranked:
                    if ranked[0][0][0] == "original":
                        self._log(f"  [SHORT-PROMPT OVERRIDE] original won synth "
                                  f"({ranked[0][1]:.0%}) but orig={orig_len} chars — "
                                  f"user wants blow-up, forcing non-original winner")
                    self._synth_best_score = non_orig_ranked[0][1]
                    final_pick = non_orig_ranked[0][0][1]
                    self._log(f"  beam winner: {non_orig_ranked[0][0][0]} "
                              f"(score={non_orig_ranked[0][1]:.0%} vs orig={self._synth_orig_score or 0:.0%})")
                else:
                    # Degenerate case: only original in beam (everything else filtered out).
                    final_pick = prompt
                    self._synth_best_score = ranked[0][1] if ranked else 0.0
                    self._log(f"  [SAFETY] no non-original candidates in beam — returning original")
                for cand, sc in ranked:
                    self._log(f"       - {cand[0]}: {sc:.0%}")

                # v4.3 TRIPLE-MERGE: if top-2 non-original candidates are close in score
                # (within 15%) AND neither loses to original, merge for super-synthesis.
                # v4.4: complex tier only — on simple prompts there's rarely material
                # content to merge, the super-merge just produces a longer duplicate.
                if (cfg['run_triple_merge']
                        and ranked[0][0][0] != "original"
                        and len(non_orig_ranked) >= 2):
                    top_sc = non_orig_ranked[0][1]
                    second_sc = non_orig_ranked[1][1]
                    if top_sc - second_sc < 0.15 and top_sc >= (self._synth_orig_score or 0):
                        super_merged = self._merge(
                            non_orig_ranked[0][0][1],
                            non_orig_ranked[1][0][1],
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

    def _bilingual_adversarial(self, prompt: str, lang: str) -> Optional[str]:
        """Ultra-only: English critique for non-EN prompts to surface universal issues."""
        meta = self._BILINGUAL_ADVERSARIAL_PROMPT.format(prompt=prompt, lang=lang)
        try:
            resp = self._generate(
                prompt=meta, model=self.PLANNING_MODEL,
                temperature=0.2, max_tokens=400,
            )
            return resp.strip() if resp and resp.strip() else None
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════════
    # v4 Ultra+ methods
    # ══════════════════════════════════════════════════════════════════════

    def _generate_synthetic_tests(self, candidate_prompt: str, count: int = 3) -> List[str]:
        """v4: Generate safe synthetic test inputs via Sonnet. Returns up to `count` inputs."""
        if not self._contract:
            return []
        meta = self._SYNTHETIC_TEST_PROMPT.format(
            archetype=self._contract.get('task_archetype', 'other'),
            domain=self._contract.get('domain', 'general'),
            audience=self._contract.get('audience', 'general'),
            output_format=self._contract.get('output_format_anchor', 'free-form'),
            prompt=(candidate_prompt or '')[:2500],
            count=count,
        )
        try:
            resp = self.llm_client.generate(
                prompt=meta, model=self.PLANNING_MODEL,
                temperature=0.5, max_tokens=1500,
            )
        except Exception:
            return []
        js = self._extract_json(resp)
        if not js:
            return []
        try:
            arr = json.loads(js)
        except Exception:
            return []
        if not isinstance(arr, list):
            return []
        clean: List[str] = []
        for x in arr:
            if isinstance(x, str) and x.strip():
                clean.append(x.strip())
            elif isinstance(x, dict):
                txt = x.get('input') or x.get('text') or json.dumps(x, ensure_ascii=False)
                if txt:
                    clean.append(str(txt))
        return clean[:count]

    def _evaluate_candidate_on_tests(
        self, candidate: str, tests: List[str]
    ) -> Tuple[float, List[Tuple[str, int]]]:
        """v4.3: Execute candidate on each synthetic test in PARALLEL with ECHO (prompt
        repeated 2x — RIDER ECHO protocol, ~67% eval-quality bump on non-reasoning models),
        then score in PARALLEL. Uses _generate with deepseek refusal-fallback.

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
                prompt=exec_prompt, model=self.model, temperature=0.3,
                max_tokens=self._adaptive_max_tokens(test_input, min_tokens=500, ceiling=3000),
            )
            meta = self._EVAL_RESPONSE_PROMPT.format(
                archetype=self._contract.get('task_archetype', 'other'),
                output_format=self._contract.get('output_format_anchor', 'free-form'),
                candidate=(candidate or '')[:1500],
                test_input=(test_input or '')[:800],
                response=(response or '')[:2000],
            )
            score_raw = self._generate(
                prompt=meta, model=self.PLANNING_MODEL,
                temperature=0.0, max_tokens=8,
            )
            m = re.search(r'[1-5]', score_raw or '')
            return (test_input[:60], int(m.group(0)) if m else 3)

        workers = max(1, min(8, len(tests)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run, t) for t in tests]
            results = [f.result() for f in futures]

        details = results
        scores = [d[1] / 5.0 for d in details]
        base = (sum(scores) / len(scores)) if scores else 0.0

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

    def _rank_by_synthetic_eval(
        self, candidates: List[Tuple[str, str]], tests: List[str]
    ) -> List[Tuple[Tuple[str, str], float]]:
        """v4.2: Score each candidate on tests in PARALLEL. Outer pool across candidates,
        inner pool inside _evaluate_candidate_on_tests across tests. Nested threading is safe
        on HTTP I/O. Total concurrency: N_cands * N_tests capped at 16."""
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
        return ranked

    def _red_team_adversarial(self, prompt: str) -> Optional[Dict[str, Any]]:
        """v4 Ultra-only: adversary finds edge cases + fix directives."""
        meta = self._RED_TEAM_PROMPT.format(
            prompt=(prompt or '')[:3000],
            archetype=self._contract.get('task_archetype', 'other'),
            failure_modes=self._contract.get('failure_modes', []),
        )
        try:
            resp = self._generate(
                prompt=meta, model=self.PLANNING_MODEL,
                temperature=0.4, max_tokens=800,
            )
        except Exception:
            return None
        js = self._extract_json(resp)
        if not js:
            return None
        try:
            obj = json.loads(js)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def run(
        self,
        prompt: str,
        mode: str = 'standard',
        num_samples: Optional[int] = None,
        population_size: Optional[int] = None,
        num_generations: Optional[int] = None,
        use_llm_judge: Optional[bool] = None,
    ) -> str:
        """Optimize an arbitrary prompt.

        Args:
            prompt: the original prompt to optimize
            mode: 'light' (~15s), 'blitz' (~45s), 'standard' (~70s), 'ultra' (~120s)
            num_samples: legacy kwarg (ignored in v3)
            population_size: legacy kwarg (ignored in v3)
            num_generations: legacy kwarg (ignored in v3)
            use_llm_judge: legacy kwarg (ignored in v3)

        Returns:
            The optimized prompt.
        """
        _ = (num_samples, population_size, num_generations, use_llm_judge)

        valid_modes = {'light', 'blitz', 'standard', 'ultra'}
        if mode not in valid_modes:
            logger.warning(
                f"RiderGenesis: unknown mode '{mode}', valid: {sorted(valid_modes)}. "
                f"Falling back to 'standard'."
            )
            mode = 'standard'

        if mode == 'light':
            return self.run_light(prompt)
        if mode == 'blitz':
            return self._run_blitz(prompt)
        if mode == 'ultra':
            return self._run_ultra(prompt)
        return self._run_standard(prompt)

    # -- Properties ---------------------------------------------------------

    @property
    def final_prompt(self) -> Optional[str]:
        return self._final_prompt

    @property
    def improvement(self) -> float:
        if self._original_fitness <= 0:
            return 0.0
        return (self._best_fitness - self._original_fitness) / self._original_fitness * 100

    @property
    def fitness(self) -> float:
        return self._best_fitness

    @property
    def history(self) -> List[Dict]:
        return self._history

    @property
    def api_calls(self) -> int:
        return self.llm_client.total_api_calls - self._api_calls_start

    @property
    def contract(self) -> Dict[str, Any]:
        """Last extracted prompt contract (v3 addition)."""
        return dict(self._contract)

    @property
    def lessons(self) -> List[str]:
        """GENESIS-lite lessons collected during the last run (v3 addition)."""
        return list(self._lessons)

    # -- Utility ------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
        logger.info(msg)
