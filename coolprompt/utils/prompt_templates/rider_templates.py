"""Prompt templates for the RIDER Genesis Ultra optimizer."""

from __future__ import annotations


RIDER_CONTRACT_PROMPT = (
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


RIDER_STRATEGY_PROMPTS = {
    "structural": (
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
    "analytical": (
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
    "creative": (
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
    "depth": (
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
    "techniques": (
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
    "vortex": (
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


RIDER_COMPARE_PROMPT = (
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


RIDER_MERGE_PROMPT = (
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


RIDER_CONSTITUTIONAL_AUDIT_PROMPT = (
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


RIDER_REFINE_PROMPT = (
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


RIDER_PRESERVE_REPAIR_PROMPT = (
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


RIDER_QUALITY_PROMPT = (
    "Rate these two prompts for quality on a scale 1-10.\n"
    "Higher = more clear, specific, structured, effective at guiding an AI.\n\n"
    "{contract_block}"
    "PROMPT A (original):\n{prompt_a}\n\n"
    "PROMPT B (improved):\n{prompt_b}\n\n"
    "Reply EXACTLY in this format:\nA: <number>\nB: <number>"
)


RIDER_BATCH_RANK_PROMPT = (
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


RIDER_BILINGUAL_ADVERSARIAL_PROMPT = (
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


RIDER_SYNTHETIC_TEST_PROMPT = (
    "You are generating SAFE synthetic test inputs to evaluate a prompt's quality.\n\n"
    "TASK CONTRACT:\n"
    "  archetype: {archetype}\n"
    "  domain: {domain}\n"
    "  audience: {audience}\n"
    "  output_format: {output_format}\n\n"
    "HARD DOMAIN RULES:\n{domain_rules}\n\n"
    "EVALUATION DIMENSIONS:\n{quality_dimensions}\n\n"
    "ORIGINAL PROMPT (what an AI will execute):\n<<<\n{prompt}\n>>>\n\n"
    "Generate {count} SHORT, SAFE, DE-IDENTIFIED synthetic test inputs for this task. "
    "For sensitive domains (moderation, legal, safety) use benign placeholder text that "
    "still exercises the prompt's logic — do NOT produce real harmful content. Cover "
    "diverse edge cases (trivial / borderline / complex). Each string may include "
    "an 'Expected invariant checks:' sentence naming what a good answer must satisfy "
    "(schema, fields, no extra prose, level priority, quote evidence, glossary usage, etc.).\n\n"
    "Output a JSON array of strings: [\"input1\", \"input2\", ...]. No prose, no fences. "
    "If a structured-output schema is enforced, use the equivalent object "
    "{{\"tests\": [\"input1\", \"input2\", ...]}}."
)


RIDER_EVAL_RESPONSE_PROMPT = (
    "Score how well this response fulfills the task on scale 1-10 "
    "(10=elite, 7=solid but ordinary, 5=acceptable, 1=broken).\n\n"
    "TASK ARCHETYPE: {archetype}\n"
    "OUTPUT_FORMAT_ANCHOR: {output_format}\n\n"
    "DOMAIN-SPECIFIC EVALUATION DIMENSIONS:\n"
    "{quality_dimensions}\n\n"
    "If a dimension is present, it is mandatory. In particular:\n"
    "- schema_compliance: response must be parseable and use exactly the expected fields.\n"
    "- level_calibration: level/priority logic must follow the source prompt.\n"
    "- false_negative_resistance: high-risk or must-keep items must not be dropped.\n"
    "- false_positive_control: clearly safe items must not be invented as violations.\n"
    "- quote_grounding: risk/explanation must cite evidence from input text, not context-only guesswork.\n"
    "- glossary_compliance: dictionary/glossary terms must be followed unless an explicit preflight warning is required.\n"
    "- no_extra_commentary: no prose outside the required output wrapper/schema.\n\n"
    "For short creative/general prompts, do NOT give 9-10 merely for being valid. "
    "Reserve 9-10 for prompts that add a distinctive angle, non-generic constraints, "
    "clear structure, concrete evidence/detail requirements, and an explicit quality bar. "
    "A raw one-sentence prompt should usually score 3-5 unless it already contains these.\n\n"
    "PROMPT (given to AI):\n<<<\n{candidate}\n>>>\n\n"
    "TEST INPUT:\n<<<\n{test_input}\n>>>\n\n"
    "AI RESPONSE:\n<<<\n{response}\n>>>\n\n"
    "Output ONLY a single integer 1-10. If a structured-output schema is enforced, "
    "use {{\"score\": <integer 1-10>}}."
)


RIDER_RED_TEAM_PROMPT = (
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


RIDER_SAFE_CODE_PROMPT_RU = (
    "Ты senior software engineer. Выполни исходную задачу ниже, но сделай ответ "
    "производственным: сначала найди корневую причину, затем дай минимальный патч, "
    "затем тесты/проверку. Не меняй публичный API, имена функций, имена полей, "
    "сигнатуры, зависимости и поведение вне указанной ошибки.\n\n"
    "Исходная задача (вербатим, является авторитетной):\n<<<\n"
    "{original}\n"
    ">>>\n\n"
    "Формат ответа:\n"
    "1. Root cause: 1-3 предложения.\n"
    "2. Minimal patch: только нужные изменения.\n"
    "3. Tests: конкретные тест-кейсы, включая регрессию для указанной ошибки.\n"
    "4. Verification: коротко как убедиться, что фикс работает.\n"
    "Не добавляй нерелевантные рефакторы."
)


RIDER_SAFE_CODE_PROMPT_EN = (
    "You are a senior software engineer. Execute the original task below, "
    "but make the answer production-grade: identify the root cause first, "
    "then provide the minimal patch, then tests and verification. Do not change "
    "the public API, function names, field names, signatures, dependencies, "
    "or behavior outside the reported failure.\n\n"
    "Original task (verbatim, authoritative):\n<<<\n"
    "{original}\n"
    ">>>\n\n"
    "Answer format:\n"
    "1. Root cause: 1-3 concise sentences.\n"
    "2. Minimal patch: only the required code changes.\n"
    "3. Tests: concrete cases, including a regression test for the reported error.\n"
    "4. Verification: how to confirm the fix works.\n"
    "Avoid unrelated refactors."
)


RIDER_SAFE_CLASSIFICATION_PROMPT_RU = (
    "Ты строгий production-классификатор. Сохрани исходную JSON-схему, уровни, "
    "категории и юридические правила без изменений; улучши только порядок "
    "применения правил, калибровку уровней и защиту от ложных отрицаний.\n\n"
    "Original task (verbatim, authoritative):\n<<<\n"
    "{original}\n"
    ">>>\n\n"
    "Перед ответом внутренне проверь: схема валидна, поля не потеряны, "
    "Level 0 не повышен без правила, high-risk элементы не пропущены. "
    "Выведи только требуемый JSON."
    "\n\nHard preservation gate: keep the source prompt's exclusive output contract, "
    "priority order, level escalation rules, caps, and exceptions exactly. If the source "
    "contains Russia/Ukraine/Russian-side escalation, preserve the one-level raise rule, "
    "the Level 0 exception, and the Level 3 cap exactly as written."
)


RIDER_SAFE_CLASSIFICATION_PROMPT_EN = (
    "You are a strict production classifier. Preserve the original JSON schema, "
    "levels, categories, and legal rules unchanged; improve only rule ordering, "
    "level calibration, and false-negative resistance.\n\n"
    "Original task (verbatim, authoritative):\n<<<\n"
    "{original}\n"
    ">>>\n\n"
    "Before answering, internally verify: schema is valid, fields are preserved, "
    "Level 0 is not raised without a rule, and high-risk items are not missed. "
    "Output only the required JSON."
    "\n\nHard preservation gate: keep the source prompt's exclusive output contract, "
    "priority order, level escalation rules, caps, and exceptions exactly. If the source "
    "contains Russia/Ukraine/Russian-side escalation, preserve the one-level raise rule, "
    "the Level 0 exception, and the Level 3 cap exactly as written."
)


RIDER_SAFE_TRANSLATION_PROMPT_RU = (
    "Ты литературный переводчик и редактор. Сохрани все placeholders, XML-теги, "
    "словарь и формат вывода из исходной инструкции; усили качество перевода за счет "
    "точной передачи смысла, ритма, регистра и авторской шероховатости.\n\n"
    "Original task (verbatim, authoritative):\n<<<\n"
    "{original}\n"
    ">>>"
)


RIDER_SAFE_TRANSLATION_PROMPT_EN = (
    "You are a literary translator and editor. Preserve every placeholder, XML tag, "
    "glossary rule, and output wrapper from the original instruction; improve translation "
    "quality through fidelity, rhythm, register, and authorial texture.\n\n"
    "Original task (verbatim, authoritative):\n<<<\n"
    "{original}\n"
    ">>>"
)


RIDER_SAFE_GENERIC_PROMPT_RU = (
    "Улучши выполнение исходной задачи: уточни роль, критерии качества, формат ответа, "
    "ограничения и типичные ошибки, но не меняй тему, язык, входные артефакты и требуемый результат.\n\n"
    "Исходная задача (вербатим, является авторитетной):\n<<<\n"
    "{original}\n"
    ">>>\n\n"
    "Сначала следуй исходной задаче, затем применяй усиленные критерии качества. "
    "Не добавляй лишний метакомментарий, если он не запрошен."
)


RIDER_SAFE_GENERIC_PROMPT_EN = (
    "Improve execution of the original task by making the role, quality criteria, "
    "answer format, constraints, and common failure modes explicit, while preserving "
    "the topic, language, input artifacts, and requested output.\n\n"
    "Original task (verbatim, authoritative):\n<<<\n"
    "{original}\n"
    ">>>\n\n"
    "Follow the original task first, then apply the strengthened quality criteria. "
    "Do not add extra meta-commentary unless requested."
)
