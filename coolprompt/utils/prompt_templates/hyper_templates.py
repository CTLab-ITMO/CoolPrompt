from dataclasses import dataclass, field
from typing import List, Optional, Union


# -------- Section-targeted recommendation --------

GENERAL_SECTION = "general"


@dataclass
class Recommendation:
    """A single feedback recommendation targeted at a specific section
    of the resulting prompt (or 'general' if it applies broadly).

    Attributes:
        section: Name of the target section (e.g. 'Role', 'Output requirements')
            or 'general' for cross-cutting recommendations.
        text: The recommendation text (concise, action-verb-leading).
        weight: Group size after aggregation. Defaults to 1 (raw rec from
            a single failure). Used for conflict resolution and sorting.
    """
    section: str
    text: str
    weight: int = 1


# -------- Meta-prompt builder (HyPER Light / MetaPromptOptimizer) --------

TARGET_PROMPT_FORMS = ["instructional "]


SIMPLE_HYPOTHETICAL_PROMPT = (
    "Write a new {target_prompt_form}prompt that will solve the user query effectively."
)

META_INFO_SECTION = "Task-related meta-information which you must mention generating a new prompt:\n<meta_info>\n{meta_info_content}\n</meta_info>\n"

# Section name constants
SECTION_ROLE = "role"
SECTION_PROMPT_STRUCTURE = "prompt_structure"
SECTION_RECOMMENDATIONS = "recommendations"
SECTION_CONSTRAINTS = "constraints"
SECTION_OUTPUT_FORMAT = "output_format"

META_PROMPT_SECTIONS = (
    SECTION_ROLE,
    SECTION_PROMPT_STRUCTURE,
    SECTION_RECOMMENDATIONS,
    SECTION_CONSTRAINTS,
    SECTION_OUTPUT_FORMAT,
)

ROLE_LINE = "You are an expert prompt engineer.\n"
TASK_SECTION_TEMPLATE = (
    "Your only task is to write a new {target_prompt_form}prompt that will "
    "solve the user query as effectively as possible.\n"
    "Do not answer the user query directly; only produce the new prompt.\n\n"
)

PROMPT_STRUCTURE_SECTION_TEMPLATE = (
    "### STRUCTURE OF THE PROMPT YOU MUST PRODUCE\n"
    "The prompt you write MUST be structured into the following sections, "
    "in this exact order, and each section must follow its guidelines:\n"
    "{sections_with_guidelines}\n\n"
)

CONSTRAINTS_SECTION_TEMPLATE = "### HARD CONSTRAINTS\n{constraints_list}\n\n"

RECOMMENDATIONS_SECTION_TEMPLATE = (
    "### RECOMMENDATIONS\n"
    "Use these recommendations when writing the new prompt, "
    "based on analysis of previous generations. Each recommendation "
    "targets a specific section of the resulting prompt, or applies generally.\n"
    "{recommendations_grouped}\n\n"
)

BASE_OUTPUT_FORMAT_SECTION = (
    "### YOUR RESPONSE FORMAT\n"
    "Return ONLY the resulting prompt, wrapped in the following XML tags:\n"
    "<result_prompt>\n"
    "  ...your resulting prompt here...\n"
    "</result_prompt>\n"
    "Do not include any explanations or additional text outside this XML element.\n\n"
)

MARKDOWN_OUTPUT_REQUIREMENTS = (
    "#### Markdown formatting for the resulting prompt\n"
    "- Write the entire prompt inside <result_prompt> using valid Markdown.\n"
    "- Use headings (e.g., `#`, `##`) for major sections of the prompt.\n"
    "- Use bulleted lists (e.g., `-` or `*`) for enumerations and checklists.\n"
    "- Preserve any code or pseudo-code using fenced code blocks (``` ... ```).\n"
    "- Do not introduce any additional formatting beyond what is necessary to make "
    "the prompt clear and well-structured."
)

HYPE_META_PROMPT_TEMPLATE = (
    "{role_section}"
    "{prompt_structure_section}"
    "{recommendations_section}"
    "{constraints_section}"
    "{output_format_section}"
)


@dataclass
class PromptSectionSpec:
    """Name and description for one generated prompt section."""

    name: str
    description: str


@dataclass
class MetaPromptConfig:
    """Configuration for building HyPER meta-prompts."""

    target_prompt_form: str = "instructional "
    require_markdown_prompt: bool = True
    include_role: bool = True
    section_names: List[str] = field(
        default_factory=lambda: [
            "Role",
            "Task context",
            "Instructions",
            "Output requirements",
        ]
    )
    section_specs: List[PromptSectionSpec] = field(
        default_factory=lambda: [
            PromptSectionSpec(
                name="Role",
                description=(
                    "Briefly define the assistant's role and expertise "
                    "relevant to the user query."
                ),
            ),
            PromptSectionSpec(
                name="Task context",
                description=(
                    "Briefly describe the task type and domain using query and meta-info. "
                    "Do not quote the user's provided data verbatim."
                ),
            ),
            PromptSectionSpec(
                name="Instructions",
                description=(
                    "Main part - instructions the assistant must follow "
                    "to solve the user's query while respecting constraints."
                ),
            ),
            PromptSectionSpec(
                name="Output requirements",
                description=(
                    "Clearly specify the desired tone "
                    "and the required level of detail for the assistant's answer. "
                    "If the user explicitly requests a particular output format or provides "
                    "an example response, restate that format and include the example verbatim, "
                    "without inventing any additional formatting or examples. Do not introduce any output format or examples that the user did not mention."
                ),
            ),
        ]
    )
    constraints: List[str] = field(
        default_factory=lambda: [
            "Preserve the language of the user's query.",
            "Preserve all code snippets, inline code, technical terms and special formatting.",
            "Do not remove or alter any explicit formatting instructions from the user.",
            "Do not change numerical values, units, or identifiers.",
        ]
    )
    recommendations: List["Recommendation"] = field(default_factory=list)
    output_format_section: Optional[str] = None
    _cached_sections: dict = field(default_factory=dict, repr=False)


class MetaPromptBuilder:
    """
    Builder for structured meta-prompts (single-shot optimization).

    Constructs meta-prompts from configurable sections. Uses a caching strategy:
    - Static sections (role, prompt_structure, output_format) are cached on init
      and rebuilt only when their config changes.
    - Dynamic sections (recommendations, constraints) are stored as lists in config
      and built on-demand during build_meta_prompt().

    Typical usage:
        builder = MetaPromptBuilder()
        meta_prompt = builder.build_meta_prompt()

        # Update a section
        builder.set_section(SECTION_RECOMMENDATIONS, ["Be concise", "Use examples"])
        meta_prompt = builder.build_meta_prompt()
    """

    def __init__(self, config: MetaPromptConfig | None = None) -> None:
        """Initialize the builder and cache static sections."""
        self.config = config or MetaPromptConfig()
        self._cache_all_sections()

    def _cache_all_sections(self) -> None:
        """Cache static sections."""
        self.config._cached_sections = {
            SECTION_ROLE: self.build_role_section(),
            SECTION_PROMPT_STRUCTURE: self.build_prompt_structure_section(),
            SECTION_OUTPUT_FORMAT: self.build_output_format_section(),
        }

    def get_cached_section(self, name: str) -> Optional[str]:
        """Return a cached section by name (role, prompt_structure, output_format)."""
        return self.config._cached_sections.get(name)

    def get_section(self, name: str) -> Union[str, List[str], List[Recommendation], None]:
        """Return section value by name.

        - recommendations → List[Recommendation]
        - constraints → List[str]
        - others → cached str
        """
        if name not in META_PROMPT_SECTIONS:
            raise ValueError(
                f"Unknown section: {name}. Expected: {META_PROMPT_SECTIONS}"
            )
        if name == SECTION_RECOMMENDATIONS:
            return list(self.config.recommendations)
        if name == SECTION_CONSTRAINTS:
            return list(self.config.constraints)
        return self.get_cached_section(name)

    def set_section(self, name: str, value: Union[str, List[str], List[Recommendation]]) -> None:
        """Update a section value. Only recommendations, constraints, and output_format are settable.

        recommendations accepts both List[str] (legacy, treated as 'general')
        and List[Recommendation] (preferred).
        """
        if name not in META_PROMPT_SECTIONS:
            raise ValueError(
                f"Unknown section: {name}. Expected: {META_PROMPT_SECTIONS}"
            )
        if name == SECTION_RECOMMENDATIONS:
            if not isinstance(value, list):
                raise ValueError("recommendations must be a list")
            normalized: List[Recommendation] = []
            for item in value:
                if isinstance(item, Recommendation):
                    normalized.append(item)
                elif isinstance(item, str):
                    normalized.append(Recommendation(section=GENERAL_SECTION, text=item))
                else:
                    raise ValueError(
                        f"recommendation must be str or Recommendation, got {type(item).__name__}"
                    )
            self.config.recommendations = normalized
        elif name == SECTION_CONSTRAINTS:
            if not isinstance(value, list):
                raise ValueError("constraints must be a list of strings")
            self.config.constraints = list(value)
        elif name == SECTION_OUTPUT_FORMAT:
            if not isinstance(value, str):
                raise ValueError("output_format must be a string")
            self.config.output_format_section = value
            self.config._cached_sections[SECTION_OUTPUT_FORMAT] = (
                self.build_output_format_section()
            )
        else:
            raise ValueError(f"Section '{name}' is read-only or not directly settable")

    def build_role_section(self, include_role: bool | None = None) -> str:
        """
        Build the opening section with role definition and task description.

        Contains two parts:
        - Role line (optional, controlled by include_role)
        - Task description: explains what the model should do (always included)

        The task description uses target_prompt_form to specify the type of prompt to generate
        (e.g., "instructional").
        """
        include_role = (
            include_role if include_role is not None else self.config.include_role
        )
        form = self.config.target_prompt_form or ""
        task_part = TASK_SECTION_TEMPLATE.format(target_prompt_form=form)
        if include_role:
            return ROLE_LINE + task_part
        return task_part

    def build_prompt_structure_section(
        self,
        specs: list[PromptSectionSpec] | None = None,
    ) -> str:
        """Build the prompt structure guidelines section."""
        specs = specs or self.config.section_specs
        lines = [f"- [{spec.name}] {spec.description}" for spec in specs]
        return (
            PROMPT_STRUCTURE_SECTION_TEMPLATE.format(
                sections_with_guidelines="\n".join(lines)
            )
            if lines
            else ""
        )

    def build_recommendations_section(
        self,
        recommendations: List[Recommendation] | List[str] | None = None,
    ) -> str:
        """Build the recommendations section, grouped by target section.

        Accepts both List[Recommendation] (preferred) and List[str] (legacy,
        treated as 'general'). Empty string if no recommendations.
        """
        recs = (
            recommendations
            if recommendations is not None
            else self.config.recommendations
        )
        if not recs:
            return ""

        # Normalize to List[Recommendation]
        normalized: List[Recommendation] = []
        for r in recs:
            if isinstance(r, Recommendation):
                normalized.append(r)
            elif isinstance(r, str):
                normalized.append(Recommendation(section=GENERAL_SECTION, text=r))

        # Group by section
        by_section: dict[str, List[str]] = {}
        for r in normalized:
            by_section.setdefault(r.section, []).append(r.text)

        # Render: known sections (in their declared order) first, then 'general',
        # then any unknown sections (defensive).
        section_order = [spec.name for spec in self.config.section_specs]
        seen_sections: set[str] = set()
        blocks: List[str] = []

        for name in section_order:
            if name in by_section:
                items = "\n".join(f"- {t}" for t in by_section[name])
                blocks.append(f"For section [{name}]:\n{items}")
                seen_sections.add(name)

        if GENERAL_SECTION in by_section:
            items = "\n".join(f"- {t}" for t in by_section[GENERAL_SECTION])
            blocks.append(f"General:\n{items}")
            seen_sections.add(GENERAL_SECTION)

        for name, texts in by_section.items():
            if name not in seen_sections:
                items = "\n".join(f"- {t}" for t in texts)
                blocks.append(f"For section [{name}]:\n{items}")

        grouped = "\n\n".join(blocks)
        return RECOMMENDATIONS_SECTION_TEMPLATE.format(recommendations_grouped=grouped)

    def build_constraints_section(
        self,
        constraints: List[str] | None = None,
    ) -> str:
        """Build the hard constraints section."""
        constraints = constraints or self.config.constraints
        if not constraints:
            return ""
        lines = "\n".join(f"- {c}" for c in constraints)
        return CONSTRAINTS_SECTION_TEMPLATE.format(constraints_list=lines)

    def build_output_format_section(self) -> str:
        """Build the output format section (with optional markdown requirements)."""
        section = self.config.output_format_section or BASE_OUTPUT_FORMAT_SECTION
        if self.config.require_markdown_prompt:
            section = section + MARKDOWN_OUTPUT_REQUIREMENTS
        return section

    def build_meta_prompt(
        self,
        *,
        target_prompt_form: str | None = None,
        section_specs: List[PromptSectionSpec] | None = None,
        recommendations: List[Recommendation] | List[str] | None = None,
        constraints: List[str] | None = None,
        output_format_section: str | None = None,
        include_role: bool | None = None,
    ) -> str:
        """
        Build the complete meta-prompt from all sections.

        Args can override config values for this build only.
        """
        # Apply overrides to config
        if target_prompt_form is not None:
            self.config.target_prompt_form = target_prompt_form
        if section_specs is not None:
            self.config.section_specs = section_specs
        if recommendations is not None:
            self.config.recommendations = recommendations
        if constraints is not None:
            self.config.constraints = constraints
        if output_format_section is not None:
            self.config.output_format_section = output_format_section
        if include_role is not None:
            self.config.include_role = include_role

        return HYPE_META_PROMPT_TEMPLATE.format(
            role_section=self.build_role_section(include_role=include_role),
            prompt_structure_section=self.build_prompt_structure_section(),
            recommendations_section=self.build_recommendations_section(
                recommendations=recommendations
            ),
            constraints_section=self.build_constraints_section(),
            output_format_section=self.build_output_format_section(),
        )


# -------- HyPER Feedback module templates --------

CONTRASTIVE_FEEDBACK_PROMPT = """You are an expert prompt engineer.

Two prompts were evaluated on the SAME task. The FAILED prompt scored worse; the \
SUCCEEDED prompt scored better. Identify what the SUCCEEDED prompt did differently \
and translate that insight into a recommendation that adapts the winning aspect to \
the FAILED prompt.

FAILED prompt:
<failed_prompt>
{failed_prompt}
</failed_prompt>

FAILED answer (raw):
<failed_answer_raw>
{failed_answer_raw}
</failed_answer_raw>

FAILED answer (parsed): {failed_answer_parsed}
FAILED metric value: {failed_score}

SUCCEEDED prompt:
<succeeded_prompt>
{succeeded_prompt}
</succeeded_prompt>

SUCCEEDED answer (raw{truncation_note}):
<succeeded_answer_raw>
{succeeded_answer_raw}
</succeeded_answer_raw>

SUCCEEDED answer (parsed): {succeeded_answer_parsed}
SUCCEEDED metric value: {succeeded_score}

Same task both prompts solved:
<task>
{instance}
</task>

Correct answer:
<ground_truth>
{ground_truth}
</ground_truth>

The resulting prompt is structured into the following sections:
{section_descriptions}

TASK:
1. Identify the KEY DIFFERENCE between the two prompts that explains the success/failure.
2. Pick a target section for your recommendation:
   - Choose a SPECIFIC section name only if the issue clearly maps to it.
   - Choose "general" if the recommendation applies broadly across sections, \
or if you are not sure.
   - Prefer "general" when in doubt.
3. Write ONE concise recommendation that adapts the winning aspect to the failed prompt.
   - Style: starts with an action verb, max 20-25 words, no task-specific details, \
no meta-comments ("as before", "similar to").
   - Prefer concrete input/output constraints, decision rules, or verification steps.
   - Do NOT recommend broad style changes such as "be more creative", "use richer context",
     "use complex structure", "add depth", or "be more descriptive" unless the
     successful prompt clearly enforced a required input/output constraint and your
     recommendation names that constraint.

For constrained generation tasks where the input supplies required items, prefer
recommendations such as:
- "Require using every provided input item in one concise grammatical sentence."
- "Require preserving the concrete relationship among the provided input items."

Avoid recommendations such as:
- "Encourage more creative sentence structures."
- "Add emotional or narrative context."
- "Use at least three provided words." unless partial use is explicitly allowed.

Output ONLY a single valid JSON object in this exact format:
{{"section": "<one of the section names listed above, or general>", "text": "<recommendation text>"}}
"""


FEEDBACK_PROMPT_TEMPLATE = """You are an expert prompt engineer.

The prompt was evaluated on a benchmark task and failed on this example. You will be given the prompt, the failed example and the correct answer.

Prompt:
<prompt>
{prompt}
</prompt>

Failed task:
<failed_task>
{instance}
</failed_task>

Model answer (raw):
<model_answer>
{model_answer}
</model_answer>

Model answer (parsed):
<model_answer_parsed>
{model_answer_parsed}
</model_answer_parsed>

Metric value: {metric_value}

Correct answer:
<ground_truth>
{ground_truth}
</ground_truth>

The resulting prompt is structured into the following sections:
{section_descriptions}

TASK:
1. Identify the prompt-level reason this failure happened: what is missing,
   underspecified, or misleading in the current prompt that could cause the
   model to make this kind of mistake?
2. Pick a target section for your recommendation:
   - Choose a SPECIFIC section name only if the issue clearly maps to that section.
   - Choose "general" if the recommendation applies broadly across sections, or if you are not sure which single section to pick.
   - Prefer a specific section ONLY when you are confident; otherwise pick "general".
3. Write ONE concise recommendation (max 20-25 words, starts with an action verb, no meta-comments like "as before" or "similar to").

**CRITICAL** — your recommendation must apply broadly to OTHER failures of the same task class, NOT just to this one specific example. Each of the following is too narrow and would NOT be acceptable:

- "Use the compound interest formula correctly"
  (cites a specific named formula — narrow to one math subdomain)
- "Identify the growth of the urban area"
  (cites a phrase from one specific input passage)
- "Specify the color used in pcLEDs method"
  (cites a unique technical entity from one example)
- "Calculate Jeff's total stars after the purchase"
  (cites named entities from one specific scenario)
- "Detect anger when the context expresses workplace criticism"
  (cites a specific contextual scenario from one example)


Avoid both extremes:
- Too narrow: names a formula, entity, scenario, or subtype from this one failure.
- Too vague: says only "be clearer", "be more specific", "improve accuracy", or
  "ensure correctness" without naming a concrete prompt change.

Good recommendations should name an actionable prompt property, such as:
- require step-by-step calculation with intermediate checks;
- require extracting the exact answer phrase from context;
- require choosing only from the provided label set;

Do not infer that a prompt needs more creativity, complexity, narrative detail,
emotional context, or richer prose unless the failure clearly shows that a required
input/output constraint was missed and the recommendation names that constraint.
For constrained generation from required input items:
- GOOD: "Require using every provided input item in one concise grammatical sentence."
- GOOD: "Require preserving the concrete relationship among the provided input items."
- BAD: "Encourage more creative sentence structures."
- BAD: "Add emotional or narrative context."
- BAD: "Use at least three provided words." unless partial use is explicitly allowed.

Output ONLY a single valid JSON object in this exact format:
{{"section": "<one of the section names listed above, or general>", "text": "<recommendation text>"}}

Examples of valid output:
{{"section": "Output requirements", "text": "Specify the required final-answer wrapper."}}
{{"section": "Instructions", "text": "Require step-by-step reasoning before producing the answer."}}
{{"section": "general", "text": "Use clear, unambiguous language across the entire prompt."}}

Recommendation:
"""

RECOMMENDATIONS_GROUP_PROMPT = """Group these prompt-improvement recommendations by semantic similarity.

Items (id and text):
{items_json}

Rules:
- Group items so that each group contains semantically similar recommendations.
- Items addressing different aspects must go to different groups.
- Singleton groups are allowed if an item has no similar peer.
- Each id MUST appear exactly once across all groups.

Output ONLY a JSON array of groups, where each group is an array of item ids.
Example: [[0, 2], [1], [3, 4, 5]]
"""


SECTION_GROUPS_FILTER_PROMPT = """You are given groups of similar prompt-improvement recommendations, all targeting the same section: "{section_name}".

Each group has semantically similar recommendations. Group `weight` equals the number of original recommendations in it.

Groups:
{groups_json}

YOUR TASK:
1. SYNTHESIZE: For every group, produce ONE concise recommendation that captures the essence of its members.
   - Style: starts with an action verb, max 20-25 words, no task-specific details, no meta-comments ("as before", "similar to").
   - For singleton groups (weight=1) you may keep the original wording or rewrite if it improves clarity.
2. RESOLVE CONFLICTS: identify pairs of groups whose recommendations are MUTUALLY CONTRADICTORY (e.g., "use JSON" vs "use plain text", "add descriptive context" vs "avoid extra context"). Overlap or similarity is NOT a conflict.
   - For each contradictory pair, KEEP the group with the larger weight and DROP the other.
   - If weights are equal, keep the one you judge most useful for prompt improvement.
3. RANK AND PRUNE: return AT MOST 3 synthesized recommendations for this section.
   - Prioritize concrete input constraints, output constraints, decision rules, and verification steps.
   - Drop lower-signal tone/style/quality preferences even when they are not direct contradictions.
   - Prefer recommendations that remove ambiguity over recommendations that add complexity.
   - Do not keep recommendations that depend on seeing a provided/correct/reference answer
     at inference time. If the required output format is explicitly stated or
     unambiguous from the task spec, rewrite them to require using that format.
     Otherwise, rewrite only to normalize the final answer if that preserves a
     useful prompt behavior.
4. PRESERVE: non-contradictory groups may survive only if they are among the strongest concrete prompt changes.

Output ONLY a single JSON object in this exact format (nothing else):
{{
  "synthesized": [
    {{"text": "<synthesized recommendation>", "weight": <int>}},
    ...
  ]
}}

Example
Input groups for section "Output requirements":
[
  {{"weight": 3, "members": ["use the required final-answer wrapper", "wrap the final answer as requested", "follow the specified answer wrapper"]}},
  {{"weight": 2, "members": ["use JSON format", "format response as JSON"]}},
  {{"weight": 1, "members": ["return plain text without any formatting"]}}
]

Expected output:
{{
  "synthesized": [
    {{"text": "Use the required final-answer wrapper.", "weight": 3}},
    {{"text": "Format the response as JSON.", "weight": 2}}
  ]
}}
(The "plain text" group was dropped — it directly contradicts the JSON group, and has smaller weight.)
"""

# Backward-compatible names (deprecated; prefer RECOMMENDATIONS_GROUP_PROMPT / SECTION_GROUPS_FILTER_PROMPT).
LLM_CLUSTER_PROMPT = RECOMMENDATIONS_GROUP_PROMPT
SECTION_CLUSTER_FILTER_PROMPT = SECTION_GROUPS_FILTER_PROMPT

DROP_INSTANCE_LEAK_PROMPT = """Audit the following prompt-improvement recommendations for overfitting to specific examples.

Problem description:
<problem_description>
{problem_description}
</problem_description>

For each recommendation, return one verdict:

- KEEP — the recommendation is broad, actionable, and applies to the problem type above.
- REWRITE — it contains a useful prompt-level change, but mentions a narrow subtype, formula, entity, scenario, dataset-specific phrase, or reference-answer leak. Rewrite it in a general form.
- DROP — after removing narrow details, no useful actionable change remains, or the recommendation is only vague/general.

Rules:
1. Judge every recommendation relative to the problem description. Details that are narrow for a general task may be valid for a narrow task if they are part of the problem description.
2. Keep recommendations that are already broad and actionable.
3. When rewriting, preserve the concrete prompt behavior, but remove named entities, formulas, scenarios, dataset-specific phrases, and narrow subdomains unless they are part of the problem description.
4. Drop vague platitudes such as "be clearer", "ensure accuracy", "consider context", "capture nuance", "be creative", or similar, unless they specify an observable prompt behavior.
5. Drop recommendations that only restate the task or desired quality without telling the prompt to do something testable.
6. Drop style-only recommendations unless they enforce an input constraint, output constraint, decision rule, or verification step.
7. Recommendations must work at inference time, when the correct/reference answer is unavailable. Do not keep or write recommendations that rely on comparing with the reference answer.
8. For output-format leaks:
   - if the task explicitly specifies a format, require that format;
   - if no format is specified, do not invent one;
   - if only answer cleanliness is useful, require normalizing the final answer to the requested value type or granularity.

Examples:
- KEEP: "Decompose multi-step problems into atomic operations."
- KEEP: "Extract the answer phrase verbatim from the context."
- REWRITE: "Use the compound interest formula correctly."
  → "Identify the relevant mathematical relationship before calculating."
- REWRITE: "Match the exact format of the provided correct answer."
  → "Use the required output format when it is specified by the task."
- DROP: "Compare the result with the correct answer before responding."
- DROP: "Calculate Jeff's total stars after the purchase."

Recommendations to audit:
{recommendations_json}

Output ONLY a single JSON object:
{{"verdicts": [{{"verdict": "KEEP" | "REWRITE" | "DROP", "text": "<rewritten text only for REWRITE>"}}]}}

The order of verdicts must match the input. Return a verdict for every item.
For KEEP and DROP, omit "text" or set it to an empty string.
"""


MEMORY_COMPATIBILITY_PROMPT = """You are checking whether a new prompt-improvement
recommendation conflicts with recommendations that were previously accepted
because they produced validation improvements.

Existing accepted recommendations:
{memory_json}

New candidate recommendation:
{candidate_json}

Your task is ONLY to identify:
1. direct conflicts between the new recommendation and existing accepted
   recommendations;
2. semantic duplicates/redundant recommendations already covered by memory.

Definitions:
- A DIRECT CONFLICT means the two recommendations cannot both be followed in
  the same prompt behavior, e.g. "return only a short label" vs "explain the
  reasoning step by step", "use JSON" vs "use plain text", "avoid extra
  context" vs "add explanatory context".
- Similarity, overlap, redundancy, or different levels of specificity are NOT
  conflicts.
- A SEMANTIC DUPLICATE means the new recommendation asks for essentially the
  same prompt behavior as an existing recommendation, even if phrased with
  different words. Example: "use every provided input item in one concise
  grammatical sentence" duplicates "use all provided words in a single coherent
  sentence".
- Do not decide which recommendation is better. Existing recommendations have
  stronger prior evidence because they were validation-accepted, but the new
  recommendation will be tested separately by validation after replacing any
  conflicting old recommendations. Duplicate recommendations will be skipped
  without validation because they do not add a new prompt behavior.

Return the zero-based indices of existing recommendations that directly conflict
with the new recommendation, and the zero-based indices of existing
recommendations that semantically duplicate it.

Output ONLY a single JSON object in this exact format:
{{"conflicting_indices": [<int>, ...], "duplicate_indices": [<int>, ...], "reason": "<brief reason>"}}
"""


MERGE_SEMANTIC_DUPLICATE_PROMPT = """Merge two semantically duplicate
prompt-improvement recommendations into one stronger recommendation.

Existing accepted recommendation:
{existing_json}

New semantically duplicate recommendation:
{candidate_json}

Task:
- Produce ONE concise recommendation that preserves the shared prompt behavior.
- Keep useful extra constraints from either recommendation only if they are
  broadly applicable and do not add task-specific details.
- Do not make the recommendation more complex than necessary.
- Style: starts with an action verb, max 20-25 words, no meta-comments.

Output ONLY a single JSON object in this exact format:
{{"section": "<section>", "text": "<merged recommendation>"}}
"""


PARAPHRASE_PROMPT = """Generate an alternative version of the following structured prompt.

HARD REQUIREMENTS:
- PRESERVE THE SECTION STRUCTURE EXACTLY: keep the same section headings (e.g. \
'# Role', '# Task context', '# Instructions', '# Output requirements', or whatever \
headings the original uses) in the SAME ORDER. Do not add new sections, do not \
remove existing sections, do not rename headings.
- Within each section, REWRITE THE BODY SUBSTANTIALLY: use different words, \
different sentence structure, different voice (active/passive), reorder ideas where \
natural. Aim for noticeable rewording, not a swap of a few synonyms.
- Preserve the core MEANING and INTENT of every section, the original LANGUAGE, \
and any code/inline-code/identifiers/numerical values.
- Per-section length within ±20% of the original.
- If the original has no clear section headings, paraphrase the whole text following \
the same rewording standards.

OUTPUT FORMAT:
- Output ONLY the alternative prompt itself.
- No surrounding commentary. No XML tags. No quoting. No leading/trailing labels \
like "Alternative prompt:".

Original prompt:
{prompt}
"""


FILTER_RECOMMENDATIONS_PROMPT = """You have a list of recommendations for prompt improvement:

{recommendations}

TASK:
1. Group them into conceptual clusters (similar ideas).
2. For each cluster, **synthesize a single, new recommendation** that captures the essence of all items in that cluster. Do not just copy an existing one.
3. Rank clusters by size (largest first). If some clusters conflict - drop the less ones.
4. Output ONLY a JSON array of the synthesized recommendations, in rank order.

GOOD EXAMPLES:
Input: ["step-by-step", "break down calc", "don't show work", "format clearly"]
Correct output: ["Require detailed step-by-step reasoning with calculations", "Specify the desired output format explicitly"]
Why good:
- Captured main ideas of reasoning cluster into 1 strong rec
- Didn't loose cluster from "format clearly"
- Resolved conflict: "don't show work" is less frequent recommendation, so its cluster was dropped

BAD EXAMPLES:
Input: ["Focus on clarifying the output format requirements",
        "Add examples of expected responses to the prompt",
        "Make sure to specify exact sentiment labels",
        "Include examples to avoid confusion with similar labels",
        "Focus on tone analysis in the text",
        "Clarify what constitutes positive vs negative",
        "Add examples of positive responses",
        "Similar to previous - add more examples"]
Wrong output: ["Similar to previous - add more examples", "Add examples of positive responses", "Make sure to specify exact sentiment labels", "Focus on tone analysis in the text"]
Why bad:
- "Similar to previous" = meta-trash
- No synthesis of 6+ example recs into 1 strong rec, uses only existing recommendations
- Two different recommendations with a similiar intent: adding examples (duplicates)
"""


HypeMetaPromptBuilder = MetaPromptBuilder
HypeMetaPromptConfig = MetaPromptConfig
