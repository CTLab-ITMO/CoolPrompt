"""Prompt templates for task-distribution inference and axis deduplication.

Pure text, no logic. Templates are filled via str.format(); every placeholder
is documented next to the function that fills it in task_distribution.py.
"""

from __future__ import annotations

DISTRIBUTION_REQUEST_TEMPLATE = """You are designing a compact coverage model for synthetic-data generation.

Do not solve the task.
Do not generate examples.
Do not describe every property that could apply to an example.

Your goal is to infer a SMALL set of high-value axes that are worth explicitly
controlling during synthetic-data generation.

User prompt:
{prompt}

TaskSpec:
{payload_json}

Trusted seed examples (primarily define correctness and I/O contract):
{seed_examples}

Distribution-reference examples (represent the source/train distribution; never the test set):
{reference_examples}

Infer 1-4 meaningful non-label axes from the TASK and the DISTRIBUTION-REFERENCE sample.

A good axis must satisfy ALL of the following:

1. TASK RELEVANCE
   The axis must describe variation that matters for this task, not merely a property
   that can be observed in the input.

2. COVERAGE VALUE
   Explicitly controlling this axis during generation should help prevent a meaningful
   region of task space from being systematically underrepresented.

3. WITHIN-CLASS / WITHIN-REGIME VARIATION
   The axis should usually be able to vary while the task answer, label, or primary
   semantic regime stays fixed.

   If an axis mostly acts as a proxy for the target answer, do not return it.

4. CLEAR PARTITION
   Axis values should be concrete, reasonably distinct, and usable for generation.

   Avoid vague partitions whose values overlap heavily or depend on subjective judgment.

5. GENERATION CONTROL
   The values must be actionable enough that a generator can deliberately create
   examples belonging to each value.

6. NON-COSMETIC
   Prefer semantic, structural, difficulty-related, or reasoning-relevant variation.
   Avoid superficial wording, punctuation, formatting, or arbitrary stylistic details
   unless they materially affect task difficulty or source-distribution fidelity.

7. COMPACTNESS
   Prefer a small number of strong axes over many weak or merely descriptive axes.

Before returning an axis, ask:

- Can this property vary meaningfully while the correct answer stays the same?
- Would synthetic generation plausibly collapse onto only one part of this dimension
  if the axis were not controlled?
- Would balancing or targeting this axis materially improve dataset coverage?
- Are the values mutually understandable and sufficiently distinct?
- Can a generator reliably produce examples for each value?

If the answer to these questions is mostly no, do not return the axis.

Prioritize axes such as:

- signal strength, explicitness, ambiguity, or inferential difficulty;
- semantic or structural regimes that materially change how the task must be solved;
- compositional or relational complexity;
- answerability or evidence sufficiency when relevant;
- meaningful source-distribution variation supported by the reference sample.

Treat generic context categories with caution.

For example, broad axes such as:
- personal vs social,
- immediate vs reflective,
- formal vs informal,
- concrete vs abstract,

should be returned ONLY when the reference distribution shows that the distinction is
both meaningful for the task and useful to control during generation.

Do not invent broad abstract domains merely because they are possible.

Do not infer an axis solely because the examples can be partitioned by it.

If the reference sample is dominated by concrete people/objects/actions, preserve that
regime instead of drifting toward generic motivational, philosophical, or abstract cases.

Avoid axes that are effectively:
- renamed versions of the target label;
- deterministic regroupings of the target label;
- weak proxies for the target label;
- arbitrary narrative categories;
- descriptive metadata with little effect on task difficulty or coverage.

Do NOT return input-size/concept-count/cardinality axes: the caller detects list-input
cardinality deterministically from the reference sample when possible.

{empirical_rule}

Never infer TARGET_PROPORTIONS from only a few seed examples.

Keep each axis compact, typically 2-6 values.

Axis descriptions must explain WHY the axis matters for generation coverage, not only
what the axis means.

Value descriptions must be concrete enough to guide generation and should minimize
overlap between values.

{label_rule}

Return only valid JSON matching the schema.
"""

AXIS_DEDUP_REQUEST_TEMPLATE = """You are selecting task-distribution axes before synthetic-data generation.

The goal is to keep a compact set of axes whose explicit control during generation
materially improves coverage of important task variation.

For every candidate axis, return exactly one decision: keep or drop.

KEEP a candidate axis only when:
1. it adds a genuinely independent dimension of variation; and
2. explicitly controlling that dimension would materially improve dataset coverage.

DROP a candidate axis when any of the following holds:

1. SEMANTIC REDUNDANCY

   The candidate measures essentially the same underlying property as another axis,
   even if the names or value labels differ.

2. FUNCTIONAL REDUNDANCY

   The candidate is deterministically derivable from another axis.

   This includes deterministic regroupings or coarsenings where every value of one
   axis maps to exactly one value of the candidate axis.

3. LOW COVERAGE VALUE

   The candidate may describe a real property, but explicitly controlling or balancing
   it would add little useful coverage for the task.

Do NOT drop an axis merely because it is correlated with another axis.

Use these tests:

INDEPENDENCE TEST:
Can the candidate meaningfully vary while the other axes stay fixed?

If not, and its value is determined by another axis, it is redundant.

COVERAGE TEST:
If generation ignored this axis, is there an important and plausible region of task
space that would likely be systematically underrepresented?

If yes, the axis has useful coverage value.

Examples:

- category = A / B / C
  category_group = X / Y
  where each category always maps to exactly one category_group
  -> drop
  Reason: deterministic coarsening.

- source_type = document / message
  wording_style = formal / informal
  where either style can occur for either source type
  -> not redundant.
  Keep only if controlling wording style is materially useful for task coverage.

- field_count = one / two / three_or_more
  size_bucket = small / large
  where one or two -> small and three_or_more -> large
  -> drop
  Reason: size_bucket adds no independent information.

- input_length = short / long
  ambiguity = low / high
  where both ambiguity levels occur at both lengths
  -> independent.
  Keep ambiguity if it represents meaningful task difficulty or coverage.

- surface_form = type_A / type_B
  reasoning_difficulty = easy / hard
  -> do not assume redundancy merely because one tends to predict the other.

Deterministic axes are authoritative and must never be dropped.

If two candidate axes are redundant with each other, keep the clearer, more informative,
and more useful coverage axis.

Prefer a compact set of strong axes over a larger set of weak axes.

Do not rename axes.
Do not rewrite axes.
Do not merge axes.
Do not invent new axes.
Do not modify deterministic axes.

Task information and axes:

{payload_json}

Return only valid JSON matching the schema.
"""