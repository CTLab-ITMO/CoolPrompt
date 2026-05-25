# SGR-Niche: making PE2+SGR the optimizer of choice for verifiable-constraint and reasoning tasks

**Date:** 2026-05-25
**Status:** Design (approved in brainstorming; pending spec review)

## Motivation

With proper error bars (8 seeds, N=50, gpt-4o-mini runtime), **PE2+SGR ≈ PE2**
overall (mean final 0.826 vs 0.821; not significant) and both ≫ APE (0.723).
The exciting 3-seed signals (ANLI win, subj catastrophe) were noise. A separate
result shows a stronger optimizer (gpt-4.1) does **not** produce better prompts
for the gpt-4o-mini runtime (mean Δ ≈ 0) — so the bottleneck is **signal and
search, not the proposer's intelligence**.

SGR's only mildly-consistent edges are on **verifiable-constraint** (ifeval,
+0.087, 4/8 seeds) and **multi-step reasoning** (bbh_formal_fallacies, +0.077,
5/8). Two failure modes were identified: a **meta-objective leak** (SGR's
diagnostic framing bleeding into the produced prompt — Claude-optimizer
specific) and **over-engineering on easy/saturated tasks** (verbose rewrites
that lose to PE2's simple ones).

**Goal (chosen):** establish SGR's *niche* — show it genuinely wins where
structured reasoning matters (verifiable-constraint flagship, reasoning
secondary) and route to it adaptively — rather than claiming universal
superiority.

## Non-goals

- Making SGR a strictly-better *general* optimizer (the data says it isn't, and
  optimizer capability isn't the lever).
- Changing the proposer model or adding a bigger optimizer (gpt-4.1 null result).
- Per-constraint feedback for non-verifiable tasks (no structured signal exists).

## Architecture

Three sequential phases, each independently implementable **and evaluated
against the committed 8-seed gpt-4o-mini PE2 baseline** (same 17 benchmarks,
seeds {42,123,2024,7,99,256,512,777}, N=50) before continuing. Each phase must
earn its keep or we stop. All changes are backward-compatible: scalar-metric
tasks and single-method runs behave exactly as today.

### Phase 1 — Auto-enabled per-constraint structured feedback (flagship lever)

The IFEval checker already knows which constraint failed on which example;
today the SGR proposer only sees raw failure examples (`examples_str`). We
surface an aggregated per-constraint breakdown and thread it into the diagnosis
so `FullDiagnosis.error_analyses` is grounded in real failure rates —
information PE2 structurally cannot use.

**Self-activating (no manual flag):** the SGR run duck-types the active metric.
- A metric MAY implement `failure_breakdown(outputs, targets) -> str | None`.
- `IFEvalMetric.failure_breakdown` returns a summary like
  `"length_constraints:number_words failed 7/10; detectable_format:json_format
  2/10"` (only the failing instruction ids, with counts), or `None` if every
  evaluated example passed.
- Scalar metrics (em, accuracy, rouge) do **not** implement it → treated as
  `None` → Phase 1 is a transparent no-op.

So Phase 1 engages exactly when (a) the metric is verifiable AND (b) there are
real constraint failures to report.

**Wiring:**
- The SGR proposer's `FULL_DIAGNOSIS` template gains an optional
  `{constraint_feedback}` slot. When the breakdown is `None`/empty, the slot
  renders to empty and the template reads as today.
- The SGR run computes the breakdown from the current-best prompt's evaluation
  on the train/val batch and passes it to the proposer alongside `examples_str`.

**Ablation override:** a single config knob `sgr_constraint_feedback:
auto | on | off` (default `auto`). Not for normal use — it exists so we can run
**SGR-without-feedback vs SGR-with-feedback** and measure Phase 1's
contribution. `on`/`off` force the behavior regardless of metric capability
(`on` is a no-op when the metric lacks `failure_breakdown`).

**Interfaces:**
- `failure_breakdown(self, outputs: list[str], targets: list[str]) -> str|None`
  on `IFEvalMetric` (and any future verifiable metric).
- Proposer accepts an optional `constraint_feedback: str | None` argument,
  default `None`.

**Success criterion:** SGR's ifeval advantage over PE2 becomes robust — wins
≥6/8 seeds and the paired-difference CI excludes 0 — with the
SGR-with-feedback vs SGR-without ablation showing the lift comes from the
feedback.

### Phase 2 — Regression guard (stop SGR losing on easy tasks)

Builds on the existing SGR beam search (`n_expand=4`, `backtrack=True`,
`best_val_score` tracking). Adds:
1. **Minimal/no-change edit** as a first-class `EditDecision` option the
   proposer selects when the diagnosis finds no systematic failure (start prompt
   already strong). The rewrite then keeps the prompt (or makes a minimal edit)
   instead of forcing a full verbose rewrite.
2. **Length tiebreak** — when a candidate's val score ties the current best,
   keep the shorter prompt.
3. **Best-on-val guarantee** — the returned prompt is never worse on val than
   the start prompt (revert to start if no candidate beats it).
4. **Meta-leak guard** — the rewrite instruction must not tell the runtime to
   categorize/identify/explain constraints (strip diagnostic framing from the
   produced prompt). Primarily relevant when a Claude-class optimizer is used.

**Success criterion:** on the tasks where SGR currently loses to PE2 (subj,
sst2, banking77, agnews), SGR's deficit shrinks to a tie (mean Δ ≥ −0.01) while
its niche wins (ifeval, formal_fallacies) are preserved.

### Phase 3 — Portfolio routing (deployment + niche map)

A thin wrapper that runs both PE2 and PE2+SGR, evaluates both final prompts on
the val/dev split, and returns the better one together with its method label.
Guaranteed ≥ max(PE2, SGR) by construction. Logging which method wins per
(benchmark, seed) produces the empirical **niche map** for the thesis.

**Interface:** a `portfolio` "method" in the benchmark runner that internally
runs `pe2` and `pe2_sgr` and selects by val score; the result records the
winning method.

**Success criterion:** portfolio ≥ both individual methods on every benchmark
(by construction), and the niche map shows SGR selected predominantly on
verifiable-constraint and reasoning tasks.

## Evaluation protocol (shared across phases)

- Runtime model: gpt-4o-mini (native OpenAI); optimizer: gpt-4o-mini (matches
  the committed baseline).
- Benchmarks: the existing 17; report the discriminating subset (ifeval, trec,
  agnews, rucola, subj, anli, banking77, the BBH tasks) separately from the
  ceilinged ones (svamp, sst2, gsm8k).
- Seeds: the 8 already used; N=50.
- Statistics: paired sign test / Wilcoxon of the new SGR variant vs the
  committed PE2 baseline, per benchmark, across seeds.
- Each phase is gated on its success criterion before the next begins.

## Testing

TDD per repo convention (`uv run python -m unittest`, `uv run --with flake8
flake8`):
- Phase 1: unit tests for `IFEvalMetric.failure_breakdown` (correct per-id
  counts; `None` on all-pass / scalar metrics) and for the proposer rendering
  the `{constraint_feedback}` slot (present vs empty).
- Phase 2: unit tests for the no-change/minimal `EditDecision` path, the length
  tiebreak, and the best-on-val guarantee.
- Phase 3: unit test for portfolio selection (returns the higher-val prompt and
  correct method label; ties resolved deterministically).
- Existing checker/loader/metric tests stay green.

## Risks & known limits

- **Phase 1 is verifiable-only.** The reasoning-secondary niche relies on
  existing SGR behavior plus Phases 2–3, not Phase 1.
- **N=50 dev noise** (ifeval dev = 13 examples) means even improved edges may
  stay small; higher N is an optional follow-up if a phase's effect is real but
  underpowered.
- **Backward compatibility:** all changes must be no-ops for scalar metrics and
  single-method runs; verified by the existing 8-seed runs reproducing.
- **Meta-leak guard** mainly matters for Claude-class optimizers; low priority
  if we standardize on gpt-4o-mini/gpt-4.1 optimizers.
