# PE2+SGR on IFEval: regression analysis

**Date:** 2026-05-25
**Data:** single-seed runs, N=50 (IFEval dev split = 13 examples), train_steps=3.
Sources: `logs/native_claude_haiku.json`, `logs/native_gpt4omini.json`,
`logs/ladder_qwen{8b,14b,32b}.json`. (The earlier OpenRouter gpt-4o-mini
sweep was retired — see "Gateway divergence" below.)

## Observation

On IFEval (verifiable instruction-following, strict prompt-level accuracy —
*every* constraint in an example must pass), PE2+SGR did **not** reliably
improve over its start prompt, and in the Claude run it **regressed**:

| optimizer / run            | target    | SGR ifeval (start→final) |
|----------------------------|-----------|--------------------------|
| claude-haiku-4.5 (self)    | claude    | 0.85 → **0.69** ⬇        |
| gpt-4o-mini (single, native) | gpt-4o-mini | 0.69 → **1.00** ⬆     |
| gpt-4o-mini (optimizer)    | qwen3-8b  | 0.77 → 0.69 ⬇            |
| gpt-4o-mini (optimizer)    | qwen3-14b | 0.69 → 0.85 ⬆            |
| gpt-4o-mini (optimizer)    | qwen3-32b | 0.62 → 0.62 (flat)       |

For contrast, PE2 (plain) on claude-haiku improved IFEval 0.77 → **0.92** (best
*on claude*) by rewriting into a clean compliance-only prompt — while on native
gpt-4o-mini it is PE2+SGR that wins IFEval outright (0.69 → **1.00**).

## Two distinct findings

### Finding 1 — "meta-objective leak" is optimizer-specific (Claude), not inherent to SGR

When **Claude Haiku was the optimizer**, the SGR-produced IFEval prompt opened
with a secondary objective that is not part of the task:

> "generate output that satisfies all requirements **while identifying which
> constraint types were applied**"

…and framed the instruction as a multi-step *"Understanding the Task → identify
constraints → Your Process"* meta-procedure, then contradicted itself with
"Present only the generated content… do not include explanations." This mirrors
SGR's internal error-categorization schema (`ErrorType`, per-constraint
diagnosis) — i.e. Claude **verbalized SGR's diagnostic frame into the produced
prompt**.

Crucially, this leak does **not** appear when **gpt-4o-mini** is the optimizer:
in all four gpt-4o-mini cases the SGR prompt contained none of the markers
(`identify which`, `constraint type`, `your process`, `error categ…`). So the
leak is a **model-specific verbalization artifact of Claude-as-optimizer**, not
a universal property of the SGR method.

For a strict all-or-nothing metric like IFEval, this dual/meta framing plausibly
lowers compliance (any stray meta-text or verbosity fails an example), which is
consistent with the claude regression — but see caveats on noise.

### Finding 2 — SGR's IFEval result is highly optimizer/context-dependent

SGR is **not uniformly weak** on IFEval — it is **highly variable**:

- **Excels** with native gpt-4o-mini as optimizer+target: 0.69 → **1.00**
  (perfect), the best IFEval result of any method/model here, and PE2+SGR is
  also the overall best method on native gpt-4o-mini (mean final 0.904, mean
  improvement +0.087).
- **Regresses** with Claude as optimizer: 0.85 → 0.69 (the meta-leak, Finding 1).
- **Mixed** when gpt-4o-mini optimizes weak qwen targets: up (14b 0.69→0.85),
  down (8b 0.77→0.69), flat (32b 0.62→0.62) — no leak in these, so this spread
  is plausibly transfer difficulty + seed noise on a 13-example dev set.

So the earlier "SGR is weak on IFEval" reading was **wrong** — driven by a
retired OpenRouter gpt-4o-mini run that scored 0.69 flat. The canonical native
gpt-4o-mini run scores **1.00**. The honest conclusion is that SGR *can* be the
best method for strict constraint-following, but its outcome depends strongly
on the optimizer model (clean rewrite vs meta-leak) and is noisy on weak
targets.

### Gateway divergence (why the OpenRouter gpt-4o-mini run was retired)

Same model, same start prompt (0.69), same seed — but `ifeval/pe2_sgr` finished
**1.00 on the native OpenAI API vs 0.69 via OpenRouter**. Start-prompt scoring
matched across gateways (the earlier equivalence check), but the **multi-step
optimization trajectory is not gateway-invariant** (generation nondeterminism +
subtle routing/version differences compound over candidate-generation rounds).
Lesson: pick one gateway per model and do not mix. We standardized on the
**native** OpenAI/Anthropic APIs; the OpenRouter gpt-4o-mini sweep was deleted
so it cannot be used as canonical.

## Interpretation (hypothesis, for the writeup)

SGR helps where *reasoning about the problem* transfers usefully into the prompt
(on claude: trec 0.54→0.92, subj 0.77→1.00, rucola 0.69→0.92). It does **not**
help — and can hurt — on tasks that want the model to *emit exact output and
nothing else*, where added reasoning/structure is a distraction. The
Claude-optimizer leak is an extreme instance of this: the diagnostic frame
itself bleeds into the instruction.

## Caveats (do not overstate)

- **Single seed, N=50, IFEval dev = 13 examples.** One or two flipped examples
  moves the score by ~0.08. The regression magnitudes are within plausible
  seed noise; the multi-seed (3-seed) qwen grid will indicate robustness.
- The **prompt-content evidence is robust** (the leak text is present/absent
  regardless of score noise); the **causal link to the score drop is not**
  (confounded by the small eval set).
- Each method starts from a different per-method template, so cross-method
  "improvement" deltas are partly confounded by starting point; compare final
  scores for deployment relevance.

## Follow-ups

- Confirm with multi-seed IFEval whether SGR's IFEval weakness is robust.
- If robust: consider an SGR guard that strips meta/diagnostic framing from the
  *produced* prompt (the rewrite should not instruct the model to categorize or
  explain), especially for strict-format tasks.
- Check whether the Claude-optimizer leak recurs on other strict-format
  benchmarks when Claude optimizes (only IFEval examined here).
