# Thesis reframe: from "SGR beats PE2" to a defensible contribution

This note exists because the experiments did not produce the result the
preliminary defense (April 1) pointed toward. SGR does not beat PE2. Rather
than weaken the thesis by overstating a win, this reframes the contribution
around what the work actually established — which is real, complete, and
defensible.

## The one-sentence thesis

> Schema-guided reasoning (SGR) yields no measurable improvement over PE2 for
> prompt optimization on any axis tested — mean accuracy, cross-seed variance,
> worst-case, or regression — and the method advantages reported at small
> evaluation sizes are statistical artifacts that vanish at adequate N; the
> contribution is a rigorous evaluation methodology that separates real method
> differences from evaluation noise.

This is a **negative result + methodology** thesis. Both halves are legitimate
academic contributions. Negative results are publishable precisely when the
methodology is strong enough to trust the "no difference" — which is exactly
what the multi-seed / adequate-N / paired-test design provides.

## Three honest contributions

1. **Evaluation methodology (the headline).** A protocol for comparing
   prompt-optimization methods that resists false positives: ≥8 seeds, adequate
   eval-set size N, paired sign tests / CIs, and an explicit N-sensitivity
   check. Demonstrated on a 17-benchmark suite across 4 methods and multiple
   model families (gpt-4o-mini native, qwen 8/14/32b, anthropic). The protocol
   *caught a real false positive*: the ifeval "win" (+0.13, 6/8 seeds at N=50)
   collapsed to a null at N=300, reproduced against the original v4 algorithm.

2. **The negative result, established completely.** SGR ≈ PE2 in mean,
   dispersion, worst-case floor, and regression rate (`robustness_variance.md`),
   and a stronger optimizer (gpt-4.1) does not change it. This is a *complete*
   null, not "we didn't find significance" — every axis we could measure agrees.

3. **An operational cost of structured reasoning.** SGR's structured-output
   requirement introduces intermittent malformed-JSON failures on complex tasks
   (ifeval on qwen-14b/32b) that free-form PE2 never exhibits
   (`CONSOLIDATED_FINDINGS.md` §5). So SGR is strictly costlier here for no
   measured benefit — itself a useful design finding.

### Optional qualitative angle (use only if honestly framed)

SGR's `FullDiagnosis` schema makes the optimizer's error analysis *inspectable*
(structured per-constraint reasoning vs PE2's free-form text). This is a real
interpretability property, but it is qualitative and comes bundled with the
fragility cost in (3). Present it as a trade-off, not a win.

## What NOT to do (and why it loses the defense)

- **Do not cherry-pick seeds.** The repo contains all 8-seed logs and the
  N-sensitivity analysis; a hand-picked "SGR wins" subset is directly
  contradicted by committed artifacts, and "I chose the seeds where it won" is
  the textbook question an examiner asks and the textbook answer that fails.
- **Do not lean on "enablement on weak models."** The data reverses it (see §5
  correction): the weak model (8b) ran SGR fine; the larger models failed it.

## Handling the April-1 preliminary results

The preliminary signal is an *asset*, not something to bury, if framed as the
scientific arc the thesis actually followed:

> "Preliminary experiments suggested SGR outperformed PE2 (notably on ifeval).
> To test whether this was robust, I scaled evaluation size and seed count.
> The advantage did not survive: it was an artifact of small-N evaluation. This
> motivated the evaluation-methodology contribution of the thesis."

That narrative — *preliminary signal → rigorous stress test → corrected
conclusion* — demonstrates scientific maturity. It is a stronger story than a
fragile win, and it directly answers "what changed since April."

## Advisor conversation

If the advisor's expectation of "a win" is rigid, the honest move is to bring
this reframe and the evidence (the N=50→N=300 collapse plot, the complete null
table, the v4 control). Frame the ask as: *the methodology and the corrected
negative result are the contribution; here is why that is defensible and
publishable.* Offer the qualitative interpretability angle as the "positive"
note if one is needed. Do not promise a win the data cannot support.

## Suggested chapter structure

1. Background: prompt optimization, PE2, SGR.
2. **Methodology**: benchmark suite, multi-seed / adequate-N / paired-test
   protocol, model families. (contribution 1)
3. Results: per-benchmark, multi-seed; the complete null; N-sensitivity case
   study (ifeval) with the v4 control. (contribution 2)
4. Robustness & cost analysis: variance, worst-case, regression, the
   structured-output fragility. (contributions 2–3)
5. Discussion: when structured reasoning helps / doesn't; interpretability
   trade-off; threats to validity; the small-N-artifact lesson for the field.
6. Conclusion: negative result + methodology.

## Artifacts backing each claim

- Methodology + null: `CONSOLIDATED_FINDINGS.md`, `aggregate_gpt_seeds.py`,
  `gpt4omini_8seed_results.txt`
- N-sensitivity: `phase1_gate_ifeval_ablation.md`, `logs/hiN_ifeval_s*.json`
- v4 control: `logs/v4_ifeval_s*.json`
- Robustness: `robustness_variance.md`, `scripts/variance_analysis.py`
- Fragility: `logs/ms_qwen{8,14,32}b_s*.json`
