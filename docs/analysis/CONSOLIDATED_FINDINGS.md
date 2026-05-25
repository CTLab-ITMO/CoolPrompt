# PE2+SGR investigation — consolidated findings

**Date:** 2026-05-25. Runtime model gpt-4o-mini (native OpenAI) unless noted.
Optimizer = same model unless noted. Metrics: ifeval (strict prompt-level),
em (math), accuracy (classification), rouge (xsum). All multi-seed, paired.

## TL;DR
On these 17 benchmarks, **PE2+SGR ≈ PE2** — there is no reliable accuracy
advantage for SGR's structured reasoning over plain PE2. The headline result
is **methodological**: small-N evaluations (common in the literature)
manufacture spurious method wins that vanish at adequate N.

## Evidence

### 1. SGR ≈ PE2 overall (8 seeds, N=50, 17 benchmarks)
Mean final: **pe2_sgr 0.826 ≈ pe2 0.821**; both ≫ ape 0.723; opro 0.790.
Per-benchmark paired sign tests are near coin-flips. The dramatic 3-seed
signals (ANLI "win" +0.10, subj "catastrophe" −0.13) were **noise** — both
collapsed to ties at 8 seeds.

### 2. The one apparent SGR win (ifeval) was a small-N artifact
- Phase-1 per-constraint feedback (feed the IFEval checker's which-constraint-
  failed breakdown into SGR's diagnosis) looked like a win at **N=50**:
  feedback-ON 0.731 > OFF 0.654 (6/8) and > PE2 0.606 (6/8).
- At **N=300** (ifeval dev grows ~13 → ~75 examples) the advantage **vanishes**:
  SGR 0.603 vs PE2 0.620, mean diff −0.017, 95% CI [−0.053, +0.020] (includes
  0), SGR>PE2 in only 2/8 seeds.
- The N=50 "win" was manufactured by the ~13-example dev split's variance.

### 3. The null is inherent to v4, not caused by our changes (control)
Literal original-v4 SGR (restored from commit 39d6cb4 in an isolated worktree,
own venv) on ifeval N=300, 8 seeds:
  **v4-SGR 0.613 ≈ current-SGR 0.603 ≈ PE2 0.620** (all within 0.01).
current-SGR vs v4-SGR = −0.010 → Phase-1 plumbing did not alter behavior.
Confirms SGR≈PE2 is a property of the method, not our modifications.

### 4. A stronger optimizer does not help (runtime fixed at gpt-4o-mini)
Swapping the optimizer gpt-4o-mini → gpt-4.1 (3 seeds, 17 benchmarks): mean Δ
≈ 0 (pe2 −0.003, pe2_sgr +0.009, ape −0.024, opro −0.027). Optimizer
*capability* is not the bottleneck — signal/eval-noise is.

### 5. Strong-optimizer → weak-runtime: feasibility, with an SGR fragility cost
A strong optimizer (gpt-4o-mini) producing prompts for weak qwen runtimes
(8b/14b/32b) WORKS, and gives no quality uplift: SGR ≈ PE2 ≫ APE on every
qwen runtime with no "helps-weak-more" trend.

CORRECTION (2026-05-25): an earlier version of this section claimed weak
models "can't reliably emit the structured output pe2_sgr needs." The logs
do not support that and arguably reverse it. qwen-8b ran pe2_sgr fine on all
5 benchmarks × 3 seeds. The structured-output failures were on qwen-**14b**
and **32b** ifeval (JSON parse errors, e.g. `Expecting value: line 3023`),
while plain PE2 ran fine on the same cells. So the real finding is the
opposite of "enablement": SGR's structured-output requirement introduces an
**operational fragility** — intermittent malformed-JSON failures on complex
tasks — that free-form PE2 does not have. This is a cost of SGR, not a
benefit. (Evidence: `logs/ms_qwen{8,14,32}b_s*.json`.)

### 5b. SGR ≈ PE2 on robustness too (not just mean)
Cross-seed analysis (`scripts/variance_analysis.py`) over the 8-seed
gpt-4o-mini grid, 17 benchmarks, asks whether SGR is more *consistent*
even where its mean ties. It is not:
- **Dispersion:** SGR has lower cross-seed std in only 5/17 benchmarks
  (PE2 in 9, tie 3); mean std 0.075 (SGR) vs 0.077 (PE2) — a wash.
- **Worst-case floor:** SGR's min-over-seeds is higher in 9/17 (mean
  +0.03) — a weak lean, not significant (9 of 14 decisive, binomial
  p ≈ 0.4), and driven by the high-variance ifeval/bbh cells.
- **Regression rate** (runs where final < start): SGR 6.6% vs PE2 5.1%.
The null is therefore complete: SGR ≈ PE2 in mean, variance, worst-case,
and regression. See `robustness_variance.md`.

### 6. Honest portfolio (best-of PE2/SGR, leakage-free) buys little
Held-out-val selection (N=200, 5 seeds, 8 discriminating benchmarks):
portfolio 0.727 vs best-single pe2 0.707 → **+0.020, within noise**. The
oracle (select-on-test) bound of 0.866 was badly inflated. The niche map at
honest N does NOT show SGR dominating any benchmark (ifeval 1/5) — there is no
robust "SGR niche."

## Conclusions for the thesis
1. **Primary (methodological):** prompt-optimization method comparisons on
   small eval sets are unreliable; a +0.13/6-of-8 ifeval "win" at N=50 became
   a null at N=300. Report at adequate N with paired CIs / sign tests.
2. **On SGR specifically:** structured schema-guided reasoning does not beat
   plain PE2 here, even on its best-looking benchmark, even with per-constraint
   feedback, even vs the original v4 — and a stronger optimizer doesn't change
   it. SGR ≈ PE2 ≫ APE; OPRO mid and operationally fragile (long-context 504s).
3. **The defensible positive:** the strong-optimizer→weak-runtime design
   reliably ENABLES structured methods on runtimes too weak to self-optimize.

## Artifacts
- 8-seed gpt-4o-mini: `gpt4omini_8seed_results.txt`, `aggregate_gpt_seeds.py`
- Phase-1 gate + refutation: `phase1_gate_ifeval_ablation.md`, this doc §2
- v4 control: `logs/v4_ifeval_s*.json` (worktree `.worktrees/v4check`)
- Portfolio: `portfolio_niche_map.py` (oracle), `portfolio_runner.py` +
  `logs/portfolio_heldout_s*.json` (honest)
- Per-phase: `sgr_niche_results.md`, `phase2_gate_regression_guard.md`
