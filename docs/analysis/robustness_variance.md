# Cross-seed robustness: pe2_sgr vs pe2

**Question.** Mean accuracy of SGR ≈ PE2 (see `CONSOLIDATED_FINDINGS.md`).
Does SGR at least produce *more consistent* prompts across random seeds —
lower variance, a higher worst-case floor, fewer runs where optimization
backfires? A structured method could plausibly trade peak for stability.

**Data.** Reuses the committed 8-seed gpt-4o-mini grid (17 benchmarks,
seeds 42/123/2024/256/512/7/777/99) — no new runs. The qwen ladder (3
seeds each, 8b/14b/32b) is a secondary check. Each run stores only the
final scalar (`final_metric`) and `start_score`; that is sufficient for
dispersion, worst-case, and regression. Script: `scripts/variance_analysis.py`.

## Result — no robustness advantage

gpt-4o-mini, 17 benchmarks, 8 paired seeds:

| Axis | Measure | SGR | PE2 | Verdict |
|------|---------|-----|-----|---------|
| Dispersion | mean cross-seed std | 0.075 | 0.077 | wash |
| Dispersion | benchmarks where SGR steadier | 5/17 (PE2 9, tie 3) | — | PE2 if anything |
| Worst-case | mean min-over-seeds | 0.706 | 0.674 | +0.03 weak lean |
| Worst-case | benchmarks where SGR floor higher | 9/17 (PE2 5, tie 3) | — | not significant |
| Regression | runs with final < start | 6.6% (9/136) | 5.1% (7/136) | PE2 slightly better |

- **Dispersion:** essentially identical. SGR is *not* steadier — by count
  PE2 has the lower std slightly more often.
- **Worst-case floor:** the one faint positive — SGR's worst seed is higher
  on average by 0.03 and in 9/17 benchmarks. But 9 of 14 decisive
  comparisons is binomial p ≈ 0.4 (not significant), and the gap is driven
  by the high-variance cells (ifeval, bbh_formal_fallacies) that the rest of
  the investigation already flagged as eval-noise dominated. Not a claim.
- **Regression:** SGR backfires slightly more often, not less.

**qwen ladder (3 seeds, secondary).** Mixed and mostly against SGR on
dispersion (8b: PE2 steadier 3/5; 32b: PE2 steadier 3/4 with much lower
std). No robustness signal for SGR here either.

## Conclusion

The SGR ≈ PE2 null is **complete**: indistinguishable in mean, cross-seed
variance, worst-case floor, and regression rate. There is no robustness
niche to claim. Combined with the operational fragility of SGR's structured
output on complex tasks (`CONSOLIDATED_FINDINGS.md` §5), the honest summary
is that schema-guided reasoning costs more (parse failures) and buys nothing
measurable over plain PE2 in this setting.

Reproduce: `uv run python scripts/variance_analysis.py`
