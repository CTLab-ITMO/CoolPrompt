# Phase 1 gate: per-constraint feedback ablation (ifeval, 8 seeds, gpt-4o-mini)

PASSED. SGR with per-constraint feedback (failure_breakdown threaded into the
diagnosis) vs SGR without, paired by seed (same code, only feedback differs),
plus the committed PE2 baseline.

Means: feedback-ON 0.731 | feedback-OFF 0.654 | PE2 0.606.
- ON > OFF in 6/8 seeds (1 tie, 1 loss); mean +0.077  -> the lever works.
- ON > PE2 in 6/8 seeds; mean +0.125 (CI lower bound ~0, seed 256 outlier).

Per-seed (ON / OFF / PE2): 7:.69/.54/.62  42:.69/.54/.69  99:.85/.77/.77
123:.69/.62/.62  256:1.0/.85/.54  512:.69/.62/.54  777:.62/.62/.31
2024:.62/.69/.77

Conclusion: structured per-constraint feedback gives SGR a real, mechanistically
-grounded edge on verifiable-constraint tasks. Caveats: N=50, ifeval dev=13;
confirm at higher N. Proceed to Phase 2 (regression guard).
