# Phase 2 gate: regression guard (8 seeds, easy tasks) — DID NOT PASS

SGR+guard vs committed PE2 and pre-guard SGR on subj/sst2/banking77/agnews.

mean guard-PE2 = -0.017 (gate threshold >= -0.01 -> FAIL, marginal/within noise)
mean guard-oldSGR = +0.002 (guard is essentially inert)

Per-benchmark (guard / oldSGR / PE2):
 subj      .788 / .788 / .788   (tie; the 3-seed -0.13 loss was noise)
 sst2      .933 / .913 / .952   (guard +.019 vs old, still -.019 vs PE2)
 banking77 .894 / .923 / .942   (guard WORSE than old by -.029; -.048 vs PE2)
 agnews    .923 / .904 / .923   (guard +.019 vs old, tie vs PE2)

Conclusion: the targeted failure mode (SGR over-engineering on easy tasks)
was largely a 3-seed artifact; at 8 seeds SGR was already ~= PE2. The no-change/
length-tiebreak heuristics are net-neutral and slightly hurt banking77.
Recommendation: revert Phase 2 (or keep only length tiebreak) and rely on
Phase 3 portfolio to avoid easy-task losses by construction.
