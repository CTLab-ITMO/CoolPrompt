# SGR-Niche: final results (8-seed gpt-4o-mini)

## Phase 1 — per-constraint feedback (KEPT, real win)
SGR with failure_breakdown threaded into the diagnosis beats SGR-without
on ifeval in 6/8 seeds (mean +0.077) and beats PE2 in 6/8 (mean +0.125).
Ablation-proven: the lift comes from the feedback, not luck.
Improved-SGR ifeval mean = 0.73 vs PE2 0.61.

## Phase 2 — regression guard (REVERTED, did not earn its keep)
Net-neutral vs pre-guard SGR (+0.002); slightly hurt banking77. The
"over-engineering on easy tasks" loss it targeted was largely 3-seed noise
(at 8 seeds SGR was already ~= PE2). Reverted (commits 9dc1b28, 37680dc).

## Phase 3 — portfolio (best-of PE2 / improved-SGR) + niche map
Overall mean: pe2 0.821 | sgr 0.828 | portfolio 0.866 (>= both everywhere,
+0.045 over either alone). CAVEAT: selection is on the eval metric ->
ORACLE UPPER BOUND; a deployable portfolio must select on a held-out val.

Niche map (seeds /8 where SGR is selected):
  ifeval 6  formal_fallacies 5  xsum 5  banking77 4  rucola 4
  agnews 2  boolean 2  causal 3  sports 3  gsm8k 3  sst2 3  subj 3
  trec 3  anli 3  svamp 3  web_of_lies 1  navigate 0

## Conclusion
SGR's defensible niche = verifiable-constraint following (ifeval), where
structured per-constraint feedback gives it a real, mechanism-backed edge,
plus some multi-step reasoning (formal_fallacies). PE2 is as good or better
on simple classification and some BBH reasoning. The portfolio operationalizes
"use SGR where it wins" and the niche map characterizes the boundary.
Follow-up: held-out-val portfolio for an honest deployment number; higher N
to tighten the marginal vs-PE2 CIs.
