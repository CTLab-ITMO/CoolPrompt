# SAPO v2

SAPO v2 is a segment-aware prompt optimizer designed to improve scientific rigor and engineering robustness over SAPO v1.

## What changed vs SAPO

- Bandit-over-segment-edits (Thompson sampling) instead of pure greedy best-of-k.
- Multi-objective selection score:
  - BERTScore
  - format compliance
  - length penalty
  - estimated cost penalty
- Strict train/val split:
  - train split drives diagnostics and weakness analysis
  - val split drives candidate selection
- Failure memory + retrieval of similar historical failures.
- Segment confidence calibration used during candidate construction.
- Adaptive candidate budget per iteration.
- Hybrid generation:
  - full prompt rewrites
  - field-level patching (LLPO-style) for selected segment arms.

## Algorithm (high-level)

1. Evaluate initial prompt on train and val with multi-objective breakdown.
2. Extract prompt segments.
3. On each iteration:
   - Rank top/bottom train examples and update failure memory.
   - Retrieve similar failures from memory.
   - Produce segment weakness analysis.
   - Choose candidate budget adaptively.
   - Generate candidates with bandit-selected segment arms (full/patch modes).
   - Score candidates on val using multi-objective breakdown.
   - Update bandit posterior and segment confidence.
   - Accept winner only if val objective improves.
4. Stop on patience or max iterations.

## Pseudocode

```text
Input: p0, D_train, D_val, T, K_base
Initialize bandit Beta(alpha_s=1, beta_s=1) for each segment s
Initialize confidence c_s=0.5, memory M=[]

p <- p0
best <- ScoreVal(p)

for t in 1..T:
  train_scores, train_responses <- ScoreTrain(p)
  best5, worst5 <- RankExamples(train_scores)
  M <- UpdateFailureMemory(M, worst5)

  analysis <- AnalyzeWeaknesses(p, best5, worst5, Retrieve(M, worst5))
  K_t <- AdaptiveBudget(K_base, no_improve, last_gain)

  C <- []
  for i in 1..K_t:
    s_i <- ThompsonSample(alpha, beta, preferred=analysis.weak_segments)
    mode_i <- sample({patch, full})
    c_i <- GenerateCandidate(p, s_i, mode_i, analysis, c_s_i, M)
    C <- C U {(c_i, s_i, mode_i)}

  for each (c_i, s_i, mode_i) in C:
    v_i <- ScoreVal(c_i)   # multi-objective

  c* <- argmax_i v_i.objective
  gain <- v* - ScoreVal(p)

  for each candidate i:
    UpdateBandit(alpha_{s_i}, beta_{s_i}, reward = 1[v_i > ScoreVal(p)])
    UpdateConfidence(c_{s_i}, delta = v_i - ScoreVal(p))

  if gain > 0:
    p <- c*
    no_improve <- 0
  else:
    no_improve <- no_improve + 1

  if no_improve >= patience: break

return best prompt on val objective
```

## Minimal integration changes

- Existing `sapo` package remains untouched.
- New package is fully isolated in `sapov2`:
  - `sapov2/pipeline.py`
  - `sapov2/schema.py`
  - `sapov2/prompt.py`
  - `sapov2/llm.py`
  - `sapov2/__init__.py`

## Complexity (relative)

Let:
- `T` iterations
- `K` candidates per iteration
- `Ntr`, `Nval` dataset sizes
- `E` cost of one model response + metric update for one example

SAPO v1 (typical):
- `O(T * K * N * E)` on one split.

SAPO v2:
- train diagnostics: `O(T * Ntr * E)`
- val candidate scoring: `O(T * K_t * Nval * E)`
- retrieval/bandit/confidence overhead: lower-order terms
- total: `O(T * (Ntr + K_t * Nval) * E)`

In practice, SAPO v2 is usually more expensive than v1 when `Nval` is large, but typically delivers stronger selection reliability and better generalization.
