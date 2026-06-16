# EvoPrompt on generic datasets with `gpt-5-nano`

This folder adapts the EvoPrompt method to optimise prompts over the eight
datasets stored under [`datasets/data/`](data/):

`squad_v2`, `gsm8k`, `common_gen`, `xsum`, `tweeteval`, `mediqa`,
`code_to_text`, `concode`.

The LLM backend is [`langchain_openai.ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai/),
configured by default to `model=gpt-5-nano`, `temperature=0.7`, and reads the
API key from the `OPENAI_API_KEY` environment variable.

## Quick start

```bash
pip install -r ../requirements.txt
export OPENAI_API_KEY=sk-...

# Differential Evolution on GSM8K
python run.py \
  --dataset gsm8k \
  --evo_mode de \
  --popsize 10 --budget 10 \
  --sample_num 50 --test_sample_num 100 \
  --output outputs/gsm8k_de --seed 5

# Genetic Algorithm on XSum
python run.py --dataset xsum --evo_mode ga --popsize 10 --budget 10 \
              --sample_num 30 --test_sample_num 100 --seed 5

# APE-style paraphrasing on TweetEval
python run.py --dataset tweeteval --evo_mode ape --popsize 10 --budget 10 \
              --sample_num 50 --test_sample_num 100 --seed 5
```

Other useful flags:

* `--model gpt-5-nano` — change the underlying chat model
* `--temperature 0.7` — sampling temperature (default 0.7)
* `--openai_api_key sk-...` — alternative to the env var
* `--metric {bert_score,exact_match,f1_mera}` — metric used to score candidate
  prompts. If omitted, the per-dataset default is applied:

  | Dataset       | Default metric |
  |---------------|----------------|
  | `gsm8k`       | `exact_match`  |
  | `tweeteval`   | `f1_mera`      |
  | `squad_v2`, `common_gen`, `xsum`, `mediqa`, `code_to_text`, `concode` | `bert_score` |

  Notes on `bert_score`: the metric is computed with the
  [`bert-score`](https://github.com/Tiiiger/bert_score) package
  (`pip install bert-score`). If the package is not available at import time,
  the code falls back to ROUGE-L F1 and logs a single warning so that
  evolution can still proceed.

## Optimisation log (JSON)

The full optimisation history is written to
`<output>/optimization_log.json` (overridable via `--results_json`). It
includes the optimisation `dataset` and `metric` both at the top level and
inside the `final` block, alongside the best prompt and its scores:

```jsonc
{
  "dataset": "gsm8k",
  "metric": "exact_match",
  "initial": [ /* ... */ ],
  "steps":   [ /* ... */ ],
  "final": {
    "dataset": "gsm8k",
    "metric": "exact_match",
    "best_prompt": "...",
    "dev_score": 0.42,
    "test_score": 0.39,
    "top_candidates": [ /* ... */ ]
  }
}
```
* `--results_json path.json` — full optimisation log (defaults to
  `<output>/optimization_log.json`)

## Output JSON

After optimisation the file at `--results_json` contains:

```json
{
  "config": { ...all CLI args... },
  "dataset": "gsm8k",
  "initial":  [ {"prompt": "...", "mark": "manual", "score": 0.42}, ... ],
  "steps": [
    {
      "step": 1,
      "best_score": 0.55,
      "avg_score": 0.47,
      "events": [
        {"parents": [...], "child_prompt": "...", "child_score": 0.51,
         "selected_prompt": "...", "selected_score": 0.51}
      ],
      "population": [ {"prompt": "...", "score": 0.55, "mark": "evoluted"} ]
    }
  ],
  "final": {
    "best_prompt": "best prompt text",
    "dev_score": 0.55,
    "test_score": 0.53,
    "top_candidates": [ ... ]
  }
}
```

The file is rewritten after every evolution step so partial progress is
preserved if the run is interrupted.

## Files of interest

* [`run.py`](run.py:1) — CLI entry point
* [`args.py`](args.py:1) — argparse definitions (`--dataset`, `--model`, …)
* [`llm_client.py`](llm_client.py:1) — `ChatOpenAI` wrapper used by everything
* [`dataset_eval.py`](dataset_eval.py:1) — per-dataset prompt builder and metric
* [`evoluter.py`](evoluter.py:1) — DE / GA / APE evoluters that record history
