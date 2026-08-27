import os
import sys
import json
import yaml
import time
import argparse
import traceback
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_script_dir, "../../../")))
sys.path.append(os.path.abspath(os.path.join(_script_dir, "../../../src")))

import torch
import gc

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from coolprompt.optimizer.reflective_prompt.prompt import Prompt
from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.utils.logging_config import setup_logging

from model_utils import create_model, normalize_model_name
from dataset_config import DATASETS_CONFIG, load_eval_data

setup_logging()

_EVAL_CONFIG_PATH = os.path.join(_script_dir, "evaluation_config.yaml")
_PROXY_CONFIG_PATH = os.path.join(_script_dir, "proxy_config.yaml")

with open(_EVAL_CONFIG_PATH) as f:
    EVAL_CONFIG = yaml.safe_load(f)

PROXY_CONFIG = (
    yaml.safe_load(open(_PROXY_CONFIG_PATH))
    if os.path.exists(_PROXY_CONFIG_PATH)
    else {}
)

DEFAULT_TEST_SIZE = EVAL_CONFIG.get("default_test_size", 1000)
DEFAULT_SEED = EVAL_CONFIG.get("default_seed", 42)
DEFAULT_PROVIDER = EVAL_CONFIG.get("provider", "openai")
DEFAULT_MODEL = EVAL_CONFIG.get("model", "gpt-4o-mini")
TEMPERATURE_CONFIG = EVAL_CONFIG.get("temperature", 0.0)

REQUESTS_PER_MINUTE_CONFIG = EVAL_CONFIG.get(
    "requests_per_minute", {"openai": 200, "openrouter": 5000}
)

_COMBO_MAP = {1: "text_only", 2: "text_role", 3: "text_role_constraints"}
_raw_combo = EVAL_CONFIG.get("combo", "all")
if str(_raw_combo).strip().lower() == "all":
    COMBO_FILTER = None
else:
    _items = (
        _raw_combo
        if isinstance(_raw_combo, list)
        else str(_raw_combo).split(",")
    )
    COMBO_FILTER = {_COMBO_MAP[int(x)] for x in _items}

_raw_output_dir = EVAL_CONFIG.get("output_dir", "./evaluation_results")
EVAL_OUTPUT_DIR = (
    _raw_output_dir
    if os.path.isabs(_raw_output_dir)
    else os.path.join(_script_dir, _raw_output_dir)
)

_raw_paths = EVAL_CONFIG.get("dataset_paths", {})
DATASET_PATHS = {
    k: (
        [os.path.join(_script_dir, p) if not os.path.isabs(p) else p for p in v]
        if isinstance(v, list)
        else (os.path.join(_script_dir, v) if not os.path.isabs(v) else v)
    )
    for k, v in _raw_paths.items()
}

_first_proxy = (PROXY_CONFIG.get("proxies") or [None])[0] or PROXY_CONFIG.get(
    "proxy", {}
).get("http")
if _first_proxy:
    os.environ.setdefault("HTTP_PROXY", _first_proxy)
    os.environ.setdefault("HTTPS_PROXY", _first_proxy)
    print(f"Proxy set for HF downloads: {_first_proxy}")


def evaluate_single_dataset(
    dataset_name,
    json_path,
    inputs,
    targets,
    provider,
    model_name,
    requests_per_minute,
    seed=None,
    full_test=False,
):
    if not json_path or not os.path.exists(json_path):
        print(f"skip {dataset_name}: no valid json path")
        return None

    with open(json_path) as f:
        data = json.load(f)

    effective_model_name = normalize_model_name(provider, model_name)

    candidates_meta = data.get("candidates")
    if candidates_meta:
        prompts_to_eval = [
            {
                "combo": c["combo"],
                "prompt": c["prompt"],
                "role": c.get("role", ""),
                "constraints": c.get("constraints", ""),
                "val_score": c.get("val_score"),
            }
            for c in candidates_meta
            if c.get("prompt")
        ]
    else:
        best_prompt_text = data.get("best_prompt")
        if not best_prompt_text:
            print(f"no best_prompt in {dataset_name} json")
            return None
        prompts_to_eval = [
            {
                "combo": "default",
                "prompt": best_prompt_text,
                "role": data.get("best_role") or "",
                "constraints": data.get("best_constraints") or "",
                "val_score": data.get("best_score"),
            }
        ]

    if COMBO_FILTER:
        prompts_to_eval = [
            p for p in prompts_to_eval if p["combo"] in COMBO_FILTER
        ]
    print(f"{dataset_name}: {len(prompts_to_eval)} combos")

    config = DATASETS_CONFIG[dataset_name]
    model = create_model(
        provider,
        model_name,
        requests_per_minute,
        EVAL_CONFIG,
        PROXY_CONFIG,
        temperature=TEMPERATURE_CONFIG,
    )
    metric = validate_and_create_metric(config["task"], config["metric"])
    evaluator = Evaluator(model, config["task"], metric)

    candidate_results = []
    for p in prompts_to_eval:
        prompt_obj = Prompt(
            text=p["prompt"], role=p["role"], constraints=p["constraints"]
        )
        try:
            score = evaluator.evaluate(
                prompt=prompt_obj.text,
                dataset=inputs,
                targets=targets,
                system_role=prompt_obj.role or None,
                constraints=prompt_obj.constraints or None,
            )
            print(f"  [{p['combo']}] test={score:.4f}  val={p['val_score']}")
            candidate_results.append(
                {
                    "combo": p["combo"],
                    "prompt": p["prompt"],
                    "role": p["role"],
                    "constraints": p["constraints"],
                    "val_score": p["val_score"],
                    "test_score": score,
                }
            )
        except Exception as e:
            print(f"  error [{p['combo']}]: {e}")
            traceback.print_exc()

    if not candidate_results:
        return None

    best_result = max(
        candidate_results,
        key=lambda r: (
            r["val_score"]
            if r.get("val_score") is not None
            else r["test_score"]
        ),
    )
    print(
        f"best: {best_result['combo']} = {best_result['test_score']:.4f} (val={best_result.get('val_score')})"
    )

    opt_params = data.get("parameters", {})
    return {
        "dataset": dataset_name,
        "role_mode": data.get("role_mode", "unknown"),
        "score": best_result["test_score"],
        "best_combo": best_result["combo"],
        "metric": config["metric"],
        "num_samples": len(inputs),
        "seed": seed,
        "full_test": full_test,
        "provider": provider,
        "model": effective_model_name,
        "opt_temperature": opt_params.get("temperature", 0.7),
        "val_temperature": opt_params.get("val_temperature", 0.7),
        "use_enhancements": opt_params.get("use_enhancements", None),
        "json_path": json_path,
        "candidate_results": candidate_results,
        "prompt_info": data,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimized prompts")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_PROVIDER,
        choices=["openai", "openrouter"],
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--requests_per_minute", type=int, default=None)
    parser.add_argument(
        "--full_test",
        action="store_true",
        help="Evaluate on the full split (no seed/size limit)",
    )
    args = parser.parse_args()

    if args.requests_per_minute is None:
        if isinstance(REQUESTS_PER_MINUTE_CONFIG, dict):
            args.requests_per_minute = REQUESTS_PER_MINUTE_CONFIG.get(
                args.provider, 500
            )
        else:
            args.requests_per_minute = int(REQUESTS_PER_MINUTE_CONFIG)

    print(
        f"Provider: {args.provider}, Model: {args.model}, RPM: {args.requests_per_minute}"
    )

    set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    output_dir = (
        EVAL_OUTPUT_DIR.rstrip("/\\") + "_full"
        if args.full_test
        else EVAL_OUTPUT_DIR
    )

    datasets_to_run = DATASET_PATHS
    if args.dataset:
        if args.dataset not in DATASET_PATHS:
            print(f"unknown dataset: {args.dataset}")
            return
        datasets_to_run = {args.dataset: DATASET_PATHS[args.dataset]}

    results = {}

    for dataset_name, dataset_json_paths in datasets_to_run.items():
        if not dataset_json_paths:
            print(f"skip {dataset_name}: no json path")
            continue
        if dataset_name not in DATASETS_CONFIG:
            print(f"skip {dataset_name}: not in config")
            continue

        if isinstance(dataset_json_paths, str):
            dataset_json_paths = [dataset_json_paths]
        elif not isinstance(dataset_json_paths, list):
            print(f"skip {dataset_name}: bad path type")
            continue

        print(f"\n{dataset_name}")

        config = DATASETS_CONFIG[dataset_name]
        try:
            inputs, targets = load_eval_data(
                dataset_name,
                config,
                args.num_samples,
                args.seed,
                args.full_test,
            )
        except Exception as e:
            print(f"failed to load {dataset_name}: {e}")
            continue

        if not inputs:
            print(f"no data: {dataset_name}")
            continue

        dataset_results = []
        for json_path in dataset_json_paths:
            max_retries = 3
            dataset_result = None
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for attempt in range(max_retries):
                try:
                    dataset_result = evaluate_single_dataset(
                        dataset_name,
                        json_path,
                        inputs,
                        targets,
                        args.provider,
                        args.model,
                        args.requests_per_minute,
                        seed=args.seed,
                        full_test=args.full_test,
                    )
                    if dataset_result:
                        break
                except Exception as e:
                    print(f"error (attempt {attempt + 1}): {e}")
                    if "429" in str(e) or "Rate limit" in str(e):
                        time.sleep(60 * (attempt + 1))
                    else:
                        time.sleep(10)

            if dataset_result:
                dataset_results.append(dataset_result)

                role_mode = dataset_result.get("role_mode", "unknown")
                method_dir = os.path.join(output_dir, dataset_name, role_mode)
                os.makedirs(method_dir, exist_ok=True)
                score = dataset_result.get("score")
                score_str = (
                    f"{score:.2f}" if isinstance(score, (int, float)) else "NA"
                )
                seed_str = "all" if args.full_test else str(args.seed)
                filename = f"{run_timestamp}_{score_str}_{role_mode}_seed{seed_str}.json"
                save_path = os.path.join(method_dir, filename)
                with open(save_path, "w") as f:
                    json.dump(dataset_result, f, indent=2)
                print(f"saved: {save_path}")

        if dataset_results:
            results[dataset_name] = dataset_results

        gc.collect()
        torch.cuda.empty_cache()

    print("\nsummary:")
    for name, dataset_runs in results.items():
        for idx, res in enumerate(dataset_runs, start=1):
            combo_info = (
                f" [{res.get('best_combo', 'default')}]"
                if res.get("best_combo")
                else ""
            )
            print(
                f"  {name} [run {idx}]{combo_info}: {res['score']:.4f} ({res['metric']}) - {res['num_samples']} samples"
            )
            if (
                res.get("candidate_results")
                and len(res["candidate_results"]) > 1
            ):
                for cr in res["candidate_results"]:
                    vs = cr.get("val_score")
                    vs_str = f"{vs:.4f}" if isinstance(vs, float) else str(vs)
                    print(
                        f"    {cr['combo']}: val={vs_str}  test={cr['test_score']:.4f}"
                    )


if __name__ == "__main__":
    main()
