import os
import sys
import json
import yaml
import time
import random
import argparse
import traceback
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_script_dir, "../../../")))
sys.path.append(os.path.abspath(os.path.join(_script_dir, "../../../src")))

import numpy as np
import torch
import gc

from coolprompt.optimizer.reflective_prompt.evoluter import ReflectiveEvoluter
from coolprompt.optimizer.reflective_prompt.factorized_evoluter import (
    FactorizedEvoluter,
)
from coolprompt.optimizer.reflective_prompt.coevo_evoluter import (
    CoevoEvoluter,
    PerFieldCoevoEvoluter,
)
from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.utils.logging_config import setup_logging

from model_utils import create_model
from dataset_config import DATASETS_CONFIG, load_train_data

setup_logging()

_EVAL_CONFIG_PATH = os.path.join(_script_dir, "evaluation_config.yaml")
_CONFIG_PATH = os.path.join(_script_dir, "config.yaml")
_PROXY_CONFIG_PATH = os.path.join(_script_dir, "proxy_config.yaml")

with open(_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

PROXY_CONFIG = (
    yaml.safe_load(open(_PROXY_CONFIG_PATH))
    if os.path.exists(_PROXY_CONFIG_PATH)
    else {}
)

if CONFIG.get("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = CONFIG["openai_api_key"]
if CONFIG.get("openrouter_api_key"):
    os.environ["OPENROUTER_API_KEY"] = CONFIG["openrouter_api_key"]

_FACTORIZED_MODES = {"factorized", "factorized_dedup", "factorized_top_prompts"}
_VALID_ROLE_MODES = {
    "with_role",
    "no_role",
    "coevo",
    "coevo_enhanced",
    "coevo_no_enhancements",
    "coevo_per_field",
} | _FACTORIZED_MODES


def _update_eval_config(dataset_name: str, result_file: str) -> None:
    with open(_EVAL_CONFIG_PATH) as f:
        eval_cfg = yaml.safe_load(f)
    rel_path = os.path.relpath(result_file, os.path.dirname(_EVAL_CONFIG_PATH))
    if os.sep != "/":
        rel_path = rel_path.replace(os.sep, "/")
    if "dataset_paths" not in eval_cfg or eval_cfg["dataset_paths"] is None:
        eval_cfg["dataset_paths"] = {}
    eval_cfg["dataset_paths"][dataset_name] = [rel_path]
    with open(_EVAL_CONFIG_PATH, "w") as f:
        yaml.dump(eval_cfg, f, allow_unicode=True, sort_keys=False)
    print(f"evaluation_config.yaml updated: {dataset_name} -> {rel_path}")


def run_optimization(
    args,
    config,
    train_inputs,
    train_targets,
    val_inputs,
    val_targets,
    logs_dir,
    role_mode,
    settings,
):
    temperature = settings["temperature"]
    model = create_model(
        args.provider,
        args.model,
        args.requests_per_minute,
        CONFIG,
        PROXY_CONFIG,
        temperature=temperature,
    )
    val_model = create_model(
        args.provider, args.model, None, CONFIG, PROXY_CONFIG, temperature=0.0
    )

    metric = validate_and_create_metric(config["task"], config["metric"])
    evaluator = Evaluator(model, config["task"], metric)
    val_evaluator = Evaluator(val_model, config["task"], metric)

    pop_size = settings["population_size"]
    num_epochs = settings["num_epochs"]
    phase_epochs = settings["factorized_phase_epochs"]
    use_enhancements = settings["use_enhancements"]
    use_dedup = settings["use_dedup"]
    evolve_constraints = settings["evolve_constraints"]
    task_desc = config["initial_task_description"]
    initial_constraints = config.get("initial_output_constraints", "")

    print(f"\nrunning: {role_mode}")

    if role_mode in _FACTORIZED_MODES:
        evoluter = FactorizedEvoluter(
            model=model,
            evaluator=evaluator,
            train_dataset=train_inputs,
            train_targets=train_targets,
            validation_dataset=val_inputs,
            validation_targets=val_targets,
            problem_description=f"Task: {config['description']}",
            initial_prompt=task_desc,
            initial_role=config["initial_system_behavior"],
            initial_constraints=(
                initial_constraints if evolve_constraints else None
            ),
            population_size=pop_size,
            phase_epochs=phase_epochs,
            run_constraints_phase=evolve_constraints,
            use_cache=True,
            output_path=logs_dir,
            use_enhancements=use_enhancements,
            use_dedup=use_dedup,
            val_evaluator=val_evaluator,
        )
    elif role_mode in ("coevo_enhanced", "coevo_no_enhancements"):
        evoluter = CoevoEvoluter(
            model=model,
            evaluator=evaluator,
            train_dataset=train_inputs,
            train_targets=train_targets,
            validation_dataset=val_inputs,
            validation_targets=val_targets,
            problem_description=f"Task: {config['description']}",
            initial_prompt=task_desc,
            initial_role=config["initial_system_behavior"],
            initial_constraints=initial_constraints,
            population_size=pop_size,
            num_epochs=num_epochs,
            use_cache=True,
            output_path=logs_dir,
            use_enhancements=(role_mode == "coevo_enhanced"),
            val_evaluator=val_evaluator,
        )
    elif role_mode == "coevo_per_field":
        evoluter = PerFieldCoevoEvoluter(
            model=model,
            evaluator=evaluator,
            train_dataset=train_inputs,
            train_targets=train_targets,
            validation_dataset=val_inputs,
            validation_targets=val_targets,
            problem_description=f"Task: {config['description']}",
            initial_prompt=task_desc,
            initial_role=config["initial_system_behavior"],
            initial_constraints=initial_constraints,
            population_size=pop_size,
            num_epochs=num_epochs,
            use_cache=True,
            output_path=logs_dir,
            use_enhancements=True,
            val_evaluator=val_evaluator,
        )
    else:
        if role_mode == "coevo":
            initial_role, evolve_role = config["initial_system_behavior"], True
        elif role_mode == "with_role":
            initial_role, evolve_role = config["initial_system_behavior"], False
        else:
            initial_role, evolve_role = "", False
        evoluter = ReflectiveEvoluter(
            model=model,
            evaluator=evaluator,
            train_dataset=train_inputs,
            train_targets=train_targets,
            validation_dataset=val_inputs,
            validation_targets=val_targets,
            problem_description=f"Task: {config['description']}",
            initial_prompt=task_desc,
            initial_role=initial_role,
            initial_constraints=(
                initial_constraints if evolve_constraints else None
            ),
            evolve_role=evolve_role,
            evolve_constraints=evolve_constraints,
            population_size=pop_size,
            num_epochs=num_epochs,
            use_cache=True,
            output_path=logs_dir,
            use_enhancements=use_enhancements,
            val_evaluator=val_evaluator,
        )

    evoluter.evolution()
    return evoluter


def main():
    parser = argparse.ArgumentParser(
        description="Optimize prompt for multiple datasets"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=CONFIG["provider"],
        choices=["openai", "openrouter"],
    )
    parser.add_argument("--model", type=str, default=CONFIG["model"])
    _output_dir = CONFIG["output_dir"]
    if not os.path.isabs(_output_dir):
        _output_dir = os.path.abspath(os.path.join(_script_dir, _output_dir))
    parser.add_argument("--output_dir", type=str, default=_output_dir)
    parser.add_argument("--requests_per_minute", type=int, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Minimal sizes (train=5, val=5, pop=2, epochs=1)",
    )
    args = parser.parse_args()

    if args.requests_per_minute is None:
        args.requests_per_minute = CONFIG["requests_per_minute"].get(
            args.provider, 500
        )

    settings = {
        "population_size": CONFIG["population_size"],
        "num_epochs": CONFIG["num_epochs"],
        "train_size": CONFIG["train_size"],
        "val_size": CONFIG["val_size"],
        "factorized_phase_epochs": tuple(
            CONFIG.get("factorized_phase_epochs", [4, 3, 3])
        ),
        "temperature": CONFIG.get("temperature", 0.0),
        "use_enhancements": CONFIG.get("use_enhancements", True),
        "use_dedup": CONFIG.get("use_dedup", True),
        "evolve_constraints": CONFIG.get("evolve_constraints", False),
    }

    if args.debug:
        settings.update(
            {
                "population_size": 2,
                "num_epochs": 1,
                "train_size": 5,
                "val_size": 5,
                "factorized_phase_epochs": (1, 1, 1),
            }
        )
        print("Debug mode: train=5, val=5, pop=2, epochs=1")

    _seed = CONFIG.get("seed", 42)
    random.seed(_seed)
    np.random.seed(_seed)

    datasets_to_run = CONFIG.get("datasets_to_run", [])
    role_modes_to_run = CONFIG.get("role_modes_to_run", ["with_role"])

    print(f"Datasets: {datasets_to_run}")
    print(f"Role modes: {role_modes_to_run}")
    print(
        f"Provider: {args.provider}, Model: {args.model}, RPM: {args.requests_per_minute}"
    )
    print(f"Enhancements: {'ON' if settings['use_enhancements'] else 'OFF'}")
    print(
        f"Constraints evolution: {'ON' if settings['evolve_constraints'] else 'OFF'}"
    )

    for dataset_name in datasets_to_run:
        if dataset_name not in DATASETS_CONFIG:
            print(f"skip unknown dataset: {dataset_name}")
            continue

        print(f"\n{dataset_name}")

        config = DATASETS_CONFIG[dataset_name]
        run_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(run_dir, exist_ok=True)

        train_size = settings["train_size"]
        val_size = settings["val_size"]
        try:
            inputs, targets = load_train_data(
                dataset_name, config, num_samples=train_size + val_size + 20
            )
        except Exception as e:
            print(f"failed to load {dataset_name}: {e}")
            continue

        if len(inputs) < train_size + val_size:
            print(
                f"warning: only {len(inputs)} samples, need {train_size + val_size}"
            )

        train_inputs = inputs[:train_size]
        train_targets = targets[:train_size]
        val_inputs = inputs[train_size : train_size + val_size]
        val_targets = targets[train_size : train_size + val_size]
        print(f"Split: train={len(train_inputs)}, val={len(val_inputs)}")

        for role_mode in role_modes_to_run:
            if role_mode not in _VALID_ROLE_MODES:
                print(f"skip unknown mode: {role_mode}")
                continue

            method_dir = os.path.join(run_dir, role_mode)
            os.makedirs(method_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logs_dir = os.path.join(
                method_dir, "logs", f"logs_{role_mode}_{timestamp}"
            )
            os.makedirs(logs_dir, exist_ok=True)
            print(f"\nmode: {role_mode}")
            print(f"Logs: {logs_dir}")

            max_retries = 3
            evoluter = None
            start_time = time.time()

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"Retry {attempt + 1}/{max_retries}...")
                        time.sleep(60)
                    evoluter = run_optimization(
                        args,
                        config,
                        train_inputs,
                        train_targets,
                        val_inputs,
                        val_targets,
                        logs_dir,
                        role_mode,
                        settings,
                    )
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(
                        f"error {dataset_name}/{role_mode} (attempt {attempt + 1}): {error_msg}"
                    )
                    if (
                        "429" in error_msg
                        or "Rate limit" in error_msg
                        or "quota" in error_msg
                    ):
                        wait_time = (attempt + 1) * 60
                        print(f"Rate limit. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        time.sleep(30)
                    if attempt == max_retries - 1:
                        print(f"max retries: {dataset_name}/{role_mode}")
                        with open(
                            os.path.join(
                                method_dir,
                                f"error_log_{role_mode}_{timestamp}.txt",
                            ),
                            "w",
                        ) as f:
                            f.write(
                                f"Failed after {max_retries} attempts.\nLast error: {error_msg}\n{traceback.format_exc()}"
                            )

            duration = time.time() - start_time

            if evoluter:
                is_factorized = role_mode in _FACTORIZED_MODES
                result_data = {
                    "dataset": dataset_name,
                    "role_mode": role_mode,
                    "model": args.model,
                    "best_prompt": evoluter.best_prompt_overall,
                    "best_role": evoluter.best_role_overall,
                    "best_constraints": evoluter.best_constraints_overall or "",
                    "best_score": evoluter.best_score_overall,
                    "candidates": getattr(evoluter, "candidates", None),
                    "initial_task_description": evoluter.initial_prompt,
                    "initial_system_behavior": evoluter.initial_role or "",
                    "initial_output_constraints": evoluter.initial_constraints
                    or "",
                    "description": config["description"],
                    "parameters": {
                        "population_size": settings["population_size"],
                        "num_epochs": (
                            None if is_factorized else settings["num_epochs"]
                        ),
                        "factorized_phase_epochs": (
                            list(settings["factorized_phase_epochs"])
                            if is_factorized
                            else None
                        ),
                        "train_size": len(train_inputs),
                        "val_size": len(val_inputs),
                        "rate_limit_rpm": args.requests_per_minute,
                        "provider": args.provider,
                        "temperature": settings["temperature"],
                        "val_temperature": 0.0,
                        "use_enhancements": (
                            (role_mode == "coevo_enhanced")
                            if role_mode
                            in ("coevo_enhanced", "coevo_no_enhancements")
                            else settings["use_enhancements"]
                        ),
                        "evolve_constraints": settings["evolve_constraints"],
                    },
                    "duration_seconds": duration,
                    "timestamp": timestamp,
                }

                score = evoluter.best_score_overall
                score_str = (
                    f"{score:.2f}" if isinstance(score, (int, float)) else "NA"
                )
                result_filename = (
                    f"{timestamp}_{score_str}_{role_mode}_seed{_seed}.json"
                )
                result_file = os.path.join(method_dir, result_filename)
                with open(result_file, "w") as f:
                    json.dump(result_data, f, indent=2)

                _update_eval_config(dataset_name, result_file)

                print(
                    f"\ndone {dataset_name}/{role_mode}, score={evoluter.best_score_overall}"
                )
                print(f"saved: {result_file}")

                del evoluter
                gc.collect()
                torch.cuda.empty_cache()

    print("\ndone")


if __name__ == "__main__":
    main()
