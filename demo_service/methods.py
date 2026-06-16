"""Public method metadata for the CoolPrompt demo UI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


METHODS: list[dict[str, Any]] = [
    {
        "id": "hyper_light",
        "label": "HyPER Light",
        "family": "single-shot",
        "description": "Fast one-step meta-prompt optimization.",
        "data_driven": False,
        "legacy": False,
        "params": [
            {
                "name": "use_structured_output",
                "label": "Structured output",
                "type": "bool",
                "default": False,
                "help": "Ask the optimizer model to use structured output when supported.",
            },
        ],
    },
    {
        "id": "hyper",
        "label": "HyPER",
        "family": "iterative",
        "description": "Iterative feedback-driven prompt optimization.",
        "data_driven": True,
        "legacy": False,
        "params": [
            {"name": "n_iterations", "label": "Iterations", "type": "int", "default": 2, "min": 1, "max": 10},
            {"name": "n_candidates", "label": "Candidates", "type": "int", "default": 2, "min": 1, "max": 8},
            {"name": "top_n_candidates", "label": "Shortlist", "type": "int", "default": 2, "min": 1, "max": 8},
            {"name": "mini_batch_size", "label": "Mini-batch", "type": "int", "default": 4, "min": 1, "max": 64},
            {"name": "k_samples", "label": "Failed examples", "type": "int", "default": 2, "min": 1, "max": 8},
            {"name": "random_seed", "label": "Seed", "type": "int", "default": 7, "min": 0, "max": 999999},
        ],
    },
    {
        "id": "rider",
        "label": "RIDER",
        "family": "evolutionary",
        "description": "RIDER Genesis Ultra with train/validation-aware reranking.",
        "data_driven": True,
        "legacy": False,
        "params": [
            {"name": "num_samples", "label": "Samples", "type": "int", "default": 1, "min": 1, "max": 8},
            {"name": "num_generations", "label": "Generations", "type": "int", "default": 1, "min": 1, "max": 6},
            {"name": "population_size", "label": "Population", "type": "int", "default": 2, "min": 2, "max": 12},
            {"name": "num_strategies", "label": "Strategies", "type": "int", "default": 2, "min": 2, "max": 5},
            {"name": "train_sample_size", "label": "Train samples", "type": "int", "default": 4, "min": 1, "max": 64},
            {"name": "validation_sample_size", "label": "Validation samples", "type": "int", "default": 2, "min": 1, "max": 64},
            {"name": "external_eval_weight", "label": "External eval weight", "type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
            {"name": "temperature", "label": "RIDER temperature", "type": "float", "default": 0.4, "min": 0.0, "max": 1.5, "step": 0.05},
            {"name": "seed", "label": "Seed", "type": "int", "default": 7, "min": 0, "max": 999999},
        ],
    },
    {
        "id": "regps",
        "label": "RE-GPS",
        "family": "evolutionary",
        "description": "Evolutionary optimizer with reflective guidance.",
        "data_driven": True,
        "legacy": False,
        "params": [
            {"name": "population_size", "label": "Population", "type": "int", "default": 4, "min": 2, "max": 16},
            {"name": "num_epochs", "label": "Epochs", "type": "int", "default": 1, "min": 1, "max": 8},
            {"name": "bad_examples_number", "label": "Bad examples", "type": "int", "default": 2, "min": 1, "max": 10},
            {"name": "use_cache", "label": "Use cache", "type": "bool", "default": False},
        ],
    },
    {
        "id": "compress",
        "label": "PromptCompressor",
        "family": "utility",
        "description": "Compresses long prompts while preserving task intent.",
        "data_driven": False,
        "legacy": False,
        "params": [],
    },
    {
        "id": "reflective",
        "label": "ReflectivePrompt",
        "family": "legacy",
        "description": "Deprecated reflective evolutionary optimizer.",
        "data_driven": True,
        "legacy": True,
        "params": [
            {"name": "population_size", "label": "Population", "type": "int", "default": 4, "min": 2, "max": 16},
            {"name": "num_epochs", "label": "Epochs", "type": "int", "default": 1, "min": 1, "max": 8},
            {"name": "use_cache", "label": "Use cache", "type": "bool", "default": False},
        ],
    },
    {
        "id": "distill",
        "label": "DistillPrompt",
        "family": "legacy",
        "description": "Deprecated distillation-based optimizer.",
        "data_driven": True,
        "legacy": True,
        "params": [
            {"name": "num_epochs", "label": "Epochs", "type": "int", "default": 1, "min": 1, "max": 8},
            {"name": "use_cache", "label": "Use cache", "type": "bool", "default": False},
        ],
    },
]


METHOD_BY_ID: dict[str, dict[str, Any]] = {method["id"]: method for method in METHODS}


def public_methods() -> list[dict[str, Any]]:
    """Return a defensive copy of UI method metadata."""

    return deepcopy(METHODS)


def coerce_method_params(method_id: str, raw_params: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only params supported by the selected method and coerce primitive values."""

    method = METHOD_BY_ID.get(method_id)
    if method is None:
        raise ValueError(f"Unknown method: {method_id}")

    raw_params = raw_params or {}
    result: dict[str, Any] = {}

    for spec in method.get("params", []):
        name = spec["name"]
        if name not in raw_params:
            value = spec.get("default")
        else:
            value = raw_params[name]

        if value is None or value == "":
            continue

        typ = spec.get("type")
        if typ == "int":
            value = int(value)
            value = max(int(spec.get("min", value)), value)
            value = min(int(spec.get("max", value)), value)
        elif typ == "float":
            value = float(value)
            value = max(float(spec.get("min", value)), value)
            value = min(float(spec.get("max", value)), value)
        elif typ == "bool":
            value = bool(value)

        result[name] = value

    return result
