from __future__ import annotations

import yaml
from langchain_core.language_models import BaseLanguageModel

from coolprompt.optimizer.autoprompting_method import AutoPromptingMethod
from coolprompt.optimizer.distill_prompt import DistillMethod
from coolprompt.optimizer.hyper.meta_prompt import HyPERLightMethod
from coolprompt.optimizer.hyper.hyper import HyPERMethod
from coolprompt.optimizer.prompt_compressor import CompressorMethod
from coolprompt.optimizer.reflective_prompt import ReflectiveMethod
from coolprompt.optimizer.regps import ReGPSMethod

_BENCHMARK_IMPL: dict[str, AutoPromptingMethod] = {
    "hyper_light": HyPERLightMethod(),
    "hyper": HyPERMethod(),
    "reflectiveprompt": ReflectiveMethod(),
    "reflective": ReflectiveMethod(),
    "distill": DistillMethod(),
    "compress": CompressorMethod(),
    "regps": ReGPSMethod(),
}


def evaluate_method(
    method: str,
    model: BaseLanguageModel,
    config: dict | str,
    start_prompt: str,
    output_file_path: str = "./method_evaluation_output.yaml",
    saving_model_answers: bool = False,
) -> None:
    """Run a benchmark for an autoprompting method and write scores to YAML.

    Args:
        method: One of
            ``hyper_light``, ``hyper``, ``reflective`` / ``reflectiveprompt``,
            ``distill``, ``compress``, ``regps`` (same names as in
            ``PromptTuner`` / ``validate_method`` where applicable).
        model: LangChain language model used for optimization and evaluation.
        config: Benchmark configuration dict or path to a YAML file.
        start_prompt: Initial prompt string.
        output_file_path: Where to write summary metrics and the final prompt.
        saving_model_answers: If True, persist per-example model outputs on the
            test split (see ``model_answers_output_path`` in config).
    """

    if isinstance(config, str):
        with open(config, "r") as file:
            config = yaml.safe_load(file)

    impl = _BENCHMARK_IMPL.get(method)
    if impl is None:
        raise ValueError(
            f"Unsupported method name: {method}. "
            f"Supported: {sorted(_BENCHMARK_IMPL.keys())}."
        )

    out = impl.run(
        model,
        config,
        start_prompt,
        saving_model_answers=saving_model_answers,
    )

    with open(output_file_path, "w") as file:
        yaml.safe_dump(
            {
                "dataset": config["dataset"]["name"],
                "configuration": config["dataset"]["configuration"],
                "start_prompt": start_prompt,
                "final_prompt": out["final_prompt"],
                "val_score": out["val_score"],
                "test_score": out["test_score"],
            },
            file,
        )
