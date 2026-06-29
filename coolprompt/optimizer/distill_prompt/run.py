"""High-level entry point for the DistillPrompt optimization process."""

from typing import List, Tuple, override, Optional

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.optimizer.distill_prompt.distiller import Distiller
from coolprompt.utils.deprecation import warn_deprecated

from coolprompt.optimizer.autoprompting_method import TelemetryCallback

def distillprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    initial_prompt: str,
    *,
    num_epochs: int = 5,
    output_path: str = "./distillprompt_outputs",
    use_cache: bool = True,
    telemetry_callback: Optional[TelemetryCallback] = None,
) -> str:
    """Runs the full DistillPrompt optimization process.

    This function serves as a convenient wrapper around the Distiller class,
    simplifying the setup and execution of a prompt optimization task.

    Args:
        model (BaseLanguageModel): The language model to use for generating
            and refining prompts.
        dataset_split (Tuple[List[str], List[str], List[str], List[str]]): A
            tuple containing the training and validation data in the order:
            (train_dataset, validation_dataset, train_targets,
            validation_targets).
        evaluator (Evaluator): The evaluator instance used to score prompts.
        initial_prompt (str): The starting prompt to be optimized.
        num_epochs (int, optional): The number of optimization rounds to
            perform. Defaults to 5.
        output_path (str, optional): The directory path to save logs and
            cached results. Defaults to './distillprompt_outputs'.
        use_cache (bool, optional): If True, caches intermediate results to
            the output path. Defaults to True.

    Returns:
        str: The best prompt found after the optimization process.
    """

    warn_deprecated("DistillPrompt")
    (
        train_dataset,
        validation_dataset,
        train_targets,
        validation_targets,
    ) = dataset_split

    distiller = Distiller(
        model=model,
        evaluator=evaluator,
        train_dataset=train_dataset,
        train_targets=train_targets,
        validation_dataset=validation_dataset,
        validation_targets=validation_targets,
        base_prompt=initial_prompt,
        num_epochs=num_epochs,
        output_path=output_path,
        use_cache=use_cache,
        telemetry_callback=telemetry_callback,
    )

    return distiller.distillation()


class DistillMethod(AutoPromptingMethod):
    """Distillation‑based method for auto‑prompting."""

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description=None,
        **kwargs,
    ):
        """Run DistillPrompt through the shared method interface."""

        telemetry_callback = kwargs.pop("telemetry_callback", None)

        return distillprompt(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            initial_prompt=initial_prompt,
            telemetry_callback=telemetry_callback,
            **kwargs,
        )

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        """Run DistillPrompt from a benchmark context."""
        mc = ctx.config.get("method", {})
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            num_epochs=mc.get("num_epochs", 5),
            output_path=mc.get("output_path", "./distillprompt_outputs"),
            use_cache=mc.get("use_cache", True),
        )

    def is_data_driven(self):
        return True

    @property
    @override
    def name(self):
        return "distill"
