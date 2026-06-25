"""High-level entry point for the DistillPrompt optimization process."""

from typing import List, Tuple, override

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.optimizer.distill_prompt.distiller import Distiller
from coolprompt.utils.deprecation import warn_deprecated


def distillprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    initial_prompt: str,
    *,
    num_epochs: int = 5,
    output_path: str = "./distillprompt_outputs",
    use_cache: bool = True,
    use_structured_output: bool = False,
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
        use_structured_output (bool, optional): Kept for interface parity
            with other optimizers. DistillPrompt is deprecated and does
            not support structured output, so passing ``True`` raises
            ``NotImplementedError``. Defaults to ``False``.

    Returns:
        str: The best prompt found after the optimization process.

    Raises:
        NotImplementedError: If ``use_structured_output`` is ``True``.
    """

    warn_deprecated("DistillPrompt")
    if use_structured_output:
        raise NotImplementedError(
            "The method is deprecated and does not support structured output"
        )
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
        *,
        use_structured_output: bool = False,
        **kwargs,
    ):
        """Run DistillPrompt through the shared method interface."""
        return distillprompt(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            initial_prompt=initial_prompt,
            use_structured_output=use_structured_output,
            **kwargs,
        )

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
        *,
        use_structured_output: bool = False,
    ) -> str:
        """Run DistillPrompt from a benchmark context."""
        mc = ctx.config.get("method", {})
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            use_structured_output=use_structured_output,
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
