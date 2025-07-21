"""High-level entry point for the DistillPrompt optimization process."""

from typing import List, Tuple

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.distill_prompt.distiller import Distiller


def distillprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    initial_prompt: str,
    *,
    num_epochs: int = 10,
    output_path: str = './distillprompt_outputs',
    use_cache: bool = True,
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
            perform. Defaults to 10.
        output_path (str, optional): The directory path to save logs and
            cached results. Defaults to './distillprompt_outputs'.
        use_cache (bool, optional): If True, caches intermediate results to
            the output path. Defaults to True.

    Returns:
        str: The best prompt found after the optimization process.
    """
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
