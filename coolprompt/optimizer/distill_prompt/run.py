"""High-level entry point for the DistillPrompt optimization process."""

from typing import List, Tuple

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.distill_prompt.distiller import Distiller


def distillprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    task: str,
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
        model: The language model to use for generating and refining prompts.
        dataset_split: A tuple containing the training and validation data in the
            order: (train_dataset, validation_dataset, train_targets,
            validation_targets).
        evaluator: The evaluator instance used to score prompts.
        task: The type of task to optimize for (e.g., 'classification').
        initial_prompt: The starting prompt to be optimized.
        num_epochs: The number of optimization rounds to perform.
        output_path: The directory path to save logs and cached results.
        use_cache: If True, caches intermediate results to the output path.

    Returns:
        The best prompt found after the optimization process.
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
        task=task,
        base_prompt=initial_prompt,
        num_epochs=num_epochs,
        output_path=output_path,
        use_cache=use_cache,
    )

    return distiller.distillation()
