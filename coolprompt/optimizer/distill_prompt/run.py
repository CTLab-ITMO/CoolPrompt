from typing import List, Tuple
from langchain.llms.base import BaseLanguageModel
from coolprompt.evaluator import Evaluator


def distillprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    task: str,
    initial_prompt: str,
    **kwargs,
) -> str:
    """Runs ReflectivePrompt evolution.

    Args:
        model (BaseLanguageModel): a LLM to use.
        dataset_split (Tuple[List[str], List[str], List[str], List[str]]):
            train/valid split of dataset and corresponding targets.
        evaluator (Evaluator): evaluator to compute metrics.
        task (str): type of task to optimize for
            (classification or generation).
        initial_prompt (str, optional): initial prompt to start optimization from.
        **kwargs (dict[str, Any]): other parameters
            (such as population_size, num_epochs, output_path, use_cache).

    Returns:
        str: best optimized prompt.
    """
    (
        train_dataset,
        validation_dataset,
        train_targets,
        validation_targets
    ) = dataset_split
    args = {
        'num_epochs': 6,
        'output_path': './reflectiveprompt_outputs',
    }
    args.update(kwargs)
    distiller = PromptDistiller(
        model=model,
        evaluator=evaluator,
        train_dataset=train_dataset,
        train_targets=train_targets,
        validation_dataset=validation_dataset,
        validation_targets=validation_targets,
        task=task,
        initial_prompt=initial_prompt,
        num_epochs=args['num_epochs'],
    )
    final_prompt = distiller.distill()
    return final_prompt
