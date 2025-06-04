from typing import List, Tuple
from langchain.llms.base import BaseLanguageModel
from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.reflective_prompt.evoluter import ReflectiveEvoluter


def reflectiveprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    task: str,
    problem_description: str,
    initial_prompt: str = None,
) -> str:
    """Runs ReflectivePrompt evolution.

    Args:
        model (BaseLanguageModel): a LLM to use.
        dataset_split (Tuple[List[str], List[str], List[str], List[str]]):
            train/valid split of dataset and corresponding targets.
        evaluator (Evaluator): evaluator to compute metrics.
        task (str): type of task to optimize for
            (classification or generation).
        problem_description (str): a string that contains
            short description of problem to optimize.
        initial_prompt (str, optional): initial prompt to start evolution from.
            Defaults to None.

    Returns:
        str: best evoluted prompt.
    """
    (
        train_dataset,
        validation_dataset,
        train_targets,
        validation_targets
    ) = dataset_split
    evoluter = ReflectiveEvoluter(
        model=model,
        evaluator=evaluator,
        train_dataset=train_dataset,
        train_targets=train_targets,
        validation_dataset=validation_dataset,
        validation_targets=validation_targets,
        task=task,
        problem_description=problem_description,
        initial_prompt=initial_prompt,
    )
    final_prompt = evoluter.evolution()
    return final_prompt
