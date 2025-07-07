from typing import List, Tuple
from langchain.llms.base import BaseLanguageModel
from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.reflective_prompt.evoluter import ReflectiveEvoluter
from coolprompt.utils.logging_config import logger


def reflectiveprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    problem_description: str,
    initial_prompt: str = None,
    **kwargs,
) -> str:
    """Runs ReflectivePrompt evolution.

    Args:
        model (BaseLanguageModel): a LLM to use.
        dataset_split (Tuple[List[str], List[str], List[str], List[str]]):
            train/valid split of dataset and corresponding targets.
        evaluator (Evaluator): evaluator to compute metrics.
        task (Task): type of task to optimize for.
        problem_description (str): a string that contains
            short description of problem to optimize.
        initial_prompt (str, optional): initial prompt to start evolution from.
            Defaults to None.
        **kwargs (dict[str, Any]): other parameters
            (such as population_size, num_epochs, output_path, use_cache).

    Returns:
        str: best evoluted prompt.
    """
    (train_dataset, validation_dataset, train_targets, validation_targets) = (
        dataset_split
    )
    args = {
        "population_size": 10,
        "num_epochs": 10,
        "output_path": "./reflectiveprompt_outputs",
        "use_cache": True,
    }
    args.update(kwargs)
    evoluter = ReflectiveEvoluter(
        model=model,
        evaluator=evaluator,
        train_dataset=train_dataset,
        train_targets=train_targets,
        validation_dataset=validation_dataset,
        validation_targets=validation_targets,
        problem_description=problem_description,
        initial_prompt=initial_prompt,
        population_size=args["population_size"],
        num_epochs=args["num_epochs"],
        output_path=args["output_path"],
        use_cache=args["use_cache"],
    )
    logger.info("Starting ReflectivePrompt optimization...")
    logger.debug(f"Start prompt:\n{initial_prompt}")
    logger.debug(f"Problem description:\n{problem_description}")
    final_prompt = evoluter.evolution()
    logger.info("ReflectivePrompt optimization completed")
    return final_prompt
