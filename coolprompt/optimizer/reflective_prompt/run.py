from typing import List, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel
from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.reflective_prompt.evoluter import ReflectiveEvoluter
from coolprompt.optimizer.reflective_prompt.coevo_evoluter import CoevoEvoluter
from coolprompt.utils.logging_config import logger


def reflectiveprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    problem_description: str,
    initial_prompt: str = None,
    initial_role: str = None,
    evolve_role: bool = True,
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
        "num_epochs": 5,
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
        initial_role=initial_role,
        evolve_role=evolve_role,
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


def coevo(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    problem_description: str,
    initial_prompt: Optional[str] = None,
    initial_role: Optional[str] = None,
    initial_constraints: Optional[str] = None,
    use_enhancements: bool = True,
    **kwargs,
) -> dict:
    """Runs CoevoEvoluter optimization — co-evolves task description, system behavior and output constraints.

    Args:
        model (BaseLanguageModel): a LLM to use.
        dataset_split (Tuple[List[str], List[str], List[str], List[str]]):
            train/valid split of dataset and corresponding targets.
        evaluator (Evaluator): evaluator to compute metrics.
        problem_description (str): short description of the task to optimize.
        initial_prompt (str, optional): initial task description. Defaults to None.
        initial_role (str, optional): initial system behavior. Defaults to None.
        initial_constraints (str, optional): initial output constraints. Defaults to None.
        use_enhancements (bool): whether to use enhanced co-evolution templates. Defaults to True.
        **kwargs: additional parameters (population_size, num_epochs, output_path, use_cache).

    Returns:
        dict: best evolved prompt with keys:
            - task_description (str): goes into the human message.
            - system_behavior (str): goes into the system message.
            - output_constraints (str): appended to the human message.
    """
    (train_dataset, validation_dataset, train_targets, validation_targets) = dataset_split
    args = {
        "population_size": 10,
        "num_epochs": 5,
        "output_path": "./coevo_outputs",
        "use_cache": True,
    }
    args.update(kwargs)
    evoluter = CoevoEvoluter(
        model=model,
        evaluator=evaluator,
        train_dataset=train_dataset,
        train_targets=train_targets,
        validation_dataset=validation_dataset,
        validation_targets=validation_targets,
        problem_description=problem_description,
        initial_prompt=initial_prompt,
        initial_role=initial_role,
        initial_constraints=initial_constraints,
        use_enhancements=use_enhancements,
        population_size=args["population_size"],
        num_epochs=args["num_epochs"],
        output_path=args["output_path"],
        use_cache=args["use_cache"],
    )
    logger.info("Starting CoEvo optimization...")
    logger.debug(f"Start prompt:\n{initial_prompt}")
    logger.debug(f"Problem description:\n{problem_description}")
    evoluter.evolution()
    logger.info("CoEvo optimization completed")
    return {
        "task_description": evoluter.best_prompt_overall or "",
        "system_behavior": evoluter.best_role_overall or "",
        "output_constraints": evoluter.best_constraints_overall or "",
    }
