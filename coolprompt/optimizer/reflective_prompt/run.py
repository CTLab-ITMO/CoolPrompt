from typing import List, Optional, Tuple, override

from langchain_core.language_models import BaseLanguageModel

from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.optimizer.reflective_prompt.evoluter import ReflectiveEvoluter
from coolprompt.optimizer.reflective_prompt.coevo_evoluter import CoevoEvoluter
from coolprompt.utils.deprecation import warn_deprecated
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

    warn_deprecated("ReflectivePrompt")
    train_dataset, validation_dataset, train_targets, validation_targets = (
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
        population_size=args["population_size"],
        num_epochs=args["num_epochs"],
        output_path=args["output_path"],
        checkpoint_path=args.get("checkpoint_path"),
        use_cache=args["use_cache"],
    )
    logger.info("Starting ReflectivePrompt optimization...")
    logger.debug(f"Start prompt:\n{initial_prompt}")
    logger.debug(f"Problem description:\n{problem_description}")
    final_prompt = evoluter.evolution()
    logger.info("ReflectivePrompt optimization completed")
    return final_prompt


class ReflectiveMethod(AutoPromptingMethod):
    """Reflective prompting method for auto-prompting."""

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description,
        **kwargs,
    ):
        """Run ReflectivePrompt through the shared method interface."""
        return reflectiveprompt(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            initial_prompt=initial_prompt,
            **kwargs,
        )

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        """Run ReflectivePrompt from a benchmark context."""
        problem_description = ctx.config.get("problem_description")
        if problem_description is None:
            generator = SyntheticDataGenerator(ctx._system_model)
            problem_description = generator._generate_problem_description(
                prompt=start_prompt
            )
        mc = ctx.config["method"]
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=problem_description,
            population_size=mc.get("population_size", 10),
            num_epochs=mc.get("num_epochs", 5),
            output_path=mc.get("output_path", "./reflectiveprompt_outputs"),
            use_cache=mc.get("use_cache", True),
            checkpoint_path=ctx.config.get("checkpoint_path"),
        )

    def is_data_driven(self) -> bool:
        return True

    @property
    @override
    def name(self) -> str:
        return "reflective"


def coevo(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    problem_description: str,
    initial_prompt: Optional[str] = None,
    initial_role: Optional[str] = None,
    initial_constraints: Optional[str] = None,
    use_enhancements: bool = True,
    use_bad_examples: Optional[bool] = None,
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
        use_bad_examples (bool, optional): whether to feed systematic error examples into mutation. If None, follows use_enhancements.
        **kwargs: additional parameters (population_size, num_epochs, output_path, use_cache).

    Returns:
        dict: best evolved prompt with keys:
            - task_description (str): goes into the human message.
            - system_behavior (str): goes into the system message.
            - output_constraints (str): appended to the human message.
    """
    train_dataset, validation_dataset, train_targets, validation_targets = (
        dataset_split
    )
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
        use_bad_examples=use_bad_examples,
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


class CoevoMethod(AutoPromptingMethod):
    """Co-evolution method: structured prompt of three fields
    (task_description, system_behavior, output_constraints).

    ``optimize`` returns the task_description as the main prompt and exposes
    the evolved role and constraints via ``last_role`` / ``last_constraints``,
    which PromptTuner surfaces as final_role / final_constraints.
    """

    last_role: str = ""
    last_constraints: str = ""

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description,
        **kwargs,
    ):
        """Run CoEvo through the shared method interface."""
        result = coevo(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
            initial_prompt=initial_prompt,
            **kwargs,
        )
        self.last_role = result["system_behavior"]
        self.last_constraints = result["output_constraints"]
        return result["task_description"]

    def run_configured_benchmark(
        self,
        ctx: BenchmarkContext,
        start_prompt: str,
    ) -> str:
        """Run CoEvo from a benchmark context."""
        problem_description = ctx.config.get("problem_description")
        if problem_description is None:
            generator = SyntheticDataGenerator(ctx._system_model)
            problem_description = generator._generate_problem_description(
                prompt=start_prompt
            )
        mc = ctx.config["method"]
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=problem_description,
            population_size=mc.get("population_size", 10),
            num_epochs=mc.get("num_epochs", 5),
            output_path=mc.get("output_path", "./coevo_outputs"),
            use_cache=mc.get("use_cache", True),
            use_enhancements=mc.get("use_enhancements", True),
            use_bad_examples=mc.get("use_bad_examples", None),
        )

    def is_data_driven(self) -> bool:
        return True

    @property
    @override
    def name(self) -> str:
        return "coevo"
