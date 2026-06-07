from typing import List, Tuple, override

from langchain_core.language_models import BaseLanguageModel

from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.optimizer.reflective_prompt.evoluter import ReflectiveEvoluter
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
    """Reflective prompting method for auto‑prompting."""

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
