from random import sample
from typing import List, Tuple, Optional, override

from langchain_core.language_models import BaseLanguageModel

from coolprompt.data_generator.generator import SyntheticDataGenerator
from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.autoprompting_method import (
    AutoPromptingMethod,
    BenchmarkContext,
)
from coolprompt.optimizer.regps.evoluter import ReGPSEvoluter
from coolprompt.utils.logging_config import logger


def regps(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    problem_description: str,
    initial_prompt: Optional[str] = None,
    use_structured_output: bool = False,
    **kwargs,
) -> str:
    """Runs Re-GPS evolution.

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
    train_dataset, validation_dataset, train_targets, validation_targets = (
        dataset_split
    )
    args = {
        "population_size": 10,
        "num_epochs": 5,
        "output_path": "./regps_outputs",
        "use_cache": True,
        "bad_examples_number": 5,
    }
    args.update(kwargs)
    evoluter = ReGPSEvoluter(
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
        bad_examples_number=args["bad_examples_number"],
        checkpoint_path=args.get('checkpoint_path'),
        use_structured_output=use_structured_output,
    )
    logger.info("Starting Re-GPS optimization...")
    logger.debug(f"Start prompt:\n{initial_prompt}")
    logger.debug(f"Problem description:\n{problem_description}")
    final_prompt = evoluter.evolution()
    logger.info("Re-GPS optimization completed")
    return final_prompt


class ReGPSMethod(AutoPromptingMethod):
    """ReGPS method for auto‑prompting."""

    def optimize(
        self,
        model,
        initial_prompt,
        dataset_split,
        evaluator,
        problem_description,
        *,
        use_structured_output: bool = False,
        **kwargs,
    ):
        return regps(
            model=model,
            dataset_split=dataset_split,
            evaluator=evaluator,
            problem_description=problem_description,
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
        problem_description = ctx.config.get("problem_description")
        mc = ctx.config["method"]
        if problem_description is None:
            generator = SyntheticDataGenerator(
                ctx._system_model,
                use_structured_output=use_structured_output,
            )
            indices = sample(range(0, len(ctx.dataset_split[0])), 5)
            examples = [
                (ctx.dataset_split[0][ind], ctx.dataset_split[2][ind])
                for ind in indices
            ]
            problem_description = generator._generate_problem_description(
                prompt=start_prompt, examples=examples
            )
        return self.optimize(
            ctx.model,
            start_prompt,
            dataset_split=ctx.dataset_split,
            evaluator=ctx.evaluator,
            problem_description=problem_description,
            use_structured_output=use_structured_output,
            population_size=mc.get("population_size", 10),
            num_epochs=mc.get("num_epochs", 5),
            output_path=mc.get("output_path", "./regps_outputs"),
            use_cache=mc.get("use_cache", True),
            bad_examples_number=ctx.config.get("bad_examples_number", 5),
            checkpoint_path=ctx.config.get("checkpoint_path"),
        )

    def is_data_driven(self):
        return True

    @property
    @override
    def name(self):
        return "regps"
