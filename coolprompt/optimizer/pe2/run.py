"""High-level entry point for the PE2 optimization process."""

from typing import List, Tuple

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.pe2.proposer import Proposer
from coolprompt.optimizer.pe2.trainer import PE2Trainer
from coolprompt.utils.logging_config import logger


def pe2_optimizer(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    initial_prompt: str,
    **kwargs,
) -> str:
    """Runs PE2 beam-search prompt optimization.

    Args:
        model (BaseLanguageModel): The language model to use.
        dataset_split (Tuple[List[str], List[str], List[str], List[str]]):
            A tuple of (train_dataset, val_dataset,
            train_targets, val_targets).
        evaluator (Evaluator): Evaluator for scoring prompts.
        initial_prompt (str): The starting prompt to optimize.
        **kwargs: Optional overrides for train_steps, n_beam, n_expand,
            batch_size, backtrack, prompt_max_tokens.

    Returns:
        str: The best prompt found after optimization.
    """
    (
        train_dataset,
        val_dataset,
        train_targets,
        val_targets,
    ) = dataset_split

    args = {
        "train_steps": 3,
        "n_beam": 3,
        "n_expand": 4,
        "batch_size": 4,
        "backtrack": True,
        "prompt_max_tokens": 300,
    }
    args.update(kwargs)

    proposer = Proposer(
        model=model,
        prompt_max_tokens=args["prompt_max_tokens"],
    )

    template = evaluator._get_default_template()

    trainer = PE2Trainer(
        model=model,
        evaluator=evaluator,
        proposer=proposer,
        train_dataset=train_dataset,
        train_targets=train_targets,
        val_dataset=val_dataset,
        val_targets=val_targets,
        template=template,
        train_steps=args["train_steps"],
        n_beam=args["n_beam"],
        n_expand=args["n_expand"],
        batch_size=args["batch_size"],
        backtrack=args["backtrack"],
    )

    logger.info("Starting PE2 optimization...")
    logger.debug(f"Start prompt:\n{initial_prompt}")
    final_prompt = trainer.train(initial_prompt)
    logger.info("PE2 optimization completed")
    return final_prompt
