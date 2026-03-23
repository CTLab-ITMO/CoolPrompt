"""High-level entry point for OPRO optimization."""

from typing import List, Tuple

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.opro.proposer import OPROProposer
from coolprompt.optimizer.opro.trainer import OPROTrainer
from coolprompt.utils.logging_config import logger


def opro_optimizer(
    model: BaseLanguageModel,
    dataset_split: Tuple[
        List[str], List[str], List[str], List[str]
    ],
    evaluator: Evaluator,
    initial_prompt: str,
    **kwargs,
) -> str:
    """Runs OPRO trajectory-based prompt optimization.

    Uses a meta-prompt showing past (prompt, score) pairs
    sorted worst-to-best, asking the LLM to propose a
    higher-scoring prompt.

    Args:
        model (BaseLanguageModel): The language model to use.
        dataset_split: A tuple of (train_dataset,
            val_dataset, train_targets, val_targets).
        evaluator (Evaluator): Evaluator for scoring prompts.
        initial_prompt (str): The starting prompt to optimize.
        **kwargs: Optional overrides for train_steps, n_beam,
            n_expand, batch_size, backtrack, prompt_max_tokens,
            max_trajectory.

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
        "max_trajectory": 20,
    }
    args.update(kwargs)

    proposer = OPROProposer(
        model=model,
        prompt_max_tokens=args["prompt_max_tokens"],
        max_trajectory=args["max_trajectory"],
    )

    template = evaluator._get_default_template()

    trainer = OPROTrainer(
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

    logger.info("Starting OPRO optimization...")
    logger.debug(f"Start prompt:\n{initial_prompt}")
    final_prompt = trainer.train(initial_prompt)
    logger.info("OPRO optimization completed")
    return final_prompt
