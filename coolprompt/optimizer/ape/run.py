"""High-level entry point for APE optimization."""

from typing import List, Tuple

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.ape.proposer import APEProposer
from coolprompt.optimizer.ape.trainer import APETrainer
from coolprompt.utils.logging_config import logger


def ape_optimizer(
    model: BaseLanguageModel,
    dataset_split: Tuple[
        List[str], List[str], List[str], List[str]
    ],
    evaluator: Evaluator,
    initial_prompt: str,
    **kwargs,
) -> str:
    """Runs APE evaluate-select-paraphrase optimization.

    Uses paraphrase-only proposals (no failure analysis).
    Evaluates on validation data, selects top-k, paraphrases,
    and repeats.

    Args:
        model (BaseLanguageModel): The language model to use.
        dataset_split: A tuple of (train_dataset,
            val_dataset, train_targets, val_targets).
        evaluator (Evaluator): Evaluator for scoring prompts.
        initial_prompt (str): The starting prompt to optimize.
        **kwargs: Optional overrides for train_steps, n_beam,
            n_expand, backtrack, prompt_max_tokens.

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
        "prompt_max_tokens": 300,
    }
    args.update(kwargs)

    proposer = APEProposer(
        model=model,
        prompt_max_tokens=args["prompt_max_tokens"],
    )

    template = evaluator._get_default_template()

    trainer = APETrainer(
        model=model,
        evaluator=evaluator,
        proposer=proposer,
        val_dataset=val_dataset,
        val_targets=val_targets,
        template=template,
        train_steps=args["train_steps"],
        n_beam=args["n_beam"],
        n_expand=args["n_expand"],
    )

    logger.info("Starting APE optimization...")
    logger.debug(f"Start prompt:\n{initial_prompt}")
    final_prompt = trainer.train(initial_prompt)
    logger.info("APE optimization completed")
    return final_prompt
