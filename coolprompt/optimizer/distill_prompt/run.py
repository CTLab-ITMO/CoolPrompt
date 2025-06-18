from typing import List, Tuple
from langchain.llms.base import BaseLanguageModel
from coolprompt.evaluator import Evaluator
from coolprompt.optimizer.distill_prompt.distiller import Distiller


def distillprompt(
    model: BaseLanguageModel,
    dataset_split: Tuple[List[str], List[str], List[str], List[str]],
    evaluator: Evaluator,
    task: str,
    initial_prompt: str,
    **kwargs,
) -> str:
    """Runs DistillPrompt optimization.

    Args:
        model (BaseLanguageModel): a LLM to use.
        dataset_split (Tuple[List[str], List[str], List[str], List[str]]):
            train/valid split of dataset and corresponding targets.
        evaluator (Evaluator): evaluator to compute metrics.
        task (str): type of task to optimize for
            (classification or generation).
        initial_prompt (str): Base prompt for optimization
        **kwargs (dict[str, Any]): other parameters
            (such as num_epochs, output_path).

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
        'num_epochs': 10,
        'output_path': './distillprompt_outputs'
    }
    args.update(kwargs)
    distiller = Distiller(
        model=model,
        evaluator=evaluator,
        train_dataset=train_dataset,
        train_targets=train_targets,
        validation_dataset=validation_dataset,
        validation_targets=validation_targets,
        task=task,
        base_prompt=initial_prompt,
        num_epochs=args['num_epochs'],
        output_path=args['output_path']
    )
    final_prompt = distiller.distillation()
    return final_prompt
