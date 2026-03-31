from typing import Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel

from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.utils.utils import get_dataset_split
from coolprompt.utils.var_validation import validate_task
from coolprompt.utils.load_dataset import load_dataset


def _parse_dataset_size(size: str) -> Optional[int]:
    """Returns integer representation of the dataset size.
    Returns None if "all" provided.

    Args:
        size (str): dataset size in string representation.

    Returns:
        Optional[int]: parsed integer value
    """

    if size == "all":
        return None
    return int(size)


class AutoPromptingMethod:

    def __init__(
        self,
        model: BaseLanguageModel,
        config: Dict[str, Any]
    ) -> None:
        """
        Basic interface for AutoPrompting method.

        Attributes:
            model: langchain.BaseLanguageModel class of model to use.
            config: (dict) provided configuration.
            dataset_split: dataset train/val split for optimization process.
            test_dataset: a dataset to use while testing the final prompt.
            test_target: string targets for testing dataset.
            evaluator: evaluator (Evaluator) to compute metrics.
        """
        self.model = BaseLanguageModel
        self.config = config

        data_split = self.config['dataset']['configuration']
        data_split = data_split.split('/')
        train_size = _parse_dataset_size(data_split[0])
        val_size = _parse_dataset_size(data_split[1])
        test_size = _parse_dataset_size(data_split[2])

        train_dataset, train_target = load_dataset(
            self.config['dataset']['name'],
            size=train_size + val_size,
            split='train'
        )

        self.dataset_split = get_dataset_split(
            dataset=train_dataset,
            target=train_target,
            validation_size=val_size / (train_size + val_size),
            train_as_test=self.config.get('train_as_test', False),
        )

        self.test_dataset, self.test_target = load_dataset(
            self.config['dataset']['name'],
            size=test_size,
            split="test"
        )

        task = validate_task(self.config['task'])
        metric = validate_and_create_metric(task, self.config['metric'])
        self.evaluator = Evaluator(self.model, task, metric)

    def _run(self, start_prompt: str) -> str:
        """Inner function. Must be implemented for every AutoPrompting method"""
        pass

    def run(
        self,
        start_prompt: str,
        saving_model_answers: bool = False
    ) -> None:
        """Runs autoprompting optimization process.

        Args:
            start_prompt (str): initial prompt.
            saving_model_answers (bool):
                either to save all model answers for test dataset or not.

        """
        self.final_prompt = self._run(
            start_prompt
        )

        self.final_val_score = self.evaluator.evaluate(
            prompt=self.final_prompt,
            dataset=self.dataset_split[1],
            targets=self.dataset_split[3],
        )

        self.final_test_score = self.evaluator.evaluate(
            prompt=self.final_prompt,
            dataset=self.test_dataset,
            targets=self.test_target,
            save_model_answers=saving_model_answers,
            model_answers_output_path=self.config.get(
                'model_answers_output_path',
                './model_answers.yaml'
            )
        )
