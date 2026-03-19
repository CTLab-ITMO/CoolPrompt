from typing import Dict, Any

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai.chat_models import ChatOpenAI

from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.utils.utils import get_dataset_split
from coolprompt.utils.var_validation import validate_task
from coolprompt.utils.load_dataset import load_dataset


class AutoPromptingMethod:

    def __init__(
        self,
        config: Dict[str, Any]
    ) -> None:
        self.config = config
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=5,
            check_every_n_seconds=0.5,
            max_bucket_size=10
        )

        self.model = ChatOpenAI(
            model=self.config['model']['name'],
            openai_api_key=self.config['openai_api_key'],
            temperature=self.config['model']['temperature'],
            max_tokens=self.config['model']['max_tokens'],
            timeout=60,
            max_retries=10,
            rate_limiter=rate_limiter
        )

        data_split = self.config['dataset']['configuration']
        data_split = data_split.split('/')
        train_size = int(data_split[0])
        val_size = int(data_split[1])
        test_size = data_split[2]
        if test_size == "all":
            test_size = None
        else:
            test_size = int(test_size)

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
        pass

    def run(
        self,
        start_prompt: str,
        saving_model_answers: bool = False
    ) -> None:
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
