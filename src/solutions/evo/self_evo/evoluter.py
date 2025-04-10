import os
from typing import List, Tuple
import numpy as np
from scipy.special import softmax
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from src.evaluation.evaluator import BaseNLPEvaluator
from src.solutions.evo.base import Evoluter, Prompt


class PromptHistory:

    def __init__(self, max_size: int = 3) -> None:
        self._hist = []
        self.max_size = max_size

    def append(
        self,
        par1: Prompt,
        par2: Prompt,
        child: Prompt,
        hint: str
    ) -> None:
        self._hist.append((par1, par2, child, hint))

        if len(self._hist) > self.max_size:
            self._hist = self._hist[1:]

    def get(self) -> List[Tuple[Prompt, Prompt, Prompt, str]]:
        return self._hist

    def __len__(self) -> int:
        return len(self._hist)

    def empty(self) -> bool:
        return len(self._hist) == 0

    def _to_str(self, record: Tuple[Prompt, Prompt, Prompt, str]) -> str:
        return f"""
    Parent1: {record[0]}
    Parent2: {record[1]}
    Hint: {record[3]}
    Child: {record[2]}
    """

    def __str__(self) -> str:
        return "\n\n".join([self._to_str(record) for record in self._hist])


class SelfEvoluter(Evoluter):

    def __init__(
        self,
        model_name: str,
        dataset: str,
        evaluator: BaseNLPEvaluator,
        metric: str,
        population_num: int = 10,
        num_epochs: int = 10,
        output_path: str = './outputs',
        use_cache: bool = True,
        history_size: int = 3,
    ) -> None:
        model = LLM(
            model=model_name,
            dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            evaluator=evaluator,
            metric=metric,
            use_cache=use_cache,
        )

        self.population_num = population_num
        self.num_epochs = num_epochs
        self.config_filename = 'config.yaml'
        self.output_path = output_path

        self._history = PromptHistory(max_size=history_size)

        self._mutation_template = self._read_yaml_data(
            self.config_filename,
            key='mutation'
        )
        self._mutation_with_hint_template = self._read_yaml_data(
            self.config_filename,
            key='mutation_with_hint_v2'
        )
        self._hint_template = self._read_yaml_data(
            self.config_filename,
            key='hint_v2'
        )

    def _selection(
        self,
        population: List[Prompt],
        n: int,
        **kwargs
    ) -> List[Prompt]:
        scores = np.array([prompt.score for prompt in population])
        if 'temperature' in kwargs:
            scores /= kwargs['temperature']
        probas = softmax(scores)
        return np.random.choice(population, size=n, p=probas, replace=False)

    def _mutation(
        self,
        population: List[Prompt],
        verbose: bool = False
    ) -> Prompt:
        parents = self._selection(population, n=2)
        if self._history.empty():
            mutation_prompt = self._mutation_template
            hint = ""
        else:
            hint_request = self._hint_template.replace(
                '<HISTORY>',
                str(self._history)
            )
            output = self._llm_query(
                hint_request,
                verbose=verbose,
                temperature=0.6
            )
            hint = self._single_out_output(output, bracket='<hint>')
            self.logger.info(f"hint: {hint}")
            mutation_prompt = self._mutation_with_hint_template.replace(
                "<HINT>",
                hint
            )

        mutation_prompt = mutation_prompt.replace(
            '<PARENT1>',
            parents[0].text
        )
        mutation_prompt = mutation_prompt.replace(
            '<PARENT2>',
            parents[1].text
        )

        if verbose:
            self.logger.info("=" * 50)
            self.logger.info("")
            self.logger.info(
                f"Parent A: {parents[0].text}\t{parents[0].score}"
            )
            self.logger.info(
                f"Parent B: {parents[1].text}\t{parents[1].score}"
            )

        output = self._llm_query(
            mutation_prompt,
            verbose=verbose,
            temperature=0.6
        )
        final_prompt = self._single_out_output(output)
        final_prompt = Prompt(final_prompt)
        self._evaluate(final_prompt)
        if verbose:
            self.logger.info(
                f"Child prompt: {final_prompt.text}\t{final_prompt.score}"
            )
            self.logger.info("")
            self.logger.info("=" * 50)
        self._history.append(parents[0], parents[1], final_prompt, hint)
        return final_prompt

    def _single_out_output(
        self,
        model_output: str,
        bracket: str = '<prompt>'
    ) -> str:
        closing_bracket = bracket[0] + '/' + bracket[1:]
        parts = model_output.split(bracket)
        if len(parts) > 1:
            prompt = parts[-1].split(closing_bracket)[0]
            prompt = prompt.strip()
            return prompt
        else:
            if (
                model_output.startswith("\"")
                and model_output.endswith("\"")
            ):
                model_output = model_output[1:-1]
            return model_output

    def _llm_query(self, request: str, verbose: bool = False, **config) -> str:
        sampling_params = {
            "max_tokens": 250,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        sampling_params.update(**config)
        sampling_params = SamplingParams(**sampling_params)

        answers = self.model.generate(
            prompts=[request],
            sampling_params=sampling_params,
            use_tqdm=verbose
        )

        result = answers[0].outputs[0].text
        return result

    def _mutate(
        self,
        population: List[Prompt],
        epoch: int = None,
        verbose: bool = False
    ) -> List[Prompt]:
        offspring = []
        for pop in range(self.population_num):
            if verbose:
                self.logger.info(f"Epoch {epoch}, Pop {pop}")
            offspring.append(self._mutation(population, verbose=verbose))
        return offspring

    def evolution(self) -> None:
        population = self._init_pop(use_cache=self.use_cache)

        for epoch in range(self.num_epochs):
            offspring = self._mutate(population, epoch, verbose=True)
            population = self._selection(
                population + offspring,
                n=self.population_num,
                temperature=1e-2
            )
            population = self._reranking(population)
            self._cache_population(
                population,
                os.path.join(
                    self.output_path,
                    f"epoch{epoch}_population.yaml"
                )
            )
