import logging
from logging.handlers import TimedRotatingFileHandler
import os

import random
import re
from langchain_core.language_models.base import BaseLanguageModel
from typing import List, Tuple, Any
import yaml
from coolprompt.evaluator import Evaluator
from tqdm import tqdm


from datetime import datetime
from zoneinfo import ZoneInfo

moscow_tz = ZoneInfo("Europe/Moscow")
current_time = datetime.now(moscow_tz).strftime("%H:%M:%S")

print("distiller reloaded at", current_time)

class Candidate:
    """Class to represent a prompt candidate with its score."""
    def __init__(self, prompt: str, train_score: float = 0.0):
        self.prompt = prompt
        self.train_score = train_score

class CandidateHistory:
    """Class to manage history of candidate prompts."""
    def __init__(self):
        self.candidates: List[Candidate] = []

    def add(self, candidate: Candidate):
        self.candidates.append(candidate)

    def extend(self, candidates: List[Candidate]):
        self.candidates.extend(candidates)

    def clear(self):
        self.candidates = []

    def get_highest_scorer(self) -> Candidate:
        if not self.candidates:
            raise Exception("No candidates in history")
        return max(self.candidates, key=lambda x: x.train_score)

class TextSampler:
    """Class to sample text examples from a dataset."""
    def __init__(self, texts: List[str], labels: List[str]):
        self.texts = texts
        self.labels = labels

    def sample(self, count: int) -> List[Tuple[str, str]]:
        indices = random.sample(range(len(self.texts)), min(count, len(self.texts)))
        return [(self.texts[i], self.labels[i]) for i in indices]

class PromptTransformer:
    """Class for expanding prompts"""
    
    def __init__(self, model: BaseLanguageModel, sampler: TextSampler):
        self.model = model
        self.sampler = sampler
        
    def aggregate_prompts(self, candidates: List[Candidate], temperature: float = 0.4) -> str:
        def format_prompts(candidates: List[Candidate]) -> str:
            prompts = [cand.prompt for cand in candidates]
            formatted_string = ""
            for i, prompt in enumerate(prompts):
                formatted_string += f"Prompt {i}: {prompt}\n\n"
            return formatted_string 
        
        aggregation_prompt = f"""Below are several prompts intended for the same task:

        {format_prompts(candidates)}

        Your task is to generate one clear and concise prompt that captures the general idea, overall objective, and key instructions conveyed by all of the above prompts.
        Focus on the shared purpose and main concepts without including specific examples or extraneous details.    

        Return only the new prompt, and enclose it with <START> and <END> tags.
        """
        aggregation_prompt = '\n'.join([line.lstrip() for line in aggregation_prompt.split('\n')])
        answer = self.model.invoke(aggregation_prompt, temperature=temperature)
        return self._parse_tagged_text(answer, "<START>", "<END>")
        
    def compress_prompt(self, candidate: Candidate, temperature: float = 0.4) -> str:
        compression_prompt = f"""I want to compress the following zero-shot classifier prompt into a shorter prompt of 2–3 concise sentences that capture its main objective and key ideas from any examples.

        Current prompt: {candidate.prompt}

        Steps:

        Identify the main task or objective.
        Extract the most important ideas illustrated by the examples.
        Combine these insights into a brief, coherent prompt.

        Return only the new prompt, and enclose it with <START> and <END> tags.
        """
        compression_prompt = '\n'.join([line.lstrip() for line in compression_prompt.split('\n')])
        answer = self.model.invoke(compression_prompt, temperature=temperature)
        return self._parse_tagged_text(answer, "<START>", "<END>")
                
    def distill_samples(self, candidate: Candidate, sample_count: int = 5, temperature: float = 0.5) -> str:
        train_samples = self.sampler.sample(sample_count)
        sample_string = self._format_samples(train_samples)
        distillation_prompt = f"""You are an expert prompt engineer.

        Current instruction prompt: {candidate.prompt}

        Training examples: {sample_string}

        Task:
        Analyze the current prompt and training examples to understand common strengths and weaknesses.
        Learn the general insights and patterns without copying any example text.
        Rewrite the instruction prompt to improve clarity and effectiveness while maintaining the original intent.
        Do not include any extraneous explanation or details beyond the revised prompt.

        Return only the new prompt, and enclose it with <START> and <END> tags.
        """
        distillation_prompt = '\n'.join([line.lstrip() for line in distillation_prompt.split('\n')])
        answer = self.model.invoke(distillation_prompt, temperature=temperature)
        return self._parse_tagged_text(answer, "<START>", "<END>")
    
    def generate_prompts(self, candidate: Candidate, n: int = 4, temperature: float = 0.7) -> List[str]:
        generation_prompt = f"""You are an expert in prompt analysis with exceptional comprehension skills.

        Below is my current instruction prompt: {candidate.prompt}

        On the train dataset, this prompt scored {candidate.train_score:0.3f} (with 1.0 being the maximum). 

        Please analyze the prompt's weaknesses and generate an improved version that refines its clarity, focus, and instructional quality. Do not assume any data labels—focus solely on the quality of the prompt.

        Return only the improved prompt, and enclose it with <START> and <END> tags.
        Improved prompt: """
        generation_prompt = '\n'.join([line.lstrip() for line in generation_prompt.split('\n')])
        requests = [generation_prompt] * n
        answers = self.model.batch(requests, temperature=temperature)
        return [self._parse_tagged_text(answer, "<START>", "<END>") for answer in answers]

    @staticmethod
    def _format_samples(samples: List[Tuple[str, str]]) -> str:
        formatted_string = ""
        for i, (input, output) in enumerate(samples):
            formatted_string += f'Example {i + 1}:\n'
            formatted_string += f'Text: \"{input.strip()}\"\nLabel: {output}\n\n'
        return formatted_string
            
    @staticmethod       
    def _parse_tagged_text(text: str, start_tag: str, end_tag: str) -> str:
        start_index = text.find(start_tag)
        if start_index == -1:
            return text
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            return text
        return text[start_index + len(start_tag):end_index].strip()

    def generate_synonyms(self, candidate: Candidate, n: int = 3, temperature: float = 0.7) -> List[str]:
        rewriter_prompt = f"Generate a variation of the following prompt while keeping the semantic meaning.\n\nInput: {candidate.prompt}\n\nOutput:"
        requests = [rewriter_prompt] * n
        responses = self.model.batch(requests, temperature=temperature)
        return [response for response in responses if response]
    
    def convert_to_fewshot(self, candidate: Candidate, sample_count: int = 3) -> str:
        train_samples = self.sampler.sample(sample_count)
        sample_string = self._format_samples(train_samples)
        instruction_prompt = candidate.prompt
        fewshot_prompt = instruction_prompt + '\n\n' + "Examples:\n" + sample_string
        return fewshot_prompt

class Distiller:
    """
    Distiller class for DistillPrompt optimization.

    Attributes:
        model: BaseLanguageModel to use for optimization.
        evaluator: Evaluator to compute metrics.
        train_dataset: Dataset to use while training.
        train_targets: Targets for train dataset.
        validation_dataset: Dataset to use while validating final prompts.
        validation_targets: Targets for validation dataset.
        task: Type of task to optimize for (classification or generation).
        num_epochs: Number of epochs to evaluate.
        output_path: Path to store logs of optimization.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        evaluator: Evaluator,
        train_dataset: List[str],
        train_targets: List[str],
        validation_dataset: List[str],
        validation_targets: List[str],
        task: str,
        base_prompt: str,
        num_epochs: int = 10,
        output_path: str = './distillprompt_outputs'
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.validation_dataset = validation_dataset
        self.validation_targets = validation_targets
        self.task = task
        self.base_prompt=base_prompt
        self.num_epochs = num_epochs
        self.output_path = output_path
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Provides logger setup for DistillPrompt"""
        self.logger = logging.getLogger('Distiller')
        self.logger.setLevel(logging.DEBUG)
        file_handler = TimedRotatingFileHandler(
            filename='DistillPrompt.log',
            when="MIDNIGHT",
            interval=1,
            backupCount=30
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] - %(message)s")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def _evaluate(self, prompt: str, split='train') -> float:
        """Evaluates a given prompt on the specified dataset split."""
        if split == 'train':
            dataset, targets = self.train_dataset, self.train_targets
        else:
            dataset, targets = self.validation_dataset, self.validation_targets
        score = self.evaluator.evaluate(
            prompt=prompt,
            dataset=dataset,
            targets=targets,
            task=self.task
        )
        return score

    def _cache_data(self, data: Any, savepath: os.PathLike) -> None:
        """Writes data to a YAML file."""
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'w') as f:
            yaml.dump(data, f)

    def _make_output_path(self, filename: str) -> str:
        """Creates full path for logging based on current iteration."""
        return os.path.join(self.output_path, f"Iteration{self.iteration}", f"{filename}.yaml")

    def distillation(self) -> str:
        """Provides DistillPrompt optimization operation."""
        self.iteration = 0
        self.logger.info("Starting DistillPrompt optimization...")

        sampler = TextSampler(self.train_dataset, self.train_targets)
        transformer = PromptTransformer(self.model, sampler)
        history = CandidateHistory()

        base_prompt = self.base_prompt
        base_score = self._evaluate(base_prompt)
        base_candidate = Candidate(base_prompt, base_score)
        best_candidate = base_candidate

        for round in tqdm(range(self.num_epochs)):
            self.iteration = round + 1
            self.logger.info(f"Starting round {round}")
            history.clear()
            history.add(best_candidate)

            # Generation
            gen_prompts = transformer.generate_prompts(best_candidate)
            gen_candidates = [Candidate(p, self._evaluate(p)) for p in gen_prompts]
            history.extend(gen_candidates)

            # Distillation
            distilled_prompts = [transformer.distill_samples(cand) for cand in gen_candidates]
            distilled_candidates = [Candidate(p, self._evaluate(p)) for p in distilled_prompts]
            history.extend(distilled_candidates)

            # Compression
            compressed_prompts = [transformer.compress_prompt(cand) for cand in distilled_candidates]
            compressed_candidates = [Candidate(p, self._evaluate(p)) for p in compressed_prompts]
            history.extend(compressed_candidates)

            # Aggregation
            aggregated_prompt = transformer.aggregate_prompts(compressed_candidates)
            aggregated_candidate = Candidate(aggregated_prompt, self._evaluate(aggregated_prompt))
            aggregated_synonyms = transformer.generate_synonyms(aggregated_candidate, n=3)
            
            final_candidates = [Candidate(p, self._evaluate(p)) for p in aggregated_synonyms]
            final_candidates.append(aggregated_candidate)
            history.extend(final_candidates)

            best_candidate = history.get_highest_scorer()
            self.logger.info(f"Best candidate score in round {round}: {best_candidate.train_score}")
            self.logger.info(f"Best candidate prompt: {best_candidate.prompt}")

            # Cache results
            # self._cache_data(
            #     {"prompts": [c.prompt for c in final_candidates], "scores": [c.train_score for c in final_candidates]},
            #     self._make_output_path("round_results")
            # )

        final_prompt = best_candidate.prompt
        final_score = self._evaluate(final_prompt, split='validation')
        self.logger.info(f"Final best prompt score on validation: {final_score}")
        self.logger.info(f"Final best prompt: {final_prompt}")

        # self._cache_data(
        #     {"final_prompt": final_prompt, "final_score": final_score},
        #     os.path.join(self.output_path, "final_results.yaml")
        # )

        return final_prompt
