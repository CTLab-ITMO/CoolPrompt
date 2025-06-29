"""Prompt Transformation Framework.

Provides the PromptTransformer class, which implements various strategies for
refining and generating prompts using a Large Language Model (LLM).
This includes methods for compression, distillation,
aggregation, and synonym generation.
"""
from typing import List

from langchain_core.language_models.base import BaseLanguageModel

from coolprompt.utils.prompt_templates import distillprompt_templates
from coolprompt.optimizer.distill_prompt.candidate import Candidate
from coolprompt.optimizer.distill_prompt.utils import TextSampler


class PromptTransformer:
    """Implements various transformations for prompt engineering."""

    def __init__(self, model: BaseLanguageModel, sampler: TextSampler) -> None:
        """Initializes the PromptTransformer.

        Args:
            model (BaseLanguageModel): The language model for transformations.
            sampler (TextSampler): The sampler for training data.
        """
        self.model = model
        self.sampler = sampler

    def aggregate_prompts(
        self, candidates: List[Candidate], temperature: float = 0.4
    ) -> str:
        """Aggregates multiple prompts into a single concise prompt.

        Args:
            candidates (List[Candidate]): List of candidate prompts to
                aggregate.
            temperature (float, optional): Temperature for model generation.
                Defaults to 0.4.

        Returns:
            str: The aggregated prompt.
        """
        formatted_prompts = self._format_prompts_for_aggregation(candidates)
        aggregation_prompt = distillprompt_templates.AGGREGATION_PROMPT.format(
            formatted_prompts=formatted_prompts
        )
        answer = self.model.invoke(aggregation_prompt, temperature=temperature)
        return self._parse_tagged_text(str(answer), "<START>", "<END>")

    def compress_prompts(
        self, candidates: List[Candidate], temperature: float = 0.4
    ) -> List[str]:
        """Compresses multiple prompts into shorter versions.

        Args:
            candidates (List[Candidate]): List of candidate prompts to
                compress.
            temperature (float, optional): Temperature for model generation.
                Defaults to 0.4.

        Returns:
            List[str]: List of compressed prompts.
        """
        request_prompts = []
        for candidate in candidates:
            compression_prompt = distillprompt_templates.COMPRESSION_PROMPT
            compression_prompt = compression_prompt.format(
                candidate_prompt=candidate.prompt
            )
            request_prompts.append(compression_prompt)

        answers = self.model.batch(request_prompts, temperature=temperature)
        return [
            self._parse_tagged_text(answer, "<START>", "<END>")
            for answer in answers
        ]

    def distill_samples(
        self, candidates: List[Candidate], sample_count: int = 5,
        temperature: float = 0.5
    ) -> List[str]:
        """Distills insights from training samples to improve prompts.

        Args:
            candidates (List[Candidate]): List of candidate prompts to
                distill.
            sample_count (int, optional): Number of samples to use.
                Defaults to 5.
            temperature (float, optional): Temperature for model generation.
                Defaults to 0.5.

        Returns:
            List[str]: List of distilled prompts.
        """
        request_prompts = []
        for candidate in candidates:
            train_samples = self.sampler.sample(sample_count)
            sample_string = self._format_samples(train_samples)
            prompt = distillprompt_templates.DISTILLATION_PROMPT
            distillation_prompt = prompt.format(
                candidate_prompt=candidate.prompt, sample_string=sample_string
            )
            request_prompts.append(distillation_prompt)

        answers = self.model.batch(request_prompts, temperature=temperature)
        return [
            self._parse_tagged_text(answer, "<START>", "<END>")
            for answer in answers
        ]

    def generate_prompts(
        self, candidate: Candidate, n: int = 4, temperature: float = 0.7
    ) -> List[str]:
        """Generates new prompts based on a candidate's score.

        Args:
            candidate (Candidate): The candidate prompt to base generation
                on.
            n (int, optional): Number of prompts to generate.
                Defaults to 4.
            temperature (float, optional): Temperature for model generation.
                Defaults to 0.7.

        Returns:
            List[str]: List of generated prompts.
        """
        generation_prompt = distillprompt_templates.GENERATION_PROMPT.format(
            candidate_prompt=candidate.prompt,
            train_score=candidate.train_score
        )
        requests = [generation_prompt] * n
        answers = self.model.batch(requests, temperature=temperature)
        return [
            self._parse_tagged_text(answer, "<START>", "<END>")
            for answer in answers
        ]

    def generate_synonyms(
        self, candidate: Candidate, n: int = 3, temperature: float = 0.7
    ) -> List[str]:
        """Generates semantic variations of a given prompt.

        Args:
            candidate (Candidate): The candidate prompt to generate synonyms
                for.
            n (int, optional): Number of synonyms to generate. Defaults to 3.
            temperature (float, optional): Temperature for model generation.
                Defaults to 0.7.

        Returns:
            List[str]: List of synonym prompts.
        """
        rewriter_prompt = distillprompt_templates.REWRITER_PROMPT.format(
            candidate_prompt=candidate.prompt
        )
        requests = [rewriter_prompt] * n
        responses = self.model.batch(requests, temperature=temperature)
        return [response for response in responses if response]

    def convert_to_fewshot(
        self, candidate: Candidate, sample_count: int = 3
    ) -> str:
        """Converts a zero-shot prompt into a few-shot format with examples.

        Args:
            candidate (Candidate): The candidate prompt to convert.
            sample_count (int, optional): Number of examples to include.
                Defaults to 3.

        Returns:
            str: The few-shot formatted prompt.
        """
        train_samples = self.sampler.sample(sample_count)
        sample_string = self._format_samples(train_samples)
        return f"{candidate.prompt}\n\nExamples:\n{sample_string}"

    @staticmethod
    def _format_prompts_for_aggregation(candidates: List[Candidate]) -> str:
        """Formats a list of candidate prompts for the aggregation prompt.

        Args:
            candidates (List[Candidate]): List of candidate prompts to format.

        Returns:
            str: Formatted string of prompts for aggregation.
        """
        return "\n\n".join(
            [f"Prompt {i}: {cand.prompt}" for i, cand in enumerate(candidates)]
        )

    @staticmethod
    def _format_samples(samples: List[tuple[str, str]]) -> str:
        """Formats training samples into a string for few-shot examples.

        Args:
            samples (List[tuple[str, str]]): List of training samples as
                input-output pairs.

        Returns:
            str: Formatted string of training samples.
        """
        formatted_strings = []
        for i, (text_input, output) in enumerate(samples):
            formatted_strings.append(
                f'Example {i + 1}:\n'
                f'Text: "{text_input.strip()}"\n'
                f'Label: {output}'
            )
        return "\n\n".join(formatted_strings)

    @staticmethod
    def _parse_tagged_text(text: str, start_tag: str, end_tag: str) -> str:
        """Parses text enclosed within start and end tags.

        Args:
            text (str): The text to parse.
            start_tag (str): The starting tag to look for.
            end_tag (str): The ending tag to look for.

        Returns:
            str: Text between tags, or original text if tags not found.
        """
        start_index = text.find(start_tag)
        if start_index == -1:
            return text

        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            return text

        return text[start_index + len(start_tag):end_index].strip()
