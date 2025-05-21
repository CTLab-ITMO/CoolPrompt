"""
Prompt Transformation Framework

Provides main class that implements different prompt transformations using LLM.
"""

from candidate import Candidate
from sampler import TextSampler
from src.utils.eval_utils import LLMWrapper


class PromptTransformer:
    """Class for expanding prompts"""
    
    def __init__(self, model_wrapper: LLMWrapper, sampler: TextSampler):
        """
        Initializes the PromptTransformer with a model wrapper and a text sampler.

        Args:
            model_wrapper (LLMWrapper): An instance of LLMWrapper for interacting with the language model.
            sampler (TextSampler): An instance of TextSampler for sampling text examples.
        """ 
        self.model_wrapper = model_wrapper
        self.sampler = sampler
        
    def aggregate_prompts(self, candidates: list[Candidate], temperature: float = 0.4) -> str:
        """
        Aggregates multiple prompts into a single concise prompt.

        Args:
            candidates (list[Candidate]): A list of Candidate objects containing prompts to aggregate.
            temperature (float): The temperature setting for the language model, controlling randomness.

        Returns:
            str: A new aggregated prompt.
        """      
        def format_prompts(candidates: list[Candidate]) -> str:
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
        answer = self.model_wrapper(aggregation_prompt, temperature=temperature)
        return self._parse_tagged_text(answer, "<START>", "<END>") # type: ignore
        
    
    def compress_prompt(self, candidate: Candidate, temperature: float = 0.4) -> str:
        """
        Compresses a zero-shot classifier prompt into a shorter version.

        Args:
            candidate (Candidate): A Candidate object containing the prompt to compress.
            temperature (float): The temperature setting for the language model, controlling randomness.

        Returns:
            str: A compressed prompt.
        """
        compression_prompt = f"""I want to compress the following zero-shot classifier prompt into a shorter prompt of 2–3 concise sentences that capture its main objective and key ideas from any examples.

        Current prompt: {candidate.prompt}

        Steps:

        Identify the main task or objective.
        Extract the most important ideas illustrated by the examples.
        Combine these insights into a brief, coherent prompt.

        Return only the new prompt, and enclose it with <START> and <END> tags.
        """

        compression_prompt = '\n'.join([line.lstrip() for line in compression_prompt.split('\n')])
        answer = self.model_wrapper(compression_prompt, temperature=temperature)

        return self._parse_tagged_text(answer, "<START>", "<END>") # type: ignore
                
    def distill_samples(self, candidate: Candidate, sample_count: int = 5, temperature: float = 0.5) -> str:
        """
        Distills insights from training samples to improve a prompt.

        Args:
            candidate (Candidate): A Candidate object containing the prompt to distill.
            sample_count (int): The number of training samples to use for distillation.
            temperature (float): The temperature setting for the language model, controlling randomness.

        Returns:
            str: A distilled prompt.
        """
        train_samples: list[tuple[str, str]] = self.sampler.sample(sample_count)
        
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
        answer = self.model_wrapper(distillation_prompt, temperature=temperature)
        
        return self._parse_tagged_text(answer, "<START>", "<END>") # type: ignore

    
    
    def generate_prompts(self, candidate: Candidate, n: int = 4,
                         temperature: float = 0.7, best_of: int = 8) -> list[str]:
        """
        Generates new prompts based on a candidate's score and training dataset examples.

        Args:
            candidate (Candidate): The original prompt candidate.
            n (int): The number of prompts to generate.
            temperature (float): The temperature setting for the language model, controlling randomness.
            best_of (int): The number of best prompts to consider.

        Returns:
            list[str]: A list of new prompts.
        """

        generation_prompt = f"""You are an expert in prompt analysis with exceptional comprehension skills.

        Below is my current instruction prompt: {candidate.prompt}

        On the train dataset, this prompt scored {candidate.train_score:0.3f} (with 1.0 being the maximum). 

        Please analyze the prompt's weaknesses and generate an improved version that refines its clarity, focus, and instructional quality. Do not assume any data labels—focus solely on the quality of the prompt.

        Return only the improved prompt, and enclose it with <START> and <END> tags.
        Improved prompt: """
        
        generation_prompt = '\n'.join([line.lstrip() for line in generation_prompt.split('\n')])
        answers = self.model_wrapper(generation_prompt, n=n, temperature=temperature, best_of=best_of)
        
        return [self._parse_tagged_text(answer, "<START>", "<END>") for answer in answers]

    @staticmethod
    def _format_samples(samples: list[tuple[str, str]]) -> str:
        """
        Formats training samples into a string representation:
        turns [("Input1", "Out1"), ("Input2", "Out2")] into
            Example 1:
            Text: Input1
            Label: Out1
            
            Example 2:
            Text: Input2
            Label: Out2

        Args:
            samples (list[tuple[str, str]]): A list of tuples containing input and output pairs.

        Returns:
            str: A formatted string of examples.
        """
        formatted_string = ""
        for i, (input, output) in enumerate(samples):
                formatted_string += f'Example {i + 1}:\n'
                formatted_string += f'Text: \"{input.strip()}\"\nLabel: {output}\n\n'
        
        return formatted_string
            
    @staticmethod       
    def _parse_tagged_text(text: str, start_tag: str, end_tag: str) -> str:
        """
        Parses text that is tagged with start and end tags.

        Args:
            text (str): The text to parse.
            start_tag (str): The start tag to look for.
            end_tag (str): The end tag to look for.

        Returns:
            str: The text enclosed between the start and end tags.
        """
        start_index = text.find(start_tag)
        if start_index == -1:
            return text
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            return text
        return text[start_index + len(start_tag):end_index].strip()


    def generate_synonyms(self, candidate: Candidate,  n: int = 3,
                          temperature: float = 0.7, best_of: int = 8) -> list[str]:
        """
        Generates synonyms for a prompt.

        Args:
            candidate (Candidate): A Candidate object containing the prompt to generate synonyms for.
            n (int): The number of synonyms to generate.
            temperature (float): The temperature setting for the language model, controlling randomness.
            best_of (int): The number of best synonyms to consider.

        Returns:
            list[str]: A list of synonym prompts.
        """
        rewriter_prompt = f"Generate a variation of the following prompt while keeping the semantic meaning.\n\nInput: {candidate.prompt}\n\nOutput:"
        new_prompts = self.model_wrapper(rewriter_prompt, n=n, temperature=temperature, best_of=best_of)
        new_prompts = [x for x in new_prompts if x]
        return new_prompts
    
    def convert_to_fewshot(self, candidate: Candidate, sample_count: int = 3) -> str:
        """
        Converts a prompt into a few-shot format with examples.

        Args:
            candidate (Candidate): A Candidate object containing the prompt to convert.
            sample_count (int): The number of examples to include.

        Returns:
            str: A few-shot formatted prompt with examples.
        """
        train_samples: list[tuple[str, str]] = self.sampler.sample(sample_count)
        
        sample_string = self._format_samples(train_samples)
        
        instruction_prompt = candidate.prompt
        
        fewshot_prompt = instruction_prompt + '\n\n' + "Examples:\n" + sample_string
        
        return fewshot_prompt
        