import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils
from src.solutions.Protegi.scorers import Cached01Scorer
from src.utils.eval_utils import Infer


class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer : Cached01Scorer, infer_wrapper: Infer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.infer_wrapper = infer_wrapper
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts):
        pass


class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """

    def _sample_error_str(self, texts, labels, preds, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        error_string = ''
        num_errors = 0
        error_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            error_string += f'## Example {error_idx + 1}\n'
            error_string += f'Text: \"{t.strip()}\"\nLabel: {l}\nPrediction: {p}\n\n'
            error_idx += 1
        return error_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index + len(end_tag):]
        return texts

    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1):
        """ Get "gradients" for a prompt based on the error string."""
        gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt.

        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        model_ans = self.infer_wrapper(gradient_prompt, n=n)[0]
        feedbacks = self.parse_tagged_text(model_ans, "<START>", "<END>")
        print("feedbacks length: ", len(feedbacks))
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.

        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        model_answer = self.infer_wrapper(transformation_prompt, n=n)[0]
        new_prompts = self.parse_tagged_text(model_answer, "<START>", "<END>")
        print("New prompts length: ", len(new_prompts))
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        # TODO: modify vllm for n > 1
        new_instructions = self.infer_wrapper(rewriter_prompt, n=n)[0]
        new_instructions = [x for x in new_instructions if x]
        return new_instructions

    def get_gradients(self, task_section, texts, labels, preds):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string = self._sample_error_str(
                texts, labels, preds, n=self.opt['errors_per_gradient'])
            
            gradients = self._get_gradients(
                task_section, error_string, self.opt['gradients_per_error'], n=1)
            
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks
    
    def _extract_texts(self, prompts_with_system):
        """Extracts input text from complete prompts"""
        texts = []
        for prompt in prompts_with_system:
            # Check for the first template: input:\n<INPUT>\n\nResponse
            input_start = prompt.lower().find("input:\n")
            if input_start != -1:
                input_end = prompt.find("\n\nResponse", input_start)
                if input_end != -1:
                    texts.append(prompt[input_start + 7:input_end].strip())
                    continue

            # Check for the second template: INPUT:\n<INPUT>\n\nRESPONSE
            input_start = prompt.find("INPUT:\n")
            if input_start != -1:
                input_end = prompt.find("\n\nRESPONSE", input_start)
                if input_end != -1:
                    texts.append(prompt[input_start + 7:input_end].strip())
                    continue

            raise ValueError("No template found for {}".format(prompt))

        return texts

    def expand_candidates(self, prompts):
        """ Expand a list of prompts by generating gradient-based successors and
            synonyms for each section.
            Prompts are without the system part

        """
        # minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):

            task_section = prompt
            # evaluate prompt on minibatch

            prompts_with_system, labels, preds = self.scorer.get_predictions(prompt, self.infer_wrapper)

            # texts are the <INPUT>s
            texts = self._extract_texts(prompts_with_system)
            
            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(task_section, texts, labels, preds)
                new_task_sections = []
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(
                        task_section, error_string, feedback, self.opt['steps_per_gradient'])
                    new_task_sections += tmp

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt['mc_samples_per_step'])
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections))  # dedup
            tmp_new_prompts = new_sections

            # filter a little
            if len(new_sections) > self.opt['max_expansion_factor']:
                tmp_new_prompts = random.sample(tmp_new_prompts,
                                                    k=self.opt['max_expansion_factor'])
                    
            new_prompts += tmp_new_prompts

        new_prompts += prompts  # add originals
        new_prompts = list(set(new_prompts))  # dedup

        return new_prompts

    def score_candidates(self, prompts):
        """ Score a list of prompts.

        They should be task_only prompts, e.g. without system (format) instruction
        """
                

        # evaluator_fn is BruteForceEvaluator and such
        evals = self.evaluator_fn(
            prompts,
            scorer=self.scorer,
            rounds=self.opt['eval_rounds'],
            num_prompts_per_round=self.opt['eval_prompts_per_round'],
            samples_per_eval=self.opt['samples_per_eval']
        )
        return evals
