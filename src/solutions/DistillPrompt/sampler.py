"""
Text Sampler Module

This module provides the TextSampler class, which is used for sampling text-label pairs
from a dataset. 
"""


import random
from src.data.base.datasets.dataset import BaseDataset
from src.data.base.datasets.generation_dataset import BaseGenerationDataset


class TextSampler:
    """
    Class used for sampling from a dataset in non-templated text format.

    This class provides functionality to extract and sample text-label pairs
    from a given dataset, which can be useful for tasks that require random
    sampling of data entries.
    """
    
    def __init__(self, ds: BaseDataset):
        """
        Initializes the TextSampler with a dataset.

        Args:
            ds (BaseDataset): The dataset from which to sample text and labels.
        """
        self.texts = TextSampler._extract_texts_from_ds(ds)
        self.labels = TextSampler._extract_labels_from_ds(ds)
        self.tuples = list(zip(self.texts, self.labels))
        
    def sample(self, n: int = 5) -> list[tuple[str, str]]:
        """
        Samples n (text, label) pairs from the underlying dataset.

        Args:
            n (int): The number of (text, label) pairs to sample.

        Returns:
            list[tuple[str, str]]: A list of sampled (text, label) pairs.
        """
        return random.sample(self.tuples, n)
    
    @staticmethod
    def _extract_texts_from_prompts(prompts_with_template: list[str]) -> list[str]:
        """
        Extracts input text from complete prompts.

        Args:
            prompts_with_template (list[str]): A list of prompts containing templates.

        Returns:
            list[str]: A list of extracted input texts.

        Raises:
            ValueError: If no template is found in a prompt.
        """
        texts = []
        for prompt in prompts_with_template:
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
    
    @staticmethod
    def _extract_texts_from_ds(ds: BaseDataset) -> list[str]:
        """
        Extracts all the dataset examples from their merged versions with template.
        
        Example: if template is "<PROMPT>\n\nINPUT:\n<INPUT>\n\nRESPONSE:\n<RESPONSE_PREFIX>"
                 then <INPUT> will be extracted, for example: 'a little too pat for its own good.' 

        Args:
            ds (BaseDataset): The dataset from which to extract texts.

        Returns:
            list[str]: A list of extracted input texts.
        """
                
        input_ids = [input_id for input_id, _, _ in ds]
        
        prompts_with_template = ds.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        return TextSampler._extract_texts_from_prompts(prompts_with_template)
    
    @staticmethod
    def _extract_labels_from_ds(ds: BaseDataset) -> list[str]:
        """
        Extracts all the dataset labels from their ID versions.

        Args:
            ds (BaseDataset): The dataset from which to extract labels.

        Returns:
            list[str]: A list of extracted labels.
        """
        label_ids = [label_id for _, _, label_id in ds]
        
        if isinstance(ds, BaseGenerationDataset):
            labels = ds.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        else:
            labels = [ds.labels[id] for id in label_ids]
        
        return labels
    