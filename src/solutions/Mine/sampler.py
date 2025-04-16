


import random
from src.data.base.datasets.dataset import BaseDataset
from src.data.base.datasets.generation_dataset import BaseGenerationDataset


class TextSampler:
    """
    Class used for sampling from datataset in non-templated text format
    """
    
    def __init__(self, ds: BaseDataset):
        self.texts = TextSampler._extract_texts_from_ds(ds) # original dataset entries
        self.labels = TextSampler._extract_labels_from_ds(ds)
        self.tuples = list(zip(self.texts, self.labels))
        
    def sample(self, n: int = 5) -> list[tuple[str, str]]:
        """Sample n (text, label) pairs from underlying dataset"""
        return random.sample(self.tuples, n)
    
    @staticmethod
    def _extract_texts_from_prompts(prompts_with_template: list[str]) -> list[str]:
        """Extracts input text from complete prompts"""
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
        
        """
                
        input_ids = [input_id for input_id, _, _ in ds]
        
        prompts_with_template = ds.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        return TextSampler._extract_texts_from_prompts(prompts_with_template)
    
    @staticmethod
    def _extract_labels_from_ds(ds: BaseDataset) -> list[str]:
        """
        Extracts all the dataset labels from their id versions.
        
        """
                
        label_ids = [label_id for _, _, label_id in ds]
        
        if isinstance(ds, BaseGenerationDataset):
            labels = ds.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        else:
            labels = [ds.labels[id] for id in label_ids]
        
        return labels