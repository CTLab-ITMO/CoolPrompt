"""LangChain-compatible LLM interface.

Example:
    >>> from language_model.llm import DefaultLLM
    >>> llm = DefaultLLM.init()
    >>> response = llm.invoke("Hello!")
"""

import gc
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer
from langchain_community.llms import VLLM
from langchain_core.language_models.base import BaseLanguageModel
from utils import DEFAULT_MODEL_NAME, DEFAULT_MODEL_PARAMETERS, seed_everything


class DefaultLLM:
    """Default LangChain-compatible LLM using vLLM engine.

    Attributes:
        DEFAULT_MODEL_NAME (str): Name of the default model to use (from utils).
        DEFAULT_MODEL_PARAMETERS (Dict[str, Any]): Default generation parameters.
    """

    @staticmethod
    def init(params: Optional[Dict[str, Any]] = None) -> BaseLanguageModel:
        """Initialize the vLLM-powered LangChain LLM.

        Args:
            params (Optional[Dict[str, Any]], optional): Optional dictionary of parameters to override defaults.

        Returns:
            BaseLanguageModel: Initialized LangChain-compatible language model instance.
        """
        seed_everything(42)
        gc.collect()
        if torch.cuda.is_available:
            torch.cuda.empty_cache()

        generation_params = DEFAULT_MODEL_PARAMETERS.copy()
        if params is not None:
            generation_params.update(params)

        try:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, padding_side="left")
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            return VLLM(
                model=DEFAULT_MODEL_NAME,
                trust_remote_code=True,
                max_new_tokens=generation_params["max_new_tokens"],
                temperature=generation_params["temperature"],
                stop_token_ids=terminators,
                torch_dtype=torch.float16,  # * взял из конфига для проведения экспериментов
            )
        except Exception as e:
            print(f"Failed to load default model: {e}")
            raise
