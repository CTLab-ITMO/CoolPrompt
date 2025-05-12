"""LangChain-compatible LLM interface.

Example:
    >>> from language_model.llm import DefaultLLM
    >>> llm = DefaultLLM.init()
    >>> response = llm.invoke("Hello!")
"""

from typing import Any

import torch
from transformers import AutoTokenizer
from langchain_community.llms import VLLM
from langchain_core.language_models.base import BaseLanguageModel
from coolprompt.utils.default import DEFAULT_MODEL_NAME, DEFAULT_MODEL_PARAMETERS


class DefaultLLM:
    """Default LangChain-compatible LLM using vLLM engine."""

    @staticmethod
    def init(config: dict[str, Any] | None = None) -> BaseLanguageModel:
        """Initialize the vLLM-powered LangChain LLM.

        Args:
            config (dict[str, Any], optional): Optional dictionary of parameters to override defaults.

        Returns:
            BaseLanguageModel: Initialized LangChain-compatible language model instance.
        """
        generation_params = DEFAULT_MODEL_PARAMETERS.copy()
        if config is not None:
            generation_params.update(config)

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, padding_side="left")
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        return VLLM(
            model=DEFAULT_MODEL_NAME,
            trust_remote_code=True,
            stop_token_ids=terminators,
            torch_dtype=torch.float16,
            **generation_params
        )
