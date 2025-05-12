"""LangChain-compatible LLM interface.

Example:
    >>> from language_model.llm import DefaultLLM
    >>> llm = DefaultLLM.init()
    >>> response = llm.invoke("Hello!")
"""

<<<<<<< HEAD
from typing import Any, Dict
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

DEFAULTS = {
    "model_name": "AnatoliiPotapov/T-lite-instruct-0.1",
    "openai_model_name": "gpt-3.5-turbo",
    "max_new_tokens": 100,
    "temperature": 0.0,  # ? а какую ставить?
}
REQUEST_TIMEOUT = 30


class BaseLLM:
    """Base LLM interface"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", DEFAULTS["model_name"])
        self.max_tokens = config.get("max_tokens", DEFAULTS["max_new_tokens"])
        self.temperature = config.get("temperature", DEFAULTS["temperature"])
        if not isinstance(self.temperature, float) or not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be float between 0.0 and 1.0")
        if self.max_tokens < 0:
            raise ValueError("Max tokens must be positive")

    def invoke(self, prompt: str):
        pass


class HuggingFaceLocalLLM(BaseLLM):
    """Hugging Face Local LLM interface"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def invoke(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens, temperature=self.temperature)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class HuggingFaceRemoteLLM(BaseLLM):
    """Hugging Face Remote LLM interface"""

    def __init__(self, config: Dict[str, Any]):
        if "api_key" not in config:
            raise ValueError("HuggingFaceRemoteLLM requires an API key")
        super().__init__(config)
        self.api_url = config.get("api_url", f"https://api-inference.huggingface.co/models/{self.model_name}")
        self.headers = {"Authorization": f"Bearer {config['api_key']}"}

    def invoke(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": self.max_tokens, "temperature": self.temperature}},
            timeout=REQUEST_TIMEOUT,
        )
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]["generated_text"]
            raise ValueError("Unexpected API response format")
        except (ValueError, KeyError) as e:
            raise ValueError(f"Failed to parse API response: {str(e)}")


class OllamaLLM(BaseLLM):
    """Ollama LLM interface"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=10)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to connect to Ollama: {e}")

    def invoke(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {"num_predict": self.max_tokens, "temperature": self.temperature},
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=REQUEST_TIMEOUT)
        return response.json()["response"]


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI Compatible LLM interface"""

    def __init__(self, config: Dict[str, Any]):
        if "api_key" not in config:
            raise ValueError("OpenAICompatibleLLM requires an API key")
        super().__init__(config)
        self.api_key = config["api_key"]
        self.model_name = config.get("model_name", DEFAULTS["openai_model_name"])
        self.client = OpenAI(base_url=config.get("base_url", "https://api.openai.com/v1"), api_key=self.api_key)

    def invoke(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"OpenAI API error: {str(e)}")
=======
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
>>>>>>> 907d593 (added naive autoprompting and default langchain model)
