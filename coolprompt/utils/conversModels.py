from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.chat_models import ChatLlamaCpp
from transformers import pipeline

from coolprompt.utils.conversCore import ConversationBase, SpecMsg
from coolprompt.utils.logging_config import logger

class InvokeMixin:
    def _call_model(self, spec_messages: List[SpecMsg], **kwargs) -> str:
        try:
            resp = self.client.invoke(spec_messages, **kwargs)
            return getattr(resp, "content", str(resp))
        except Exception as exc:
            logger.error(f"[InvokeMixin] Error in {self.__class__.__name__}: {exc}", exc_info=True)
            raise


class ModelOpenAi(InvokeMixin, ConversationBase):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1", history_limit: int = 20, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 256, input_prompts: Optional[List[Dict[str, str]]] = None):
        super().__init__(history_limit=history_limit, input_prompts=input_prompts)
        self.client = ChatOpenAI(model=model, openai_api_key=api_key, openai_api_base=base_url, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

class ModelHFEndpoint(InvokeMixin, ConversationBase):
    def __init__(self, api_token: str, model: str, provider: str = "huggingface", history_limit: int = 20, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 256, input_prompts: Optional[List[Dict[str, str]]] = None):
        super().__init__(history_limit=history_limit, input_prompts=input_prompts)
        llm = HuggingFaceEndpoint(endpoint_url=model, provider=provider, huggingfacehub_api_token=api_token, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        self.client = ChatHuggingFace(llm=llm)


class ModelLlamaCpp(InvokeMixin, ConversationBase):
    def __init__(self, model_path: str, history_limit: int = 20, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 256, n_ctx: int = 2048, n_threads: Optional[int] = None, input_prompts: Optional[List[Dict[str, str]]] = None):
        super().__init__(history_limit=history_limit, input_prompts=input_prompts)
        self.client = ChatLlamaCpp(model_path=model_path, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n_ctx=n_ctx, n_threads=n_threads)


class ModelPipeline(ConversationBase):
    def __init__(self, model_name: str, mode: str = "text-generation", device: int = -1, history_limit: int = 20, input_prompts: Optional[List[Dict[str, str]]] = None, **pipeline_kwargs):
        super().__init__(history_limit=history_limit, input_prompts=input_prompts)
        self.mode = mode
        self.pipeline = pipeline(mode, model=model_name, device=device, **pipeline_kwargs)

    def _call_model(self, spec_messages: List[SpecMsg], **kwargs) -> str:
        last = spec_messages[-1]
        last_prompt = last["content"] if isinstance(last, dict) else last.content
        try:
            result = self.pipeline(last_prompt, **kwargs)
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    if "generated_text" in first: return first["generated_text"]
                    if "text" in first: return first["text"]
            elif isinstance(result, dict):
                if "text" in result: return result["text"]
            raise ValueError(f"incorrect format of pipeline output: {result!r}")
        except Exception as exc:
            logger.error(f"[ModelPipeline] Error during pipeline call: {exc}", exc_info=True)
            raise


def ConversationFactory(mode: str, **kwargs) -> ConversationBase:
    if mode == "openai": return ModelOpenAi(**kwargs)
    if mode in ("hf", "huggingface", "novita"): return ModelHFEndpoint(**kwargs)
    if mode == "llama": return ModelLlamaCpp(**kwargs)
    if mode == "pipeline": return ModelPipeline(**kwargs)
    raise ValueError(f"Unknown mode: {mode}")