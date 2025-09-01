from enum import Enum
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional
import os
from coolprompt.utils.prompt_templates.red_templates import JailbreakTemplates

@dataclass(frozen=True)
class Config:
    name: str
    provider: str = "novita"
    mode: str = "hf"
    api_token: Optional[str] = os.getenv("HF_TOKEN")
    temperature: float = 0.7
    max_tokens: int = 512
    history_limit: int = 10
    path2model: str = ""

class ModelCfg(Enum):
    MISTRAL_7B = Config(name="mistralai/Mistral-7B-Instruct-v0.3")
    GPT_OSS_20B = Config(name="gpt-oss-20b", provider="novita")
    # VICUNA_13B = Config(name="lmsys/vicuna-13b-v1.5", provider="featherless-ai") doesn't support task 'conversational'.
    ZEPHYR_7B = Config(name="HuggingFaceH4/zephyr-7b-beta", provider="featherless-ai")
    FALCON_40B = Config(name="tiiuae/falcon-40b-instruct", provider="novita")

    def asdict(self) -> Dict[str, Any]:
        return asdict(self.value)

class JailbreakPrompts:
    DEFAULT_MODEL_CONFIGS = {
        "attack": {
            "temperature": 0.7,
            "max_tokens": 512,
            "history_limit": 10
        },
        "target": {
            "temperature": 0.1,
            "max_tokens": 256,
            "history_limit": 5
        },
        "judge": {
            "temperature": 0.1,
            "max_tokens": 128,
            "history_limit": 3
        }
    }

    @staticmethod
    def get_attack_instruction(target_prompt: str) -> str:  # noqa: D401
        return JailbreakTemplates.ATTACK_INSTRUCTION.format(target_prompt=target_prompt)

    @staticmethod
    def get_judge_prompt(original_prompt: str, attack_prompt: str, target_response: str) -> str:
        return JailbreakTemplates.JUDGE_EVALUATION.format( original_prompt=original_prompt, attack_prompt=attack_prompt, target_response=target_response, )

    SUCCESS_INDICATORS = JailbreakTemplates.SUCCESS_KEYWORDS

    @classmethod
    def get_model_defaults(cls, model_type: str) -> Dict[str, Any]:
        return cls.DEFAULT_MODEL_CONFIGS.get(model_type, {})