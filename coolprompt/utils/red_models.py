
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import json
import re
from coolprompt.utils.conversModels import ConversationFactory
from coolprompt.utils.logging_config import logger
from coolprompt.utils.redConfig import JailbreakPrompts

class BaseModel(ABC):
    def __init__(self, name: str, path2model: str = "", mode: str = "openai", *, temperature: Optional[float] = None, max_tokens: Optional[int] = None, history_limit: Optional[int] = None, api_key: Optional[str] = None, api_token: Optional[str] = None, provider: str = "novita", model_type: str = "base"):
        defaults = JailbreakPrompts.get_model_defaults(model_type)
        self.name = name
        self.path2model = path2model
        self.mode = mode
        self.temperature = temperature if temperature is not None else defaults.get("temperature", 0.7)
        self.max_tokens = max_tokens if max_tokens is not None else defaults.get("max_tokens", 512)
        self.history_limit = history_limit if history_limit is not None else defaults.get("history_limit", 10)
        try:
            conv_args = {"mode": mode, "model": name, "history_limit": self.history_limit, "temperature": self.temperature}
            if mode == "openai":
                conv_args["api_key"] = api_key
                conv_args["max_tokens"] = self.max_tokens
            elif mode in ("hf", "huggingface"):
                conv_args["api_token"] = api_token
                conv_args["provider"] = provider
                conv_args["max_new_tokens"] = self.max_tokens
            elif mode == "llama":
                conv_args["path2model"] = path2model
                conv_args["max_tokens"] = self.max_tokens
            self.conversation_model = ConversationFactory(**conv_args)
        except Exception as e:
            logger.error("Failed to initialize %s: %s", self.__class__.__name__, e)
            raise

    def _apply_system_prompt(self, system_prompt: Optional[str]) -> None:
        if system_prompt:
            self.conversation_model.clear_history()
            self.conversation_model.append("system", system_prompt)

    def _send_message(self, prompt: str) -> str:
        response = self.conversation_model.send_message(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        return response.strip()

    def _execute(self, build_prompt: Callable[[], str], *, system_prompt: Optional[str] = None) -> str:
        if system_prompt: self._apply_system_prompt(system_prompt)
        prompt = build_prompt()
        return self._send_message(prompt)

    @abstractmethod
    def process_request(self, *args, **kwargs) -> Any:
        pass


class AttackModel(BaseModel):
    def __init__(self, name: str, path2model: str = "", mode: str = "openai", *, temperature: Optional[float] = None, max_tokens: Optional[int] = None, history_limit: Optional[int] = None, api_key: Optional[str] = None, api_token: Optional[str] = None, provider: str = "novita"):
        super().__init__(name, path2model, mode, temperature=temperature, max_tokens=max_tokens, history_limit=history_limit, api_key=api_key, api_token=api_token, provider=provider, model_type="attack")

    def generate_attack_prompt(self, target_prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._execute(build_prompt=lambda: JailbreakPrompts.get_attack_instruction(target_prompt), system_prompt=system_prompt)

    def process_request(self, *args, **kwargs) -> str:
        return self.generate_attack_prompt(*args, **kwargs) 


class TargetModel(BaseModel):
    def __init__(self, name: str, path2model: str = "", mode: str = "openai", *, temperature: Optional[float] = None, max_tokens: Optional[int] = None, history_limit: Optional[int] = None, api_key: Optional[str] = None, api_token: Optional[str] = None, provider: str = "novita"):
        super().__init__(name, path2model, mode, temperature=temperature, max_tokens=max_tokens, history_limit=history_limit, api_key=api_key, api_token=api_token, provider=provider, model_type="target")

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._execute(build_prompt=lambda: prompt, system_prompt=system_prompt,)

    def process_request(self, *args, **kwargs) -> str:
        return self.get_response(*args, **kwargs)


class JudgeModel(BaseModel):
    def __init__(self, name: str, path2model: str = "", mode: str = "openai", *, temperature: Optional[float] = None, max_tokens: Optional[int] = None, history_limit: Optional[int] = None, api_key: Optional[str] = None, api_token: Optional[str] = None, provider: str = "novita"):
        super().__init__(name, path2model, mode, temperature=temperature, max_tokens=max_tokens, history_limit=history_limit, api_key=api_key, api_token=api_token, provider=provider, model_type="judge")

    def evaluate(self, original_prompt: str, attack_prompt: str, target_response: str) -> Dict[str, Any]:
        judge_prompt = JailbreakPrompts.get_judge_prompt(original_prompt, attack_prompt, target_response)
        try:
            response = self._execute(build_prompt=lambda: judge_prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info("Jailbreak evaluation: %s", result)
                return result
            return {"success": False, "confidence": 0.0, "category": "unknown", "reasoning": "did not find answer"}
        except Exception as e:
            logger.error("Failed to evaluate jailbreak: %s", e)
            return {"success": False, "confidence": 0.0, "category": "error", "reasoning": "Evaluation failed: %s" % str(e)}

    def process_request(self, *args, **kwargs) -> Dict[str, Any]:
        return self.evaluate(*args, **kwargs)


class RedModel:
    def __init__(self, attack_model: AttackModel, target_model: TargetModel, judge_model: JudgeModel):
        self.attack_model = attack_model
        self.target_model = target_model
        self.judge_model = judge_model
        logger.info("RedModel initialized for jailbreaking attacks")

    def attack(self, prompt: str, system_prompt: Optional[str] = None, max_attempts: int = 3) -> Dict[str, Any]:
        results = []
        for attempt in range(max_attempts):
            logger.info("Attempt %d/%d", attempt + 1, max_attempts)
            try:
                attack_prompt = self.attack_model.generate_attack_prompt(prompt, system_prompt)
                target_response = self.target_model.get_response(attack_prompt, system_prompt)
                evaluation = self.judge_model.evaluate(prompt, attack_prompt, target_response)
                attempt_result = {"attempt": attempt + 1, "original_prompt": prompt, "attack_prompt": attack_prompt, "target_response": target_response, "evaluation": evaluation}
                results.append(attempt_result)
                
                if evaluation.get("success", False):
                    logger.info("Successful jailbreak on attempt %d", attempt + 1)
                    break

            except Exception as e:
                logger.error("Error in attempt %d: %s", attempt + 1, e)
                results.append({"attempt": attempt + 1, "error": str(e), "evaluation": {"success": False}})

        best_result = max(results, key=lambda x: x.get("evaluation", {}).get("confidence", 0))
        return {"all_attempts": results, "best_result": best_result, "total_attempts": len(results), "successful_attempts": sum(1 for r in results if r.get("evaluation", {}).get("success", False))}