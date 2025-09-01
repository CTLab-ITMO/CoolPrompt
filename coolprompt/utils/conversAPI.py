# pylint: disable=logging-fstring-interpolation
import concurrent.futures
from typing import Any, Dict, List, Optional, Callable

from coolprompt.utils.conversCore import ConversationBase    
from coolprompt.utils.conversModels import ConversationFactory   

from coolprompt.utils.logging_config import logger


class ConversationByApi:
    def __init__(self, mode: str, history_limit: int = 20, input_prompts: Optional[List[Dict[str, str]]] = None, stop_criterion: Optional[Callable[[List[Dict[str, str]]], bool]] = None, timeout: Optional[int] = None, **kwargs):
        self.model: ConversationBase = ConversationFactory(mode=mode, history_limit=history_limit, input_prompts=input_prompts, **kwargs)
        self.stop_criterion = stop_criterion or (lambda h: False)
        self.timeout = timeout

    def _call_with_timeout(self, func, *args, **kwargs) -> Any:
        if not self.timeout: return func(*args, **kwargs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"timeout reached={self.timeout} in ConversationByApi")
                raise TimeoutError("reached timeout")

    def send_message(self, prompt: str, **kwargs) -> str:
        try:
            reply = self._call_with_timeout(self.model.send_message, prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error while sending message: {e}")
            raise
        if self.stop_criterion(self.model.get_raw_history()):
            logger.warning("Stop criterion reached in ConversationByApi")
        return reply
    
    def get_raw_history(self) -> List[Dict[str, str]]:
        return self.model.get_raw_history()

    def get_spec_history(self):
        return self.model.get_spec_history()


class MultiConversationByApi:
    def __init__(self):
        self.conversations: Dict[str, ConversationByApi] = {}
        self.stopped: bool = False
        self.last_error: Optional[Exception] = None

    def add_conversation(self, name: str, mode: str, history_limit: int = 20, input_prompts: Optional[List[Dict[str, str]]] = None, stop_criterion: Optional[Callable[[List[Dict[str, str]]], bool]] = None, timeout: Optional[int] = None, **kwargs):
        if name in self.conversations:
            logger.error(f"Conversation '{name}' already exists")
            raise ValueError(f"Conversation with name '{name}' already exists")
        self.conversations[name] = ConversationByApi(mode=mode, history_limit=history_limit, input_prompts=input_prompts, stop_criterion=stop_criterion, timeout=timeout, **kwargs)

    def _check_guard(self):
        if self.stopped: 
            logger.error(f"Conversation stopped. Last error: {self.last_error}")
            raise RuntimeError(f"Conversation stopped. Last error: {self.last_error}")

    def send_message(self, name: str, prompt: str, **kwargs) -> str:
        self._check_guard()
        if name not in self.conversations: 
            logger.error(f"No conversation with name '{name}'")
            raise KeyError(f"No conversation with name '{name}'")
        try:
            reply = self.conversations[name].send_message(prompt, **kwargs)
        except Exception as e:
            self.stopped, self.last_error = True, e
            logger.error(f"Error in {name}: {e}")
            raise

        if self.conversations[name].stop_criterion(self.conversations[name].get_raw_history()):
            self.stopped = True
            logger.warning(f"Stop criterion reached for '{name}'")

        return reply

    def forward_message(self, src: str, dst: str, role: str = "user", **kwargs) -> str:
        self._check_guard()
        if src not in self.conversations or dst not in self.conversations:
            logger.error("Source or destination conversation not found")
            raise KeyError("Source or destination conversation not found")

        src_hist = self.conversations[src].get_raw_history()
        if not src_hist:
            logger.error(f"Conversation {src} has empty history")
            raise ValueError(f"Conversation '{src}' has empty history")

        last_msg = src_hist[-1]
        if last_msg["role"] != "assistant" and role == "user":
            logger.error(f"Last message in {src} is not from assistant")
            raise ValueError(f"Last message in '{src}' is not from assistant")

        logger.info(f"Forwarding message from {src} to {dst}")
        return self.send_message(dst, last_msg["content"], **kwargs)

    def get_history(self, name: str) -> List[Dict[str, str]]:
        if name not in self.conversations:
            logger.error(f"No conversation with name {name}")
            raise KeyError(f"No conversation with name '{name}'") 
        return self.conversations[name].get_raw_history()

    def list_conversations(self) -> List[str]:
        return list(self.conversations.keys())
