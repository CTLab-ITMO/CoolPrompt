from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Union

from langchain.schema import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

SpecMsg = Union[BaseMessage, Dict[str, str]]

class ConversationBase(ABC):
    def __init__(self, history_limit: int = 20, input_prompts: Optional[Iterable[Dict[str, str]]] = None, use_langchain: bool = True):
        self.use_langchain = use_langchain
        self.raw_history: Deque[Dict[str, str]] = deque(maxlen=history_limit)
        self.spec_history: Deque[SpecMsg] = deque(maxlen=history_limit)
        for d in (input_prompts or []):
            d = {"role": d["role"], "content": str(d["content"])}
            self.raw_history.append(d)
            self.spec_history.append(self.build_message(d["role"], d["content"], self.use_langchain))

    @staticmethod
    def lc_from_dict(message: Dict[str, str]) -> BaseMessage:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "user": return HumanMessage(content=content)
        if role == "assistant": return AIMessage(content=content)
        if role == "system": return SystemMessage(content=content)
        raise ValueError(f"Unsupported role: {role}")

    @staticmethod
    def dict_from_lc(message: BaseMessage) -> Dict[str, str]:
        if isinstance(message, HumanMessage): role = "user"
        elif isinstance(message, AIMessage): role = "assistant"
        elif isinstance(message, SystemMessage): role = "system"
        else: raise ValueError(f"Unsupported type: {type(message)}")
        return {"role": role, "content": str(message.content)}

    @staticmethod
    def build_message(role: str, content: str, use_langchain: bool) -> SpecMsg:
        if use_langchain: return ConversationBase.lc_from_dict({"role": role, "content": content})
        return {"role": role, "content": content}

    def set_use_langchain(self, value: bool):
        if self.use_langchain == value: return
        self.use_langchain = value
        maxlen = self.spec_history.maxlen
        self.spec_history = deque((self.build_message(d["role"], d["content"], self.use_langchain) for d in self.raw_history), maxlen=maxlen)

    def clear_history(self):
        self.raw_history.clear()
        self.spec_history.clear()

    def get_raw_history(self) -> List[Dict[str, str]]:
        return list(self.raw_history)

    def get_spec_history(self) -> List[SpecMsg]:
        return list(self.spec_history)

    def append(self, role: str, content: str):
        d = {"role": role, "content": str(content)}
        self.raw_history.append(d)
        self.spec_history.append(self.build_message(role, d["content"], self.use_langchain))

    def send_message(self, prompt: str, **kwargs) -> str:
        user_spec = self.build_message("user", prompt, self.use_langchain)
        all_messages = list(self.spec_history) + [user_spec]
        reply = self._call_model(all_messages, **kwargs)
        self.append("user", prompt)
        self.append("assistant", reply)
        return reply

    @abstractmethod
    def _call_model(self, spec_messages: List[SpecMsg], **kwargs) -> str:
        pass
