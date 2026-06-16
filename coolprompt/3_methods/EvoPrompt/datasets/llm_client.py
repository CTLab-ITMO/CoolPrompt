"""LLM client wrapper around ``langchain_openai.ChatOpenAI``.

This module replaces the original ``openai``-based implementation. It exposes
the same public functions used by the rest of the EvoPrompt code base
(``llm_init``, ``llm_query``, ``paraphrase``, ``turbo_query``, ``davinci_query``)
but routes everything through a single ``ChatOpenAI`` instance configured for
``gpt-5-nano`` (or any chat model selected via CLI) with temperature ``0.7``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Union

from tqdm import tqdm

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "langchain_openai is required. Install it with "
        "`pip install langchain-openai`."
    ) from exc

from utils import batchify

# A single shared client. ``llm_init`` populates this.
_CLIENT: ChatOpenAI | None = None
_CONFIG: Dict[str, Any] = {}

DEFAULT_SYSTEM_PROMPT = (
    "Follow the given examples and answer the question."
)


def _build_client(model: str, temperature: float, api_key: str | None,
                  base_url: str | None = None) -> ChatOpenAI:
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key found. Pass --openai_api_key or set the "
            "OPENAI_API_KEY environment variable."
        )
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def llm_init(auth_file: str | None = None,
             llm_type: str = "turbo",
             setting: str = "default",
             model: str = "gpt-5-nano",
             temperature: float = 0.7,
             api_key: str | None = None,
             base_url: str | None = None) -> Dict[str, Any]:
    """Initialise the shared ``ChatOpenAI`` client.

    The ``auth_file`` / ``llm_type`` / ``setting`` arguments are kept for
    backward compatibility but are now ignored; configuration is taken from
    the explicit keyword arguments and the ``OPENAI_API_KEY`` environment
    variable.
    """
    global _CLIENT, _CONFIG
    _CLIENT = _build_client(model=model, temperature=temperature,
                            api_key=api_key, base_url=base_url)
    _CONFIG = {
        "model": model,
        "temperature": temperature,
    }
    return dict(_CONFIG)


def get_client() -> ChatOpenAI:
    if _CLIENT is None:
        # Lazy initialisation from environment variable.
        llm_init()
    assert _CLIENT is not None
    return _CLIENT


# ---------------------------------------------------------------------------
# Low level chat call with retry
# ---------------------------------------------------------------------------

def _chat(prompt: str, system: str = DEFAULT_SYSTEM_PROMPT,
          retries: int = 6, **_ignored) -> str:
    """Call the chat model with simple exponential back-off."""
    client = get_client()
    delay = 3
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ]
            response = client.invoke(messages)
            content = getattr(response, "content", None)
            if isinstance(content, list):
                # Some langchain versions return a list of parts.
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return (content or "").strip()
        except Exception as e:  # noqa: BLE001 - propagate after retries
            last_err = e
            print(f"[llm_client] retry {attempt + 1}/{retries}: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Public wrappers preserving the original API
# ---------------------------------------------------------------------------

def turbo_query(prompt: str, temperature: float | None = None, **kwargs) -> str:
    """Single-shot chat completion (drop-in replacement for the old function)."""
    return _chat(prompt, **kwargs)


def davinci_query(data: Union[str, List[str]], client=None, **kwargs):
    """Batched completion used by the legacy BBH path. Returns a list when given
    a list, and a string when given a single prompt (mirrors prior behaviour)."""
    if isinstance(data, list):
        return [_chat(item, **kwargs) for item in data]
    return _chat(data, **kwargs)


def llm_query(data: Union[str, List[str]], client=None, type: str = "turbo",
              task: bool = False, **config) -> Union[str, List[str]]:
    """Backward compatible query helper.

    Parameters
    ----------
    data : str | list[str]
        Prompt(s) to send to the chat model.
    client : ignored
        Present only for signature compatibility with the old code.
    type : str
        Ignored; kept for backward compatibility.
    task : bool
        If True, returned strings are truncated at the first ``\\n\\n``.
    """
    # Filter out keys that ChatOpenAI.invoke does not understand.
    config = {k: v for k, v in config.items()
              if k not in {"model", "temperature"}}

    if isinstance(data, list):
        results: List[str] = []
        for batch in tqdm(batchify(data, 20), desc="llm_query"):
            for item in batch:
                resp = _chat(item, **config)
                if task:
                    resp = resp.split("\n\n")[0]
                results.append(resp.strip())
        return results

    resp = _chat(data, **config)
    if task:
        resp = resp.split("\n\n")[0]
    return resp.strip()


def paraphrase(sentence: Union[str, List[str]], client=None,
               type: str = "turbo", **kwargs) -> Union[str, List[str]]:
    """Generate a semantically-equivalent paraphrase of one or many sentences."""
    template = (
        "Generate a variation of the following instruction while keeping the "
        "semantic meaning.\nInput:{s}\nOutput:"
    )
    if isinstance(sentence, list):
        prompts = [template.format(s=s) for s in sentence]
    else:
        prompts = template.format(s=sentence)
    return llm_query(prompts, client=client, type=type, task=False, **kwargs)


if __name__ == "__main__":
    llm_init(model="gpt-5-nano", temperature=1.0)
    print(paraphrase("Let's think step by step."))
