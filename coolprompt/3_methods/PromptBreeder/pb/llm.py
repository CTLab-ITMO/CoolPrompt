"""OpenAI / langchain_openai adapter that mimics the small subset of the
Cohere ``Client`` interface used by the rest of the codebase.

The original implementation called ``model.generate(prompt)`` and
``model.batch_generate(prompts, temperature=...)`` on a ``cohere.Client``
and consumed the response as ``result[0].text``. To keep all call sites in
:mod:`pb` and :mod:`pb.mutation_operators` untouched, this adapter exposes
the same surface area, but routes all requests to ``ChatOpenAI``.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """Tiny shim that exposes a ``.text`` attribute, mirroring the shape of
    Cohere's ``Generation`` objects so existing ``result[0].text`` access
    in the codebase keeps working."""

    text: str


class OpenAIClient:
    """A drop-in replacement for the small subset of ``cohere.Client`` that
    PromptBreeder uses.

    Args:
        model: OpenAI chat model identifier (default: ``"gpt-4o-mini"``).
        api_key: OpenAI API key. If ``None``, ``ChatOpenAI`` will pick it up
            from the ``OPENAI_API_KEY`` environment variable.
        num_workers: Maximum number of threads used for ``batch_generate``.
        max_retries: Forwarded to ``ChatOpenAI``.
        timeout: Per-request timeout in seconds, forwarded to ``ChatOpenAI``.
        temperature: Default sampling temperature.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        num_workers: int = 8,
        max_retries: int = 5,
        timeout: int = 30,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model
        self.num_workers = max(1, int(num_workers))
        self.default_temperature = temperature
        self._llm = ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            temperature=temperature,
        )

    # ------------------------------------------------------------------ utils
    def _invoke_one(self, prompt: str, temperature: Optional[float]) -> Result:
        """Invoke the chat model on a single prompt and return a ``Result``."""
        try:
            if temperature is not None and temperature != self.default_temperature:
                # ``ChatOpenAI`` is immutable; ``bind`` returns a runnable
                # with overridden default kwargs for this call.
                llm = self._llm.bind(temperature=temperature)
            else:
                llm = self._llm
            response = llm.invoke([HumanMessage(content=prompt)])
            text = getattr(response, "content", "") or ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("OpenAI call failed: %s", exc)
            text = ""
        return Result(text=text)

    # ----------------------------------------------------------- public API
    def generate(self, prompt: str, temperature: Optional[float] = None, **_: object) -> List[Result]:
        """Single-prompt generation. Returns a length-1 list so callers can
        keep using ``result[0].text``."""
        return [self._invoke_one(prompt, temperature)]

    def batch_generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        **_: object,
    ) -> List[List[Result]]:
        """Generate completions for a batch of prompts in parallel.

        Returns a list with the same length as ``prompts``; each element is
        itself a length-1 list of ``Result`` (mirroring the Cohere shape).
        """
        if not prompts:
            return []

        results: List[Optional[List[Result]]] = [None] * len(prompts)
        max_workers = min(self.num_workers, len(prompts))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._invoke_one, prompt, temperature): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = [future.result()]
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("OpenAI batch call failed: %s", exc)
                    results[idx] = [Result(text="")]

        # type: ignore[return-value] -- all entries populated above
        return results  # type: ignore[return-value]
