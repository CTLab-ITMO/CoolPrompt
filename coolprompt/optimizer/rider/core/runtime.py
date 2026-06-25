"""Model routing and LLM runtime helpers for RIDER Genesis Ultra."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from rider.llm.client import LLMClient

from coolprompt.utils.prompt_templates.rider_templates import (
    RIDER_CONTRACT_PROMPT,
    RIDER_SAFE_CLASSIFICATION_PROMPT_EN,
    RIDER_SAFE_CLASSIFICATION_PROMPT_RU,
    RIDER_SAFE_CODE_PROMPT_EN,
    RIDER_SAFE_CODE_PROMPT_RU,
    RIDER_SAFE_GENERIC_PROMPT_EN,
    RIDER_SAFE_GENERIC_PROMPT_RU,
    RIDER_SAFE_TRANSLATION_PROMPT_EN,
    RIDER_SAFE_TRANSLATION_PROMPT_RU,
)
from .schemas import (
    _JudgeScoreSchema,
    _PromptContractSchema,
    _RedTeamAdversarialSchema,
    _SyntheticTestsSchema,
)

logger = logging.getLogger(__name__)
_CONTRACT_PROMPT = RIDER_CONTRACT_PROMPT


class RiderRuntimeMixin:
    """Model routing, schema parsing, and LLM runtime helpers."""

    @staticmethod
    def _split_model_chain(value: Optional[str]) -> List[str]:
        if not value:
            return []
        return [m.strip() for m in value.split(",") if m.strip()]

    @classmethod
    def _env_model_chain(cls, mode: str, role: str) -> List[str]:
        keys = (
            f"RIDER_GENESIS_{mode.upper()}_{role.upper()}_MODELS",
            f"RIDER_GENESIS_{mode.upper()}_{role.upper()}_MODEL",
            f"RIDER_GENESIS_{role.upper()}_MODELS",
            f"RIDER_GENESIS_{role.upper()}_MODEL",
        )
        for key in keys:
            chain = cls._split_model_chain(os.environ.get(key))
            if chain:
                return chain
        return []

    @classmethod
    def _dedupe_models(cls, models: List[str], allow_blocked: bool = False) -> List[str]:
        seen = set()
        result: List[str] = []
        for model in models:
            if not allow_blocked and any(model.startswith(prefix) for prefix in cls._BLOCKED_MODEL_PREFIXES):
                continue
            if model and model not in seen:
                result.append(model)
                seen.add(model)
        return result

    @staticmethod
    def _allow_chinese_model_fallbacks() -> bool:
        return os.environ.get("RIDER_GENESIS_ALLOW_CHINESE_MODELS", "").strip().lower() in {
            "1", "true", "yes", "on",
        }

    def _record_llm_attempt(
        self,
        *,
        role: str,
        model: str,
        status: str,
        reason: str = "",
        structured: bool = False,
    ) -> None:
        self._llm_attempts.append({
            "role": role,
            "model": model,
            "status": status,
            "reason": reason,
            "structured": structured,
        })

    def _build_role_model_chains(self, mode: str) -> Dict[str, List[str]]:
        base = self._MODE_ROLE_MODELS.get(mode, self._MODE_ROLE_MODELS["standard"])
        chains: Dict[str, List[str]] = {}
        for role in self._ROLES:
            chain = self._env_model_chain(mode, role) or list(base.get(role, []))
            if role == "worker" and self._model_override:
                chain = [self._model_override] + chain
            if self.FALLBACK_MODEL not in chain:
                chain.append(self.FALLBACK_MODEL)
            chains[role] = self._dedupe_models(chain)
        return chains

    def _role_models(self, role: str = "worker") -> List[str]:
        if not self._role_model_chains:
            self._role_model_chains = self._build_role_model_chains(self._mode or "standard")
        chain = self._role_model_chains.get(role) or self._role_model_chains.get("worker") or [self.model]
        return chain

    def _role_model(self, role: str = "worker") -> str:
        return self._role_models(role)[0]

    def _strategy_role(self, strategy: str) -> str:
        # Analytical / techniques / VORTEX variants are critic-style mutations:
        # they first look for failure modes, then rewrite. Other strategies are
        # worker/synthesis mutations.
        if strategy in {"analytical", "techniques", "vortex"}:
            return "critic"
        return "worker"

    @classmethod
    def _is_refusal(cls, text: str) -> bool:
        """Detect LLM safety refusal (so we can retry on a permissive model)."""
        if not text or len(text.split()) > 80:
            return False  # full-length outputs are not refusals
        if cls._REFUSAL_RE is None:
            import re as _re
            cls._REFUSAL_RE = _re.compile("|".join(cls._REFUSAL_PATTERNS), _re.IGNORECASE)
        return bool(cls._REFUSAL_RE.search(text))

    @staticmethod
    def _unusable_response_reason(text: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Return a failure reason if an LLM response should be retried/fallbacked."""
        meta = metadata or {}
        error_type = str(meta.get("error_type") or "").lower().strip()
        if error_type:
            return error_type
        finish_reason = str(meta.get("finish_reason") or "").lower().strip()
        if finish_reason in {"content_filter", "safety", "blocked"}:
            return "content_filter"
        if finish_reason in {"length", "max_tokens", "model_length"}:
            return "length"
        if finish_reason and finish_reason not in {"stop", "tool_calls", "end_turn"}:
            return f"finish_reason={finish_reason}"
        if not (text or "").strip():
            return "empty"
        completion_tokens = meta.get("completion_tokens")
        max_tokens = meta.get("max_tokens")
        if isinstance(completion_tokens, int) and isinstance(max_tokens, int):
            if max_tokens > 32 and completion_tokens >= max_tokens - 8:
                return "near_ceiling"
        return ""

    def _main_model_chain(self, role: str, model: Optional[str], allow_fallback: bool) -> List[str]:
        chain = [model] if model else list(self._role_models(role))
        if allow_fallback and self.FALLBACK_MODEL not in chain:
            chain.append(self.FALLBACK_MODEL)
        return self._dedupe_models([m for m in chain if m])

    def _content_filter_fallback_chain(self, chain: List[str]) -> List[str]:
        if not self._allow_chinese_model_fallbacks():
            return []
        return self._dedupe_models(
            [m for m in self._CONTENT_FILTER_FALLBACK_MODELS if m not in chain],
            allow_blocked=True,
        )

    def _generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        role: str = "worker",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        allow_fallback: bool = True,
    ) -> str:
        """LLM call through role-specific model chains with finish_reason routing."""
        max_tokens = max(16, int(max_tokens or 16))
        main_chain = self._main_model_chain(role, model, allow_fallback)
        chains = [main_chain]
        last_resp = ""
        saw_content_filter = False

        chain_index = 0
        while chain_index < len(chains):
            chain = chains[chain_index]
            for effective_model in chain:
                model_max_tokens = max_tokens
                local_attempts = 2 if allow_fallback else 1
                for local_attempt in range(local_attempts):
                    try:
                        gen_kwargs: Dict[str, Any] = {
                            "prompt": prompt,
                            "model": effective_model,
                            "temperature": temperature,
                            "max_tokens": model_max_tokens,
                        }
                        # ULTRA: force max reasoning effort for top-tier
                        # Anthropic models (Opus 4.7) via OpenRouter extra_body.
                        if (
                            self._mode == "ultra"
                            and effective_model in self._MAX_EFFORT_ULTRA_MODELS
                        ):
                            gen_kwargs["extra_body"] = {"reasoning": {"effort": "high"}}
                        resp = self.llm_client.generate(**gen_kwargs)
                        metadata = getattr(self.llm_client, "last_response_metadata", {}) or {}
                        reason = self._unusable_response_reason(resp or "", metadata)
                    except Exception as exc:
                        metadata = getattr(self.llm_client, "last_response_metadata", {}) or {}
                        reason = (
                            str(metadata.get("error_type") or "")
                            or getattr(self.llm_client, "last_error_type", None)
                            or "exception"
                        )
                        logger.debug(f"_generate({effective_model}, role={role}) exc: {exc}")
                        resp = ""

                    last_resp = resp or ""
                    if last_resp and self._is_refusal(last_resp):
                        reason = "refusal"

                    if not allow_fallback:
                        self._record_llm_attempt(
                            role=role, model=effective_model,
                            status="success" if not reason else "failed", reason=reason,
                        )
                        return last_resp

                    if not reason:
                        self._record_llm_attempt(role=role, model=effective_model, status="success")
                        return last_resp

                    self._record_llm_attempt(
                        role=role, model=effective_model, status="failed", reason=reason,
                    )
                    if reason == "content_filter":
                        saw_content_filter = True
                    if reason in {"length", "near_ceiling"} and local_attempt == 0:
                        model_max_tokens = max(model_max_tokens + 256, int(model_max_tokens * 1.35))
                        continue
                    if reason == "refusal":
                        logger.info(
                            f"_generate: refusal on {effective_model} for role={role}; trying next model"
                        )
                    break

            if chain_index == 0 and saw_content_filter:
                emergency = self._content_filter_fallback_chain(main_chain)
                if emergency:
                    chains.append(emergency)
            chain_index += 1

        return last_resp or ""

    @staticmethod
    def _schema_to_dict(obj: BaseModel) -> Dict[str, Any]:
        return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)

    def _instructor_client(self, model: str) -> Optional[Any]:
        """Return a cached Instructor client for OpenRouter, or None if unavailable."""
        try:
            import instructor  # type: ignore
        except Exception:
            return None
        key = f"openrouter/{model}"
        if key in self._instructor_clients:
            return self._instructor_clients[key]
        kwargs: Dict[str, Any] = {
            "api_key": self._api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "async_client": False,
        }
        mode = getattr(getattr(instructor, "Mode", None), "JSON", None)
        if mode is not None:
            kwargs["mode"] = mode
        client = instructor.from_provider(key, **kwargs)
        self._instructor_clients[key] = client
        return client

    def _parse_structured_text(
        self,
        text: str,
        schema: Type[BaseModel],
        *,
        allowed_starts: Tuple[str, ...],
    ) -> Optional[BaseModel]:
        if schema is _JudgeScoreSchema:
            try:
                return schema.model_validate({"score": text})
            except Exception:
                return None
        js = self._extract_json_value(text, allowed_starts=allowed_starts)
        if not js:
            return None
        try:
            value = json.loads(js)
            if schema is _SyntheticTestsSchema and isinstance(value, list):
                value = {"tests": value}
            if schema is _JudgeScoreSchema and isinstance(value, int):
                value = {"score": value}
            return schema.model_validate(value)
        except Exception as exc:
            logger.debug(f"Structured parse failed for {schema.__name__}: {exc}")
            return None

    def _generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        role: str,
        temperature: float,
        max_tokens: int,
        allowed_starts: Tuple[str, ...] = ("{",),
        max_retries: int = 2,
    ) -> Optional[BaseModel]:
        """Instructor-backed structured call with manual JSON/Pydantic fallback."""
        max_tokens = max(16, int(max_tokens or 16))
        main_chain = self._main_model_chain(role, None, True)
        chains = [main_chain]
        saw_content_filter = False

        chain_index = 0
        while chain_index < len(chains):
            chain = chains[chain_index]
            for effective_model in chain:
                client = self._instructor_client(effective_model)
                if client is not None:
                    try:
                        obj = client.create(
                            response_model=schema,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            max_retries=max_retries,
                            extra_body={
                                "provider": {
                                    "ignore": ["Google AI Studio"],
                                    "allow_fallbacks": True,
                                }
                            },
                        )
                        self._record_llm_attempt(
                            role=role, model=effective_model, status="success", structured=True,
                        )
                        return obj
                    except Exception as exc:
                        reason = LLMClient._classify_api_exception(exc)
                        if reason == "content_filter":
                            saw_content_filter = True
                        self._record_llm_attempt(
                            role=role, model=effective_model,
                            status="failed", reason=f"instructor:{reason}",
                            structured=True,
                        )
                        logger.debug(
                            f"Instructor structured call failed on {effective_model} "
                            f"for {schema.__name__}: {exc}"
                        )

                # Manual fallback keeps RIDER usable even when Instructor is absent
                # or a provider/model lacks structured-output/tool support.
                text = self._generate(
                    prompt=prompt,
                    model=effective_model,
                    role=role,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_fallback=False,
                )
                metadata = getattr(self.llm_client, "last_response_metadata", {}) or {}
                reason = self._unusable_response_reason(text or "", metadata)
                if reason == "content_filter":
                    saw_content_filter = True
                    continue
                obj = self._parse_structured_text(text or "", schema, allowed_starts=allowed_starts)
                if obj is not None:
                    self._record_llm_attempt(
                        role=role, model=effective_model, status="success", structured=True,
                    )
                    return obj
                self._record_llm_attempt(
                    role=role, model=effective_model,
                    status="failed", reason="schema_validation",
                    structured=True,
                )

            if chain_index == 0 and saw_content_filter:
                emergency = self._content_filter_fallback_chain(main_chain)
                if emergency:
                    chains.append(emergency)
            chain_index += 1

        return None

    # ══════════════════════════════════════════════════════════════════════
    # Primitive helpers
    # ══════════════════════════════════════════════════════════════════════

    # v4.3 PHASE REACTOR temperatures — 4-phase progression.
    _PHASE_T = {
        'ignition': 1.15,       # broad exploration
        'fusion': 0.85,         # balanced refine
        'crystallization': 0.55,  # polish, low drift
        'validation': 0.3,      # stabilize, near-deterministic
    }
