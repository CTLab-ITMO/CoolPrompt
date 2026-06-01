"""
Централизованный LLM клиент с поддержкой нескольких провайдеров.

Этот модуль предоставляет унифицированный интерфейс для работы с OpenRouter,
OpenAI и DeepSeek. Gemini-модели вида `google/gemini-*` следует вызывать через
OpenRouter; прямой провайдер `gemini` пока не реализован.

Включает:
- Автоматический retry при ошибках
- Rate limiting
- Token counting
- Маппинг моделей (реальное имя → псевдоним для отображения)
"""

import asyncio
import os
import threading
import time
import logging
from typing import List, Dict, Optional, Any, Iterable
from dataclasses import dataclass

# LLM клиенты
from openai import OpenAI
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Конфигурация LLM модели"""
    name: str  # Реальное имя для API
    display_name: str  # Псевдоним для отображения
    max_tokens: int = 4096
    timeout: int = 60


# Маппинг моделей (реальное имя → псевдоним)
MODEL_DISPLAY_NAMES = {
    # OpenRouter
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic/claude-opus-4.7": "claude-opus-4.7",
    "google/gemini-3-flash-preview": "gemini-3-flash",
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro",
    "openai/gpt-5.4-mini": "gpt-5.4-mini",
    "openai/gpt-5.5": "gpt-5.5",

    # OpenAI
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.5": "gpt-5.5",

}


# Цены моделей OpenRouter ($/token, не $/M tokens).
# Источник: https://openrouter.ai/docs/models
# Формат: model_id → (prompt_price_per_token, completion_price_per_token)
_MODEL_PRICES = {
    'google/gemini-3-flash-preview': (0.50 / 1_000_000, 3.00 / 1_000_000),
    'google/gemini-3.1-pro-preview': (2.00 / 1_000_000, 12.00 / 1_000_000),
    'anthropic/claude-sonnet-4.6': (3.00 / 1_000_000, 15.00 / 1_000_000),
    'anthropic/claude-opus-4.7': (15.00 / 1_000_000, 75.00 / 1_000_000),
    'openai/gpt-5.4-mini': (0.25 / 1_000_000, 2.00 / 1_000_000),
    'openai/gpt-5.5': (2.00 / 1_000_000, 10.00 / 1_000_000),
    'default': (1.00 / 1_000_000, 5.00 / 1_000_000),  # fallback — conservative
}


def get_model_display_name(real_name: str) -> str:
    """
    Получить псевдоним модели для отображения.

    Args:
        real_name: Реальное имя модели для API

    Returns:
        Псевдоним модели или само имя, если маппинг не найден
    """
    return MODEL_DISPLAY_NAMES.get(real_name, real_name)


class LLMClient:
    """
    Централизованный клиент для работы с LLM API.

    Поддерживает множественные провайдеры с автоматическим retry,
    rate limiting и error handling.

    Args:
        provider: Название провайдера ("openrouter", "openai", "deepseek")
        api_key: API ключ (опционально, берется из .env если не указан)
        max_retries: Максимальное количество повторных попыток (default: 10)
        retry_delay: Задержка между попытками в секундах (default: 1.0)

    Example:
        >>> from dotenv import load_dotenv
        >>> load_dotenv()
        >>> client = LLMClient(provider="openrouter")
        >>> response = client.generate(
        ...     prompt="Solve this: 2+2=?",
        ...     model="deepseek/deepseek-v3.2-exp",
        ...     temperature=0.7
        ... )
        >>> print(response)
    """

    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        max_retries: int = 10,
        retry_delay: float = 1.0,
        use_neural_firewall: Optional[bool] = None,
        use_uniai: Optional[bool] = None,
    ):
        if provider == "gemini":
            raise NotImplementedError(
                "Gemini provider not yet implemented. "
                "Use provider='openrouter' with google/gemini-* models via OpenRouter."
            )

        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Получить API ключ
        if api_key is None:
            api_key = self._get_api_key_from_env(provider)
        self.api_key = api_key

        # Инициализировать клиент провайдера
        self.client = self._initialize_client(provider, api_key)
        if use_neural_firewall is None and use_uniai is not None:
            use_neural_firewall = use_uniai
        if use_neural_firewall is None:
            use_neural_firewall = False
        self.use_neural_firewall = bool(use_neural_firewall)
        self._neural_firewall_unavailable_warned = False

        # Инициализировать tokenizer для подсчета токенов
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken: {e}")
            self.tokenizer = None

        # Трекинг API-вызовов, токенов, стоимости.
        # Для анализа сходимости и таблицы затрат (запрос руководителя).
        # OpenRouter возвращает реальную стоимость в response (total_cost).
        self.total_api_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0  # USD, из OpenRouter API напрямую
        # Снимки на каждое поколение (generation → snapshot)
        self._generation_snapshots: Dict[int, Dict] = {}
        # Last-call diagnostics for higher-level routers. RiderGenesis uses this
        # to switch models on content_filter/length/empty outputs instead of
        # treating every non-exception response as usable text.
        self.last_response_metadata: Dict[str, Any] = {}
        self.last_error_type: Optional[str] = None

    def reset_usage(self) -> None:
        """Сброс счётчиков перед новым экспериментом."""
        self.total_api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._generation_snapshots = {}

    def _get_api_key_from_env(self, provider: str) -> str:
        """
        Получить API ключ из переменных окружения.

        Args:
            provider: Название провайдера

        Returns:
            API ключ

        Raises:
            ValueError: Если ключ не найден
        """
        env_var_mapping = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }

        env_var = env_var_mapping.get(provider)
        if env_var is None:
            supported = ", ".join(sorted(env_var_mapping))
            raise ValueError(
                f"Unknown provider: {provider}. Supported providers: {supported}"
            )

        api_key = os.getenv(env_var)
        if api_key is None:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Please set {env_var} environment variable."
            )

        return api_key

    def _initialize_client(self, provider: str, api_key: str) -> Any:
        """
        Инициализировать клиент для указанного провайдера.

        Args:
            provider: Название провайдера
            api_key: API ключ

        Returns:
            Инициализированный клиент
        """
        if provider == "openrouter":
            return OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        elif provider == "openai":
            return OpenAI(api_key=api_key)
        elif provider == "deepseek":
            return OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
        elif provider == "gemini":
            raise NotImplementedError(
                "Gemini provider not yet implemented. "
                "Use provider='openrouter' with google/gemini-* models via OpenRouter."
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def _classify_api_exception(exc: Exception) -> str:
        """Classify provider errors so callers can decide retry vs fallback."""
        text = str(exc).lower()
        if any(x in text for x in ("content_filter", "safety", "blocked")):
            return "content_filter"
        if any(x in text for x in ("401", "403", "authentication", "unauthorized", "forbidden")):
            return "auth"
        if any(x in text for x in ("404", "not found", "model_not_found", "no endpoints found")):
            return "not_found"
        if any(x in text for x in ("context_length", "maximum context", "context window")):
            return "context_too_large"
        if any(x in text for x in ("429", "rate limit", "rate_limit", "too many requests")):
            return "rate_limit"
        if any(x in text for x in ("timeout", "readtimeout", "connection", "502", "503", "504")):
            return "retryable"
        return "unknown"

    @staticmethod
    def _run_coro_sync(coro):
        """Run an async NeuralFirewall call from sync RIDER code, including async hosts."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        box: Dict[str, Any] = {}

        def _runner() -> None:
            try:
                box["result"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - re-raised below
                box["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _openrouter_extra_body(self, extra_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Default OpenRouter routing used by both direct OpenAI SDK and NeuralFirewall."""
        extra = dict(extra_body or {})
        if self.provider == "openrouter" and "provider" not in extra:
            extra["provider"] = {
                "ignore": ["Google AI Studio"],
                "allow_fallbacks": True,
            }
        return extra

    @staticmethod
    def _neural_firewall_provider_enum(provider: str):
        from neural_firewall import AiProviderEnum

        if provider == "openrouter":
            return AiProviderEnum.OPENROUTER
        if provider == "openai":
            return AiProviderEnum.OPENAI
        raise ValueError(f"NeuralFirewall provider is not configured for {provider!r}")

    @staticmethod
    def _neural_firewall_messages(messages: Iterable[Dict[str, str]]):
        from neural_firewall.core.capabilities.chat.chat_request import ChatMessage
        from neural_firewall.core.capabilities.chat.chat_request import ChatRoleEnum

        converted = []
        for msg in messages:
            role = str(msg.get("role", "user")).lower()
            if role == "system":
                converted.append(ChatMessage(role=ChatRoleEnum.SYSTEM, content=msg.get("content", "")))
            elif role == "user":
                converted.append(ChatMessage(role=ChatRoleEnum.USER, content=msg.get("content", "")))
            else:
                raise ValueError(f"NeuralFirewall sync bridge supports user/system messages, got {role!r}")
        return converted

    def _record_neural_firewall_metadata(self, metadata_stack: Iterable[Any]) -> None:
        """Merge NeuralFirewall metadata into RIDER counters and last-call diagnostics."""
        stack = list(metadata_stack or [])
        if not stack:
            return
        for metadata in stack:
            self.total_api_calls += 1
            usage = getattr(metadata, "usage", None)
            prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
            completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
            total_tokens = (
                int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
                if usage is not None else 0
            )
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            cost = getattr(metadata, "cost", None)
            if cost is not None:
                self.total_cost += float(cost)
            else:
                prices = _MODEL_PRICES.get(getattr(metadata, "model", ""), _MODEL_PRICES.get("default"))
                self.total_cost += prompt_tokens * prices[0] + completion_tokens * prices[1]

        last = stack[-1]
        last_error = getattr(last, "error", None)
        self.last_error_type = getattr(last_error, "err_code", None)
        usage = getattr(last, "usage", None)
        self.last_response_metadata = {
            "model": getattr(last, "model", None),
            "attempt": len(stack),
            "finish_reason": getattr(last, "finish_reason", None),
            "completion_tokens": getattr(usage, "output_tokens", None) if usage is not None else None,
            "max_tokens": getattr(last, "max_tokens", None),
            "empty": False,
            "error_type": self.last_error_type,
            "neural_firewall": True,
        }

    def _generate_with_neural_firewall(
        self,
        *,
        messages: List[Dict[str, str]],
        models: List[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        extra_body: Optional[Dict[str, Any]],
        extra_kwargs: Dict[str, Any],
    ) -> str:
        """Validated NeuralFirewall path for RIDER and baseline LLM calls."""
        from neural_firewall import AiClientSwitchingError
        from neural_firewall import NeuralFirewallClient
        from neural_firewall import NeuralFirewallSettings

        provider = self._neural_firewall_provider_enum(self.provider)
        settings = NeuralFirewallSettings(api_key=self.api_key, models=models)
        client = NeuralFirewallClient(provider=provider, settings=settings)
        request_extra = dict(extra_kwargs)
        routed_extra_body = self._openrouter_extra_body(extra_body)
        if routed_extra_body:
            request_extra["extra_body"] = routed_extra_body

        try:
            response = self._run_coro_sync(
                client.chat_with_model_switching(
                    messages=self._neural_firewall_messages(messages),
                    response_format=str,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    max_retries_per_model=max(1, self.max_retries),
                    initial_delay_seconds=self.retry_delay,
                    delta_near_ceiling=64,
                    **request_extra,
                )
            )
        except AiClientSwitchingError as exc:
            self._record_neural_firewall_metadata(getattr(exc, "metadata_stack", ()))
            err = self.last_error_type or self._classify_api_exception(exc)
            self.last_error_type = err
            self.last_response_metadata.update({
                "error_type": err,
                "error": str(exc)[:500],
                "neural_firewall": True,
            })
            raise
        finally:
            try:
                self._run_coro_sync(client.aclose())
            except Exception as close_exc:
                logger.debug("NeuralFirewall client close failed: %s", close_exc)

        self._record_neural_firewall_metadata(response.all_metadata)
        text = response.result or ""
        self.last_response_metadata["empty"] = not bool(text.strip())
        return text

    def generate(
        self,
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs
    ) -> str:
        """
        Генерация текста от LLM с автоматическим retry.

        Args:
            prompt: Текст промпта (используется если messages=None)
            messages: Список сообщений в формате OpenAI chat API
            model: Имя модели
            temperature: Температура генерации (0.0 - 2.0)
            max_tokens: Максимальное количество токенов в ответе
            top_p: Nucleus sampling параметр
            **kwargs: Дополнительные параметры для API

        Returns:
            Сгенерированный текст

        Raises:
            Exception: После исчерпания всех retry попыток
        """
        # Подготовить messages
        if messages is None:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
            messages = [{"role": "user", "content": prompt}]

        fallback_models = kwargs.pop("fallback_models", None) or []
        if isinstance(fallback_models, str):
            fallback_models = [fallback_models]
        model_chain = [model] + [m for m in fallback_models if m and m != model]
        extra_body = kwargs.pop('extra_body', None)

        if self.use_neural_firewall and self.provider in {"openrouter", "openai"}:
            try:
                return self._generate_with_neural_firewall(
                    messages=messages,
                    models=model_chain,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    extra_body=extra_body,
                    extra_kwargs=dict(kwargs),
                )
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                if not self._neural_firewall_unavailable_warned:
                    logger.warning(f"NeuralFirewall path unavailable, falling back to direct SDK: {exc}")
                    self._neural_firewall_unavailable_warned = True

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                self.last_error_type = None
                self.last_response_metadata = {
                    "model": model,
                    "attempt": attempt + 1,
                    "finish_reason": None,
                    "error_type": None,
                }
                # OpenRouter provider routing: avoid Google AI Studio
                # (blocks requests from Russia) — route through Azure/Together
                extra = self._openrouter_extra_body(extra_body)

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    extra_body=extra,
                    **kwargs
                )

                # Трекинг токенов и стоимости из ответа OpenRouter API.
                # OpenRouter возвращает cost напрямую в response.usage.model_extra['cost'].
                self.total_api_calls += 1
                if hasattr(response, 'usage') and response.usage is not None:
                    pt = getattr(response.usage, 'prompt_tokens', 0) or 0
                    ct = getattr(response.usage, 'completion_tokens', 0) or 0
                    self.total_prompt_tokens += pt
                    self.total_completion_tokens += ct
                    self.total_tokens += getattr(response.usage, 'total_tokens', 0) or 0
                    # Стоимость напрямую из OpenRouter API (как в censor-v3)
                    api_cost = (getattr(response.usage, 'model_extra', None) or {}).get('cost')
                    if api_cost is not None:
                        self.total_cost += float(api_cost)
                    else:
                        # Fallback: расчёт по таблице цен если API не вернул cost
                        prices = _MODEL_PRICES.get(model, _MODEL_PRICES.get('default'))
                        self.total_cost += pt * prices[0] + ct * prices[1]

                # Извлечь текст ответа
                choice = response.choices[0] if response.choices else None
                finish_reason = getattr(choice, "finish_reason", None) if choice else None
                generated_text = (choice.message.content if choice else "") or ""
                usage = getattr(response, "usage", None)
                completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
                self.last_response_metadata = {
                    "model": model,
                    "attempt": attempt + 1,
                    "finish_reason": finish_reason,
                    "completion_tokens": completion_tokens,
                    "max_tokens": max_tokens,
                    "empty": not bool(generated_text.strip()),
                    "error_type": None,
                }
                return generated_text

            except Exception as e:
                error_type = self._classify_api_exception(e)
                self.last_error_type = error_type
                self.last_response_metadata = {
                    "model": model,
                    "attempt": attempt + 1,
                    "finish_reason": None,
                    "error_type": error_type,
                    "error": str(e)[:500],
                }
                logger.warning(
                    f"LLM API call failed [{error_type}] "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if error_type in {"auth", "not_found", "context_too_large", "content_filter"}:
                    logger.info(f"Non-retryable for this model ({error_type}); falling back upstream.")
                    raise

                if attempt < self.max_retries - 1:
                    # Экспоненциальная задержка: 1s, 2s, 4s, ...
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    # Исчерпаны все попытки
                    logger.error(f"All retry attempts exhausted. Last error: {e}")
                    raise

    def snapshot_generation(self, generation: int) -> Dict[str, int]:
        """
        Сохранить снимок счётчиков для текущего поколения.
        Вызывается из RIDER.run() в конце каждого поколения.

        Returns:
            Словарь с кумулятивной статистикой на момент снимка.
        """
        snapshot = {
            'api_calls': self.total_api_calls,
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'cost_usd': round(self.total_cost, 6),
        }
        self._generation_snapshots[generation] = snapshot
        return snapshot

    def get_generation_usage(self, generation: int) -> Dict[str, int]:
        """
        Получить инкрементальную статистику за одно поколение.
        Вычисляется как разница между текущим и предыдущим снимком.
        """
        current = self._generation_snapshots.get(generation, {})
        prev_gen = generation - 1
        prev = self._generation_snapshots.get(prev_gen, {
            'api_calls': 0, 'prompt_tokens': 0,
            'completion_tokens': 0, 'total_tokens': 0, 'cost_usd': 0.0
        })
        return {
            'api_calls': current.get('api_calls', 0) - prev.get('api_calls', 0),
            'prompt_tokens': current.get('prompt_tokens', 0) - prev.get('prompt_tokens', 0),
            'completion_tokens': current.get('completion_tokens', 0) - prev.get('completion_tokens', 0),
            'total_tokens': current.get('total_tokens', 0) - prev.get('total_tokens', 0),
            'cost_usd': round(current.get('cost_usd', 0.0) - prev.get('cost_usd', 0.0), 6),
        }

    def get_usage_stats(self) -> Dict[str, any]:
        """
        Полная статистика использования API.
        Для сохранения в summary.json и generation_summaries.json.
        """
        return {
            'total_api_calls': self.total_api_calls,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'total_cost_usd': round(self.total_cost, 6),
            'per_generation': dict(self._generation_snapshots),
        }

    def count_tokens(self, text: str) -> int:
        """
        Подсчет токенов в тексте.

        Args:
            text: Текст для подсчета

        Returns:
            Количество токенов (приблизительное)
        """
        if self.tokenizer is None:
            # Фоллбэк: грубая оценка (1 токен ≈ 4 символа)
            return len(text) // 4

        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using fallback.")
            return len(text) // 4


def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_output_tokens: int = 2048,
    top_p: float = 1.0,
    client: Optional[LLMClient] = None
) -> str:
    """
    Удобная функция-обертка для вызова LLM (совместимость с notebook).

    Args:
        messages: Список сообщений в формате [{"role": "user", "content": "..."}]
        model: Имя модели
        temperature: Температура генерации
        max_output_tokens: Максимальное количество токенов в ответе
        top_p: Nucleus sampling параметр
        client: LLM клиент (опционально, создается если не указан)

    Returns:
        Сгенерированный текст
    """
    if client is None:
        # Создать временный клиент
        client = LLMClient(provider="openrouter")

    return client.generate(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        top_p=top_p
    )
