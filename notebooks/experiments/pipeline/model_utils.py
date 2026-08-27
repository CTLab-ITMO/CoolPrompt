import os
import time
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter


class MultiKeyModel:
    _MAX_INNER_CONCURRENCY = 8

    def __init__(self, models: list):
        self._models = models

    def batch(self, requests: list) -> list:
        n = len(self._models)
        if n == 1:
            return self._batch_with_retry(self._models[0], requests)

        chunks = [[] for _ in range(n)]
        chunk_indices = [[] for _ in range(n)]
        for i, req in enumerate(requests):
            slot = i % n
            chunks[slot].append(req)
            chunk_indices[slot].append(i)

        results = [None] * len(requests)
        for i in range(n):
            if not chunks[i]:
                continue
            responses = self._batch_with_retry(self._models[i], chunks[i])
            for idx, response in zip(chunk_indices[i], responses):
                results[idx] = response
        return results

    def _batch_with_retry(self, model, requests: list, max_attempts: int = 4) -> list:
        c = self._MAX_INNER_CONCURRENCY
        for attempt in range(max_attempts):
            try:
                return model.batch(requests, config={"max_concurrency": c})
            except Exception:
                if attempt < max_attempts - 1:
                    c = max(1, c // 2)
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

    def invoke(self, request):
        return self._models[0].invoke(request)

    def __getattr__(self, name):
        return getattr(self._models[0], name)


def load_proxy_list(proxy_config: dict) -> list:
    proxy_list = proxy_config.get("proxies") or []
    if not proxy_list:
        legacy = proxy_config.get("proxy", {}).get("http")
        if legacy:
            proxy_list = [legacy]
    return proxy_list


def normalize_model_name(provider: str, model_name: str) -> str:
    if provider == "openrouter" and "/" not in model_name:
        return f"openai/{model_name}"
    if provider == "openai" and model_name.startswith("openai/"):
        return model_name.split("/", 1)[1]
    return model_name


def _resolve_api_keys(config: dict) -> list:
    all_keys = config.get("openai_api_keys") or (
        [config["openai_api_key"]] if config.get("openai_api_key") else [os.getenv("OPENAI_API_KEY")]
    )
    active = config.get("active_keys")
    keys = [all_keys[i] for i in active if i < len(all_keys)] if active is not None else all_keys
    return [k for k in keys if k]


def _build_models(api_keys, model_name, base_url, temperature, proxy_list, model_kwargs, requests_per_minute=None):
    models = []
    for idx, key in enumerate(api_keys):
        key_http_client = None
        if proxy_list:
            proxy_url = proxy_list[idx % len(proxy_list)]
            key_http_client = httpx.Client(proxy=proxy_url, timeout=60.0)
        rate_limiter = None
        if requests_per_minute:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_minute / 60.0,
                check_every_n_seconds=0.1,
                max_bucket_size=requests_per_minute,
            )
        models.append(ChatOpenAI(
            model=model_name,
            api_key=key,
            base_url=base_url,
            temperature=temperature,
            rate_limiter=rate_limiter,
            max_retries=3,
            model_kwargs=model_kwargs,
            http_client=key_http_client,
        ))
    return MultiKeyModel(models) if len(models) > 1 else models[0]


def create_model(provider, model_name, requests_per_minute, config, proxy_config, temperature=0.0):
    proxy_list = load_proxy_list(proxy_config)
    model_name = normalize_model_name(provider, model_name)

    if provider == "openrouter":
        api_keys = [config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")]
        base_url = "https://openrouter.ai/api/v1"
        model_kwargs = {"extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": False}}}
    else:
        api_keys = _resolve_api_keys(config)
        base_url = None
        model_kwargs = {}

    api_keys = [k for k in api_keys if k]
    if not api_keys:
        raise ValueError("No API keys found in config")

    if requests_per_minute:
        print(f"{len(api_keys)} keys, {requests_per_minute} RPM (total {len(api_keys) * requests_per_minute})")
    if proxy_list:
        print(f"{len(proxy_list)} proxies")

    return _build_models(api_keys, model_name, base_url, temperature, proxy_list, model_kwargs, requests_per_minute)
