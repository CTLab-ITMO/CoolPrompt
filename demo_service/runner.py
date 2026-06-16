"""Execution layer for CoolPrompt Interface Demo jobs."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable

from .methods import coerce_method_params
from .schemas import CompareRequest, OptimizationRequest, OptimizationResult
from .settings import DemoSettings


TunerFactory = Callable[..., Any]
ProgressCallback = Callable[[str, int, str], None]


def default_tuner_factory(
    request: OptimizationRequest,
    settings: DemoSettings,
    progress_callback: ProgressCallback | None = None,
) -> PromptTuner:
    """Build a PromptTuner with an env-configured LangChain OpenAI model."""

    # Keep heavy CoolPrompt/metric/torch imports out of FastAPI startup.
    # Render's small instances must open the HTTP port before any optimizer job
    # pays the full ML import cost.
    if progress_callback:
        progress_callback("loading", 24, "Загружаем клиент модели")
    from langchain_openai import ChatOpenAI

    if progress_callback:
        progress_callback("loading", 30, "Загружаем библиотеку CoolPrompt")
    from coolprompt.assistant import PromptTuner

    model_name = request.model_name or settings.model_name
    model_kwargs = {
        "model": model_name,
        "temperature": request.model_temperature,
        "max_tokens": request.model_max_tokens,
    }
    if settings.openai_api_key:
        model_kwargs["api_key"] = settings.openai_api_key
    if settings.openai_base_url:
        model_kwargs["base_url"] = settings.openai_base_url

    if progress_callback:
        progress_callback("model", 36, "Подключаем модель")
    model = ChatOpenAI(**model_kwargs)
    return PromptTuner(target_model=model, system_model=model)


def _build_tuner(
    tuner_factory: TunerFactory,
    request: OptimizationRequest,
    settings: DemoSettings,
    progress_callback: ProgressCallback | None,
) -> Any:
    """Call custom test factories without requiring the new progress argument."""

    try:
        signature = inspect.signature(tuner_factory)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "progress_callback" in signature.parameters:
        return tuner_factory(request, settings, progress_callback=progress_callback)
    return tuner_factory(request, settings)


def _effective_mock(request: OptimizationRequest, settings: DemoSettings) -> bool:
    return settings.force_mock or (settings.allow_mock and request.mock)


def _dataset_size(request: OptimizationRequest, tuner: PromptTuner | None = None) -> int:
    if request.dataset is not None:
        return len(request.dataset)
    if tuner is not None and tuner.synthetic_dataset is not None:
        return len(tuner.synthetic_dataset)
    return 0


def _validation_count(total: int, validation_size: float, train_as_test: bool) -> int:
    if total <= 0:
        return 0
    if train_as_test:
        return total
    return max(1, int(round(total * validation_size)))


def run_mock_optimization(request: OptimizationRequest, settings: DemoSettings) -> OptimizationResult:
    """Deterministic local demo path used for tests and no-key demos."""

    started = time.perf_counter()
    time.sleep(0.6)
    params = coerce_method_params(request.method, request.method_params)
    dataset_size = len(request.dataset or [])
    if dataset_size == 0:
        dataset_size = request.generate_num_samples

    method_titles = {
        "hyper_light": "HyPER Light",
        "hyper": "HyPER",
        "rider": "RIDER",
        "regps": "RE-GPS",
        "compress": "PromptCompressor",
        "reflective": "ReflectivePrompt",
        "distill": "DistillPrompt",
    }
    title = method_titles.get(request.method, request.method)
    final_prompt = (
        f"{request.start_prompt.strip()}\n\n"
        f"Optimized by {title}:\n"
        "1. State the task and expected output format explicitly.\n"
        "2. Use the provided examples as calibration signals.\n"
        "3. Return only the final answer inside <ans>...</ans> tags."
    )
    init_metric = 0.52 if request.task == "generation" else 0.58
    final_metric = min(0.99, init_metric + (0.12 if request.method == "rider" else 0.08))
    elapsed = time.perf_counter() - started
    return OptimizationResult(
        method=request.method,
        initial_prompt=request.start_prompt,
        final_prompt=final_prompt,
        init_metric=init_metric,
        final_metric=final_metric,
        metric_delta=final_metric - init_metric,
        dataset_size=dataset_size,
        validation_size=_validation_count(dataset_size, request.validation_size, request.train_as_test),
        elapsed_seconds=elapsed,
        used_mock=True,
        model_name=request.model_name or settings.model_name,
        method_params=params,
        synthetic_dataset=None if request.dataset else ["mock input A", "mock input B"],
        synthetic_target=None if request.target else ["mock target A", "mock target B"],
    )


def run_single_optimization(
    request: OptimizationRequest,
    settings: DemoSettings,
    *,
    tuner_factory: TunerFactory = default_tuner_factory,
    progress_callback: ProgressCallback | None = None,
) -> OptimizationResult:
    """Run a single CoolPrompt optimization and return serializable results."""

    if progress_callback:
        progress_callback("preparing", 15, "Проверяем параметры и данные")
    if _effective_mock(request, settings):
        if progress_callback:
            progress_callback("optimizing", 70, "Выполняем тестовую оптимизацию")
        return run_mock_optimization(request, settings)
    if not settings.has_openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in Railway Variables or enable "
            "COOLPROMPT_DEMO_MOCK=1 for a no-cost mock demo."
        )

    started = time.perf_counter()
    params = coerce_method_params(request.method, request.method_params)
    if progress_callback:
        progress_callback("loading", 22, "Подготавливаем библиотеку оптимизации")
    tuner = _build_tuner(tuner_factory, request, settings, progress_callback)
    if progress_callback:
        progress_callback("optimizing", 45, "Оптимизатор выполняет поиск промпта")
    final_prompt = tuner.run(
        start_prompt=request.start_prompt,
        task=request.task,
        dataset=request.dataset,
        target=request.target,
        method=request.method,
        metric=request.metric,
        problem_description=request.problem_description,
        validation_size=request.validation_size,
        train_as_test=request.train_as_test,
        generate_num_samples=request.generate_num_samples,
        batch_size=request.batch_size,
        verbose=request.verbose,
        return_final_prompt=True,
        **params,
    )
    if progress_callback:
        progress_callback("collecting", 88, "Собираем метрики и итоговый промпт")
    elapsed = time.perf_counter() - started

    dataset_size = _dataset_size(request, tuner)
    init_metric = None if tuner.init_metric is None else float(tuner.init_metric)
    final_metric = None if tuner.final_metric is None else float(tuner.final_metric)
    delta = None if init_metric is None or final_metric is None else final_metric - init_metric

    return OptimizationResult(
        method=request.method,
        initial_prompt=tuner.init_prompt or request.start_prompt,
        final_prompt=final_prompt or tuner.final_prompt or "",
        init_metric=init_metric,
        final_metric=final_metric,
        metric_delta=delta,
        dataset_size=dataset_size,
        validation_size=_validation_count(dataset_size, request.validation_size, request.train_as_test),
        elapsed_seconds=elapsed,
        used_mock=False,
        model_name=request.model_name or settings.model_name,
        method_params=params,
        synthetic_dataset=list(tuner.synthetic_dataset) if tuner.synthetic_dataset is not None else None,
        synthetic_target=list(tuner.synthetic_target) if tuner.synthetic_target is not None else None,
    )


def run_comparison(
    request: CompareRequest,
    settings: DemoSettings,
    *,
    tuner_factory: TunerFactory = default_tuner_factory,
    progress_callback: ProgressCallback | None = None,
) -> list[OptimizationResult]:
    """Run selected methods on the same input."""

    methods = list(dict.fromkeys(request.methods))
    if len(methods) > settings.max_compare_methods:
        raise ValueError(
            f"Too many methods selected: {len(methods)}. "
            f"Maximum is {settings.max_compare_methods}."
        )

    results: list[OptimizationResult] = []
    total = len(methods)
    for index, method_id in enumerate(methods, start=1):
        if progress_callback:
            base_percent = 10 + int((index - 1) / total * 75)
            progress_callback(
                "optimizing",
                base_percent,
                f"Сравниваем методы: {index}/{total} — {method_id}",
            )
        method_request = request.base.model_copy(deep=True)
        method_request.method = method_id
        method_request.method_params = request.method_params_by_method.get(method_id, {})
        results.append(
            run_single_optimization(
                method_request,
                settings,
                tuner_factory=tuner_factory,
                progress_callback=progress_callback,
            )
        )
    if progress_callback:
        progress_callback("collecting", 90, "Собираем результаты сравнения")
    return results
