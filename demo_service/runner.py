"""Execution layer for CoolPrompt Interface Demo jobs."""

from __future__ import annotations

import inspect
import time
from contextlib import contextmanager
from difflib import SequenceMatcher
from threading import Event, Lock, Thread
from typing import Any, Callable

from .methods import coerce_method_params
from .schemas import OptimizationRequest, OptimizationResult
from .settings import DemoSettings


TunerFactory = Callable[..., Any]
ProgressCallback = Callable[[str, int, str], None]
_hyper_similarity_lock = Lock()


@contextmanager
def _optimization_heartbeat(progress_callback: ProgressCallback | None):
    """Keep long optimizer calls visibly alive in the web UI."""

    if progress_callback is None:
        yield
        return

    stop = Event()
    messages = [
        "Генерируем кандидаты промпта",
        "Оцениваем кандидаты на примерах",
        "Сравниваем метрики и выбираем лучший вариант",
        "Проверяем формат итогового промпта",
    ]

    def beat() -> None:
        tick = 0
        while not stop.wait(18):
            percent = min(82, 48 + tick * 4)
            progress_callback("optimizing", percent, messages[tick % len(messages)])
            tick += 1

    thread = Thread(target=beat, name="coolprompt-demo-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


class _LightweightPromptSimilarity:
    """Small demo-only substitute for HyPER prompt-to-prompt MMR similarity."""

    def compute(
        self,
        *,
        predictions: list[str],
        references: list[str],
        **_: Any,
    ) -> dict[str, list[float]]:
        scores = [
            SequenceMatcher(None, prediction, reference).ratio()
            for prediction, reference in zip(predictions, references)
        ]
        return {"f1": scores}


@contextmanager
def _demo_hyper_similarity_patch(request: OptimizationRequest, settings: DemoSettings):
    """Avoid a heavyweight local BERTScore model load in the customer demo.

    HyPER itself is still executed through CoolPrompt and still calls the real
    model API. This patch only replaces the internal prompt-similarity helper
    used for MMR in the web demo, so small Render instances do not look hung
    while downloading/loading a transformer model.
    """

    if request.method != "hyper" or not settings.lightweight_hyper_similarity:
        yield
        return

    with _hyper_similarity_lock:
        from coolprompt.optimizer.hyper import hyper as hyper_module

        original = hyper_module._get_bertscore_evaluate
        hyper_module._get_bertscore_evaluate = lambda _metric: _LightweightPromptSimilarity()
        try:
            yield
        finally:
            hyper_module._get_bertscore_evaluate = original


def _clean_prompt(text: Any) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()

    wrappers = (
        ("<ans>", "</ans>"),
        ("[PROMPT_START]", "[PROMPT_END]"),
        ("<result_prompt>", "</result_prompt>"),
        ("<prompt>", "</prompt>"),
        ("<final_prompt>", "</final_prompt>"),
        ("<optimized_prompt>", "</optimized_prompt>"),
    )
    lowered = value.lower()
    for start, end in wrappers:
        if lowered.startswith(start.lower()) and lowered.endswith(end.lower()):
            return value[len(start) : -len(end)].strip()

    embedded_wrappers = (
        ("[PROMPT_START]", "[PROMPT_END]"),
        ("<result_prompt>", "</result_prompt>"),
        ("<final_prompt>", "</final_prompt>"),
        ("<optimized_prompt>", "</optimized_prompt>"),
    )
    for start, end in embedded_wrappers:
        start_idx = lowered.find(start.lower())
        end_idx = lowered.rfind(end.lower())
        if start_idx >= 0 and end_idx > start_idx:
            prefix = value[:start_idx].strip()
            suffix = value[end_idx + len(end) :].strip()
            if len(prefix) <= 120 and len(suffix) <= 120:
                return value[start_idx + len(start) : end_idx].strip()
    return value


def _is_incomplete_prompt_response(raw_text: Any, cleaned_text: str) -> bool:
    """Detect malformed optimizer markup that should not be shown as final."""

    raw = str(raw_text or "").strip()
    lowered_raw = raw.lower()
    lowered_cleaned = cleaned_text.lower()
    paired_markers = (
        ("<ans>", "</ans>"),
        ("<result_prompt>", "</result_prompt>"),
        ("<prompt>", "</prompt>"),
        ("<final_prompt>", "</final_prompt>"),
        ("<optimized_prompt>", "</optimized_prompt>"),
        ("[prompt_start]", "[prompt_end]"),
    )
    for start, end in paired_markers:
        has_start = start in lowered_raw
        has_end = end in lowered_raw
        if has_start != has_end:
            return True

    leftover_markers = (
        "<result_prompt",
        "</result_prompt>",
        "<final_prompt",
        "</final_prompt>",
        "<optimized_prompt",
        "</optimized_prompt>",
        "[prompt_start]",
        "[prompt_end]",
    )
    if any(marker in lowered_cleaned for marker in leftover_markers):
        return True

    tail = cleaned_text.strip().lower().rstrip("`*_")
    if tail.count("```") % 2 == 1:
        return True
    last_line = tail.splitlines()[-1].strip() if tail else ""
    if last_line.startswith(("#", "-", "*")) and len(last_line) <= 4:
        return True
    dangling_tails = (
        " на",
        " в",
        " во",
        " к",
        " ко",
        " с",
        " со",
        " от",
        " для",
        " по",
        " о",
        " об",
        " and",
        " or",
        " to",
        " for",
        " of",
        " in",
        " on",
        " with",
    )
    return bool(tail) and any(tail.endswith(item) for item in dangling_tails)


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
        "3. Return only the requested answer without service markup or extra commentary."
    ).strip()
    init_metric = 0.52 if request.task == "generation" else 0.58
    final_metric = min(0.99, init_metric + (0.12 if request.method == "rider" else 0.08))
    elapsed = time.perf_counter() - started
    return OptimizationResult(
        method=request.method,
        initial_prompt=request.start_prompt,
        final_prompt=_clean_prompt(final_prompt),
        task=request.task,
        metric=request.metric,
        init_metric=init_metric,
        final_metric=final_metric,
        metric_delta=final_metric - init_metric,
        dataset_size=dataset_size,
        validation_size=_validation_count(dataset_size, request.validation_size, request.train_as_test),
        validation_ratio=request.validation_size,
        batch_size=request.batch_size,
        generate_num_samples=request.generate_num_samples,
        elapsed_seconds=elapsed,
        used_mock=True,
        model_name=request.model_name or settings.model_name,
        model_temperature=request.model_temperature,
        model_max_tokens=request.model_max_tokens,
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
            "OPENAI_API_KEY is not configured. Set it in Render or Railway environment variables."
        )

    started = time.perf_counter()
    params = coerce_method_params(request.method, request.method_params)
    if progress_callback:
        progress_callback("loading", 22, "Подготавливаем библиотеку оптимизации")
    tuner = _build_tuner(tuner_factory, request, settings, progress_callback)
    if progress_callback:
        progress_callback("optimizing", 45, "Оптимизатор выполняет поиск промпта")
    with _optimization_heartbeat(progress_callback):
        with _demo_hyper_similarity_patch(request, settings):
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
    initial_prompt = _clean_prompt(tuner.init_prompt or request.start_prompt)
    raw_final_response = final_prompt or tuner.final_prompt or ""
    raw_final_prompt = _clean_prompt(raw_final_response)
    init_metric = None if tuner.init_metric is None else float(tuner.init_metric)
    final_metric = None if tuner.final_metric is None else float(tuner.final_metric)
    quality_guard = None
    surfaced_final_prompt = raw_final_prompt
    if not surfaced_final_prompt and initial_prompt:
        surfaced_final_prompt = initial_prompt
        quality_guard = "Метод не вернул финальный промпт; показан исходный промпт."
    elif _is_incomplete_prompt_response(raw_final_response, raw_final_prompt) and initial_prompt:
        surfaced_final_prompt = initial_prompt
        if final_metric is not None and init_metric is not None:
            final_metric = init_metric
        quality_guard = "Модель вернула незавершённый служебный ответ; показан исходный промпт."
    elif init_metric is not None and final_metric is not None and final_metric < init_metric:
        surfaced_final_prompt = initial_prompt
        final_metric = init_metric
        quality_guard = "Финальный кандидат отклонён защитой качества: на этой валидации лучше остался исходный промпт."
    elif raw_final_prompt.strip() == initial_prompt.strip() and initial_prompt:
        quality_guard = "Исходный промпт остался лучшим вариантом на этой валидации."
    delta = None if init_metric is None or final_metric is None else final_metric - init_metric

    return OptimizationResult(
        method=request.method,
        initial_prompt=initial_prompt,
        final_prompt=surfaced_final_prompt,
        task=request.task,
        metric=request.metric,
        init_metric=init_metric,
        final_metric=final_metric,
        metric_delta=delta,
        dataset_size=dataset_size,
        validation_size=_validation_count(dataset_size, request.validation_size, request.train_as_test),
        validation_ratio=request.validation_size,
        batch_size=request.batch_size,
        generate_num_samples=request.generate_num_samples,
        elapsed_seconds=elapsed,
        used_mock=False,
        model_name=request.model_name or settings.model_name,
        model_temperature=request.model_temperature,
        model_max_tokens=request.model_max_tokens,
        method_params=params,
        quality_guard=quality_guard,
        synthetic_dataset=list(tuner.synthetic_dataset) if tuner.synthetic_dataset is not None else None,
        synthetic_target=list(tuner.synthetic_target) if tuner.synthetic_target is not None else None,
    )
