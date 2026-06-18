"""Real API end-to-end matrix for the CoolPrompt demo service.

This script intentionally talks to the running FastAPI service instead of
calling runner functions directly. It verifies the same job lifecycle used by
the browser: create job, poll job, inspect serializable result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Example:
    name: str
    task: str
    prompt: str
    description: str
    metric: str
    rows: list[tuple[str, str]]


EXAMPLES: dict[str, Example] = {
    "support_reply": Example(
        name="support_reply",
        task="generation",
        prompt="Напиши короткий профессиональный ответ клиенту.",
        description=(
            "Нужно генерировать короткие ответы поддержки: признать проблему, "
            "дать конкретный следующий шаг, сохранить спокойный профессиональный "
            "тон и не обещать того, чего оператор не может гарантировать. "
            "Итоговый ответ должен быть готовым сообщением без служебных пометок "
            "и лишних пояснений."
        ),
        metric="llm_as_judge",
        rows=[
            (
                "Клиент пишет, что оплатил заказ, но статус до сих пор не изменился.",
                "Здравствуйте! Проверим оплату и статус заказа. Пришлите, пожалуйста, номер заказа или чек, чтобы мы быстрее нашли платёж.",
            ),
            (
                "Клиент просит перенести доставку на другой день.",
                "Здравствуйте! Да, дату доставки можно изменить. Напишите удобный день и интервал, а мы проверим доступные варианты.",
            ),
            (
                "Клиент сообщает, что приложение не открывается после обновления.",
                "Здравствуйте! Попробуйте перезапустить приложение и очистить кэш. Если ошибка повторится, пришлите модель устройства и скриншот.",
            ),
            (
                "Клиент получил товар с повреждённой упаковкой и просит заменить заказ.",
                "Здравствуйте! Нам жаль, что заказ пришёл повреждённым. Пришлите фото упаковки и товара, а мы проверим варианты замены или возврата.",
            ),
            (
                "Клиент не может войти в аккаунт: код подтверждения не приходит.",
                "Здравствуйте! Проверьте, пожалуйста, номер телефона и папку со спамом. Если код не придёт повторно, напишите нам, и мы проверим вход вручную.",
            ),
            (
                "Клиент спрашивает, почему промокод не уменьшил итоговую сумму заказа.",
                "Здравствуйте! Проверим условия промокода и состав заказа. Пришлите код акции и номер заказа, чтобы мы быстро нашли причину.",
            ),
            (
                "Клиент просит отменить заказ, который уже передан в доставку.",
                "Здравствуйте! Проверим, можно ли ещё отменить отправление. Пришлите номер заказа, и мы подскажем доступные варианты.",
            ),
        ],
    ),
    "summary": Example(
        name="summary",
        task="generation",
        prompt="Сократи текст до короткого делового резюме.",
        description=(
            "Нужно делать короткое деловое резюме текста в 1-2 предложениях: "
            "сохранять главные факты, не добавлять новых деталей, не терять "
            "числа, сроки и ограничения. Итог должен быть готовым резюме без "
            "служебных пометок и лишних пояснений."
        ),
        metric="llm_as_judge",
        rows=[
            (
                "Клиент сообщил, что заказ был оплачен вчера вечером, но в личном кабинете до сих пор отображается статус «ожидает оплаты». Он приложил чек и просит не отменять заказ автоматически.",
                "Клиент оплатил заказ, но статус оплаты не обновился. Нужно проверить платёж по чеку и предотвратить автоматическую отмену заказа.",
            ),
            (
                "Команда поддержки заметила, что после последнего обновления часть пользователей не получает SMS-коды. Ошибка проявляется только у номеров одного оператора и уже передана технической группе.",
                "После обновления у части пользователей одного оператора не приходят SMS-коды. Проблема передана технической группе.",
            ),
            (
                "Поставщик предупредил, что партия товара задержится на два дня из-за проверки документов на складе. Менеджер должен предупредить клиентов, у которых доставка была назначена на пятницу.",
                "Партия товара задерживается на два дня из-за проверки документов. Нужно предупредить клиентов с доставкой на пятницу.",
            ),
            (
                "Пользователь просит вернуть деньги за повреждённый товар. Он уже отправил фотографии упаковки, но не указал номер заказа и предпочитаемый способ компенсации.",
                "Пользователь просит возврат за повреждённый товар и отправил фото. Нужно запросить номер заказа и способ компенсации.",
            ),
            (
                "В отчёте за неделю указано, что среднее время ответа поддержки сократилось с 18 до 11 минут, но доля повторных обращений выросла на 6 процентных пунктов из-за неполных инструкций.",
                "Поддержка стала отвечать быстрее: 11 минут вместо 18. При этом повторные обращения выросли на 6 п.п. из-за неполных инструкций.",
            ),
            (
                "Клиент хочет изменить адрес доставки, но заказ уже передан курьерской службе. По правилам адрес можно изменить только до передачи в доставку, поэтому оператор должен предложить перенаправление через службу доставки.",
                "Клиент хочет изменить адрес после передачи заказа в доставку. По правилам нужно предложить перенаправление через курьерскую службу.",
            ),
        ],
    ),
    "support": Example(
        name="support",
        task="classification",
        prompt="Разбери обращение клиента и верни только одну метку.",
        description=(
            "Нужно классифицировать реальные обращения поддержки строго в одну "
            "из меток: оплата, доставка, техника, аккаунт, возврат. В ответе "
            "должна быть только метка без пояснений и дополнительного текста."
        ),
        metric="f1",
        rows=[
            ("С меня дважды списали деньги за один заказ, но в личном кабинете видна только одна покупка.", "оплата"),
            ("После обновления приложение открывается и сразу закрывается на экране профиля.", "техника"),
            ("Курьер не приехал в выбранный интервал, хочу понять новый срок доставки.", "доставка"),
            ("Не могу войти: SMS-код не приходит уже третий раз подряд.", "аккаунт"),
            ("Товар оказался повреждённым, хочу оформить возврат и получить деньги обратно.", "возврат"),
            ("Оплата прошла, но заказ все еще висит как неоплаченный.", "оплата"),
            ("Нужно поменять адрес, пока посылку еще не передали курьеру.", "доставка"),
            ("На сайте бесконечно крутится загрузка при попытке открыть корзину.", "техника"),
            ("Хочу удалить старый номер телефона из профиля и привязать новый.", "аккаунт"),
            ("Я отправил товар назад неделю назад, но статус возврата не меняется.", "возврат"),
            ("Промокод применился, но итоговая сумма в чеке не уменьшилась.", "оплата"),
            ("В пункт выдачи пришел не тот размер, который я заказывал.", "возврат"),
        ],
    ),
    "qa": Example(
        name="qa",
        task="generation",
        prompt="Ответь на вопрос строго по контексту.",
        description=(
            "Нужно извлекать короткий точный ответ строго из контекста. Если "
            "в контексте нет ответа, нужно вернуть: нет данных."
        ),
        metric="em",
        rows=[
            ("Контекст: Заказ 4821 был оплачен 12 июня и передан в доставку 13 июня. Вопрос: когда заказ передали в доставку?", "13 июня"),
            ("Контекст: Возврат средств занимает до 10 рабочих дней после подтверждения заявки. Вопрос: сколько занимает возврат?", "до 10 рабочих дней"),
            ("Контекст: Подписку можно отменить в разделе «Профиль» → «Платежи». Вопрос: где отменить подписку?", "в разделе «Профиль» → «Платежи»"),
            ("Контекст: Доставка в Санкт-Петербург занимает 2-3 дня, а в Казань 4-5 дней. Вопрос: сколько занимает доставка в Казань?", "4-5 дней"),
            ("Контекст: Техподдержка работает ежедневно с 9:00 до 21:00 по московскому времени. Вопрос: до скольки работает поддержка?", "до 21:00"),
            ("Контекст: Бесплатный возврат доступен только для заказов, оформленных за последние 14 дней. Вопрос: какой срок бесплатного возврата?", "14 дней"),
        ],
    ),
}


METHOD_RUNTIME_DEFAULTS: dict[str, dict[str, float | int]] = {
    "hyper_light": {"validation_size": 0.34, "batch_size": 4, "generate_num_samples": 6, "model_temperature": 0.2, "model_max_tokens": 2000},
    "hyper": {"validation_size": 0.4, "batch_size": 1, "generate_num_samples": 6, "model_temperature": 0.25, "model_max_tokens": 2200},
    "rider": {"validation_size": 0.34, "batch_size": 2, "generate_num_samples": 6, "model_temperature": 0.25, "model_max_tokens": 2600},
    "regps": {"validation_size": 0.25, "batch_size": 2, "generate_num_samples": 4, "model_temperature": 0.3, "model_max_tokens": 2200},
}

METHOD_TASK_OVERRIDES: dict[str, dict[str, dict[str, float | int]]] = {
    "classification": {
        "hyper_light": {"validation_size": 0.4, "batch_size": 4, "model_temperature": 0.15, "model_max_tokens": 1800},
        "hyper": {"validation_size": 0.4, "batch_size": 1, "model_temperature": 0.2, "model_max_tokens": 2200},
        "rider": {"validation_size": 0.34, "batch_size": 2, "generate_num_samples": 6, "model_temperature": 0.25, "model_max_tokens": 2600},
    },
    "generation": {
        "hyper_light": {"validation_size": 0.34, "batch_size": 2, "model_temperature": 0.3, "model_max_tokens": 3000},
        "hyper": {"validation_size": 0.4, "batch_size": 1, "generate_num_samples": 6, "model_temperature": 0.25, "model_max_tokens": 2200},
        "rider": {"validation_size": 0.34, "batch_size": 1, "generate_num_samples": 6, "model_temperature": 0.3, "model_max_tokens": 3000},
    },
}


def request_json(base_url: str, path: str, *, method: str = "GET", body: Any | None = None, timeout: int = 30) -> Any:
    data = None if body is None else json.dumps(body, ensure_ascii=True).encode("ascii")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed {path}: {exc}") from exc


def method_params(method: dict[str, Any]) -> dict[str, Any]:
    return {item["name"]: item.get("default") for item in method.get("params", [])}


def runtime_settings(method_id: str, task: str) -> dict[str, Any]:
    result = dict(METHOD_RUNTIME_DEFAULTS.get(method_id, METHOD_RUNTIME_DEFAULTS["hyper_light"]))
    result.update(METHOD_TASK_OVERRIDES.get(task, {}).get(method_id, {}))
    return result


def build_payload(method: dict[str, Any], example: Example, model: str) -> dict[str, Any]:
    settings = runtime_settings(method["id"], example.task)
    return {
        "mode": "single",
        "request": {
            "start_prompt": example.prompt,
            "task": example.task,
            "method": method["id"],
            "metric": example.metric,
            "problem_description": example.description,
            "dataset": [row[0] for row in example.rows],
            "target": [row[1] for row in example.rows],
            "validation_size": settings["validation_size"],
            "train_as_test": False,
            "generate_num_samples": settings["generate_num_samples"],
            "batch_size": settings["batch_size"],
            "model_name": model,
            "model_temperature": settings["model_temperature"],
            "model_max_tokens": settings["model_max_tokens"],
            "method_params": method_params(method),
            "mock": False,
        },
    }


def wait_job(base_url: str, job_id: str, timeout_seconds: int, log) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_stage = None
    while time.time() < deadline:
        job = request_json(base_url, f"/api/jobs/{job_id}", timeout=20)
        stage_line = (job.get("status"), job.get("progress_stage"), job.get("progress_percent"), job.get("progress_message"))
        if stage_line != last_stage:
            log({"event": "progress", "job_id": job_id, "status": job.get("status"), "stage": job.get("progress_stage"), "percent": job.get("progress_percent"), "message": job.get("progress_message")})
            last_stage = stage_line
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(1.5)
    raise TimeoutError(f"Job {job_id} did not finish in {timeout_seconds}s")


def validate_result(job: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if job.get("status") != "completed":
        errors.append(f"status={job.get('status')} error={job.get('error')}")
        return errors
    result = job.get("result") or {}
    if isinstance(result, list):
        errors.append("result is comparison list, expected single result")
        return errors
    final_prompt = (result.get("final_prompt") or "").strip()
    initial_prompt = (result.get("initial_prompt") or "").strip()
    if not final_prompt:
        errors.append("final_prompt is empty")
    if "<ans>" in final_prompt.lower() or "</ans>" in final_prompt.lower():
        errors.append("final_prompt still contains <ans> service tags")
    if result.get("used_mock") is not False:
        errors.append(f"used_mock={result.get('used_mock')}")
    if result.get("init_metric") is None:
        errors.append("init_metric is null")
    if result.get("final_metric") is None:
        errors.append("final_metric is null")
    if result.get("dataset_size", 0) < 2:
        errors.append(f"dataset_size={result.get('dataset_size')}")
    if (
        result.get("method") in {"hyper", "rider", "regps", "hyper_light"}
        and final_prompt == initial_prompt
        and not result.get("quality_guard")
    ):
        errors.append("final_prompt did not change")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8022")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--mode", choices=["all", "model-smoke", "matrix"], default="all")
    parser.add_argument("--methods", default="hyper_light,hyper,rider,regps")
    parser.add_argument("--examples", default="support_reply,summary,support,qa")
    args = parser.parse_args()

    def log(payload: dict[str, Any]) -> None:
        payload["ts"] = round(time.time(), 3)
        print(json.dumps(payload, ensure_ascii=True), flush=True)

    config = request_json(args.base_url, "/api/config")
    methods = request_json(args.base_url, "/api/methods")
    visible_methods = [method for method in methods if method["id"] in METHOD_RUNTIME_DEFAULTS]
    by_id = {method["id"]: method for method in visible_methods}
    default_model = config["defaultModel"]
    model_options = [item["value"] for item in config.get("modelOptions", [])]
    log({"event": "config", "default_model": default_model, "models": model_options, "methods": [m["id"] for m in visible_methods]})

    runs: list[tuple[str, dict[str, Any], str, str, str]] = []
    if args.mode in {"all", "model-smoke"}:
        for model in model_options:
            runs.append((f"model-smoke:{model}", build_payload(by_id["hyper_light"], EXAMPLES["support_reply"], model), model, "hyper_light", "support_reply"))
    if args.mode in {"all", "matrix"}:
        selected_examples = [item.strip() for item in args.examples.split(",") if item.strip()]
        selected_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
        for example_name in selected_examples:
            for method_id in selected_methods:
                runs.append((f"matrix:{example_name}:{method_id}", build_payload(by_id[method_id], EXAMPLES[example_name], default_model), default_model, method_id, example_name))

    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, (name, payload, model, method_id, example_name) in enumerate(runs, start=1):
        log({"event": "run-start", "index": index, "total": len(runs), "name": name, "model": model, "method": method_id, "example": example_name})
        started = time.time()
        try:
            job = request_json(args.base_url, "/api/jobs", method="POST", body=payload, timeout=60)
            final_job = wait_job(args.base_url, job["job_id"], args.timeout, log)
            errors = validate_result(final_job)
            result = final_job.get("result") or {}
            summary = {
                "event": "run-finish",
                "name": name,
                "status": final_job.get("status"),
                "elapsed_wall": round(time.time() - started, 1),
                "method": method_id,
                "example": example_name,
                "model": model,
                "init_metric": result.get("init_metric") if isinstance(result, dict) else None,
                "final_metric": result.get("final_metric") if isinstance(result, dict) else None,
                "delta": result.get("metric_delta") if isinstance(result, dict) else None,
                "final_len": len((result.get("final_prompt") or "")) if isinstance(result, dict) else None,
                "quality_guard": result.get("quality_guard") if isinstance(result, dict) else None,
                "errors": errors,
            }
            summaries.append(summary)
            log(summary)
            if errors:
                failures.append(summary)
        except Exception as exc:  # noqa: BLE001 - E2E report should continue
            failure = {
                "event": "run-exception",
                "name": name,
                "method": method_id,
                "example": example_name,
                "model": model,
                "elapsed_wall": round(time.time() - started, 1),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            failures.append(failure)
            log(failure)

    log({"event": "summary", "total": len(runs), "failed": len(failures), "failures": failures, "runs": summaries})
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
