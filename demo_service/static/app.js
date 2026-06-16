const state = {
  methods: [],
  config: {},
  task: "generation",
};

const examples = {
  support: {
    task: "classification",
    prompt: "Определи тему обращения.",
    description: "Нужно классифицировать реальные обращения поддержки строго в одну из меток: оплата, доставка, техника, аккаунт, возврат. В ответе должна быть только метка без пояснений.",
    metric: "f1",
    rows: [
      ["С меня дважды списали деньги за один заказ, но в личном кабинете видна только одна покупка.", "оплата"],
      ["После обновления приложение открывается и сразу закрывается на экране профиля.", "техника"],
      ["Курьер не приехал в выбранный интервал, хочу понять новый срок доставки.", "доставка"],
      ["Не могу войти: SMS-код не приходит уже третий раз подряд.", "аккаунт"],
      ["Товар оказался поврежденным, хочу оформить возврат и получить деньги обратно.", "возврат"],
      ["Оплата прошла, но заказ все еще висит как неоплаченный.", "оплата"],
      ["Нужно поменять адрес, пока посылку еще не передали курьеру.", "доставка"],
      ["На сайте бесконечно крутится загрузка при попытке открыть корзину.", "техника"],
      ["Хочу удалить старый номер телефона из профиля и привязать новый.", "аккаунт"],
      ["Я отправил товар назад неделю назад, но статус возврата не меняется.", "возврат"],
      ["Промокод применился, но итоговая сумма в чеке не уменьшилась.", "оплата"],
      ["В пункт выдачи пришел не тот размер, который я заказывал.", "возврат"],
    ],
  },
  support_reply: {
    task: "generation",
    prompt: "Ответь клиенту.",
    description: "Нужно генерировать короткие ответы поддержки: признать проблему, дать конкретный следующий шаг, сохранить спокойный профессиональный тон и не обещать того, чего оператор не может гарантировать.",
    metric: "rouge",
    rows: [
      [
        "Клиент пишет, что оплатил заказ, но статус до сих пор не изменился.",
        "Здравствуйте! Проверим оплату и статус заказа. Пришлите, пожалуйста, номер заказа или чек, чтобы мы быстрее нашли платёж.",
      ],
      [
        "Клиент просит перенести доставку на другой день.",
        "Здравствуйте! Да, дату доставки можно изменить. Напишите удобный день и интервал, а мы проверим доступные варианты.",
      ],
      [
        "Клиент сообщает, что приложение не открывается после обновления.",
        "Здравствуйте! Попробуйте перезапустить приложение и очистить кэш. Если ошибка повторится, пришлите модель устройства и скриншот.",
      ],
      [
        "Клиент получил товар с поврежденной упаковкой и просит заменить заказ.",
        "Здравствуйте! Нам жаль, что заказ пришёл повреждённым. Пришлите фото упаковки и товара, а мы проверим варианты замены или возврата.",
      ],
      [
        "Клиент не может войти в аккаунт: код подтверждения не приходит.",
        "Здравствуйте! Проверьте, пожалуйста, номер телефона и папку со спамом. Если код не придёт повторно, напишите нам, и мы проверим вход вручную.",
      ],
      [
        "Клиент спрашивает, почему промокод не уменьшил итоговую сумму заказа.",
        "Здравствуйте! Проверим условия промокода и состав заказа. Пришлите код акции и номер заказа, чтобы мы быстро нашли причину.",
      ],
      [
        "Клиент просит отменить заказ, который уже передан в доставку.",
        "Здравствуйте! Проверим, можно ли ещё отменить отправление. Пришлите номер заказа, и мы подскажем доступные варианты.",
      ],
    ],
  },
  qa: {
    task: "generation",
    prompt: "Ответь на вопрос.",
    description: "Нужно извлекать короткий точный ответ строго из контекста. Если в контексте нет ответа, нужно вернуть: нет данных.",
    metric: "em",
    rows: [
      ["Контекст: Заказ 4821 был оплачен 12 июня и передан в доставку 13 июня. Вопрос: когда заказ передали в доставку?", "13 июня"],
      ["Контекст: Возврат средств занимает до 10 рабочих дней после подтверждения заявки. Вопрос: сколько занимает возврат?", "до 10 рабочих дней"],
      ["Контекст: Подписку можно отменить в разделе «Профиль» → «Платежи». Вопрос: где отменить подписку?", "в разделе «Профиль» → «Платежи»"],
      ["Контекст: Доставка в Санкт-Петербург занимает 2-3 дня, а в Казань 4-5 дней. Вопрос: сколько занимает доставка в Казань?", "4-5 дней"],
      ["Контекст: Техподдержка работает ежедневно с 9:00 до 21:00 по московскому времени. Вопрос: до скольки работает поддержка?", "до 21:00"],
      ["Контекст: Бесплатный возврат доступен только для заказов, оформленных за последние 14 дней. Вопрос: какой срок бесплатного возврата?", "14 дней"],
    ],
  },
};

const defaultExampleByTask = {
  classification: "support",
  generation: "support_reply",
};

const generationMetrics = ["bertscore", "rouge", "meteor", "bleu", "em", "llm_as_judge", "geval"];
const classificationMetrics = ["f1", "accuracy"];
const taskMetricDefaults = {
  classification: "f1",
  generation: "rouge",
};
const modelFallbackOptions = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"];
const progressSteps = [
  { id: "queued", label: "Очередь" },
  { id: "preparing", label: "Подготовка" },
  { id: "loading", label: "Библиотека" },
  { id: "model", label: "Модель" },
  { id: "optimizing", label: "Оптимизация" },
  { id: "collecting", label: "Результат" },
  { id: "completed", label: "Готово" },
];

const methodRuntimeDefaults = {
  hyper_light: {
    validationSize: 0.34,
    batchSize: 4,
    generateSamples: 6,
    modelTemperature: 0.2,
    modelMaxTokens: 2000,
  },
  hyper: {
    validationSize: 0.34,
    batchSize: 3,
    generateSamples: 8,
    modelTemperature: 0.25,
    modelMaxTokens: 2500,
  },
  rider: {
    validationSize: 0.4,
    batchSize: 2,
    generateSamples: 12,
    modelTemperature: 0.25,
    modelMaxTokens: 3000,
  },
  regps: {
    validationSize: 0.34,
    batchSize: 3,
    generateSamples: 8,
    modelTemperature: 0.35,
    modelMaxTokens: 2500,
  },
  compress: {
    validationSize: 0.34,
    batchSize: 4,
    generateSamples: 6,
    modelTemperature: 0.1,
    modelMaxTokens: 1600,
  },
  reflective: {
    validationSize: 0.34,
    batchSize: 3,
    generateSamples: 8,
    modelTemperature: 0.35,
    modelMaxTokens: 2500,
  },
  distill: {
    validationSize: 0.34,
    batchSize: 3,
    generateSamples: 8,
    modelTemperature: 0.25,
    modelMaxTokens: 2200,
  },
};

const methodTaskOverrides = {
  classification: {
    hyper_light: {
      validationSize: 0.4,
      batchSize: 4,
      modelTemperature: 0.15,
      modelMaxTokens: 1800,
    },
    hyper: {
      validationSize: 0.4,
      batchSize: 3,
      modelTemperature: 0.2,
      modelMaxTokens: 2400,
    },
    rider: {
      validationSize: 0.4,
      batchSize: 2,
      generateSamples: 12,
      modelTemperature: 0.25,
      modelMaxTokens: 3000,
    },
  },
  generation: {
    hyper_light: {
      validationSize: 0.34,
      batchSize: 2,
      modelTemperature: 0.3,
      modelMaxTokens: 3000,
    },
    hyper: {
      validationSize: 0.34,
      batchSize: 2,
      generateSamples: 8,
      modelTemperature: 0.35,
      modelMaxTokens: 3500,
    },
    rider: {
      validationSize: 0.34,
      batchSize: 1,
      generateSamples: 8,
      modelTemperature: 0.35,
      modelMaxTokens: 4000,
    },
    compress: {
      validationSize: 0.34,
      batchSize: 2,
      modelTemperature: 0.1,
      modelMaxTokens: 2200,
    },
  },
};

const $ = (id) => document.getElementById(id);

function fmtMetric(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return Number(value).toFixed(4);
}

function methodById(id) {
  return state.methods.find((method) => method.id === id);
}

async function loadConfig() {
  const [config, methods] = await Promise.all([
    fetch("/api/config").then((r) => r.json()),
    fetch("/api/methods").then((r) => r.json()),
  ]);
  state.config = config;
  state.methods = methods;
  renderModelOptions(config);
  updateRuntimeStatus();
  $("mockMode").checked = Boolean(config.forceMock);
  $("mockMode").disabled = !config.allowMock && !config.forceMock;
  renderMethods();
  renderProgress();
  setExample("support");
}

function renderModelOptions(config) {
  const select = $("modelSelect");
  const options = (config.modelOptions || modelFallbackOptions.map((value) => ({ value, label: value })));
  const values = new Set(options.map((item) => item.value));
  if (config.defaultModel && !values.has(config.defaultModel)) {
    options.unshift({ value: config.defaultModel, label: config.defaultModel });
  }
  select.innerHTML = "";
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    select.appendChild(option);
  });
  const custom = document.createElement("option");
  custom.value = "__custom__";
  custom.textContent = "Другая модель...";
  select.appendChild(custom);
  select.value = config.defaultModel || options[0]?.value || "gpt-4o-mini";
  $("modelName").placeholder = config.defaultModel || "openai/gpt-4o-mini";
  toggleCustomModel();
}

function toggleCustomModel() {
  const isCustom = $("modelSelect").value === "__custom__";
  $("customModelLabel").classList.toggle("hidden", !isCustom);
  updateRuntimeStatus();
}

function currentModelName() {
  const selectedModel = $("modelSelect").value;
  if (selectedModel === "__custom__") {
    return $("modelName").value.trim();
  }
  return selectedModel;
}

function updateRuntimeStatus() {
  if (!state.config.hasOpenAIKey) {
    $("runtimeStatus").textContent = state.config.forceMock
      ? "Режим: тестовый"
      : "Нет OPENAI_API_KEY";
    return;
  }
  const model = currentModelName();
  $("runtimeStatus").textContent = model ? `Модель: ${model}` : "Модель: не выбрана";
}

function renderMethods() {
  const select = $("methodSelect");
  select.innerHTML = "";
  state.methods.forEach((method) => {
    const option = document.createElement("option");
    option.value = method.id;
    option.textContent = `${method.label}${method.legacy ? " · устаревший" : ""}`;
    select.appendChild(option);
  });
  select.value = "hyper_light";
  applyMethodRuntimeDefaults(select.value);

  const compare = $("compareMethods");
  compare.innerHTML = "";
  state.methods.forEach((method) => {
    const label = document.createElement("label");
    label.className = "method-check";
    label.innerHTML = `<input type="checkbox" value="${method.id}" ${["hyper_light", "hyper", "rider"].includes(method.id) ? "checked" : ""} /> <span>${method.label}</span>`;
    compare.appendChild(label);
  });
  renderMetricOptions();
  renderParams();
  renderMethodHint();
}

function setNumberValue(id, value) {
  const input = $(id);
  if (input) input.value = String(value);
}

function applyMethodRuntimeDefaults(methodId) {
  const defaults = {
    ...(methodRuntimeDefaults[methodId] || methodRuntimeDefaults.hyper_light),
    ...(methodTaskOverrides[state.task]?.[methodId] || {}),
  };
  setNumberValue("validationSize", defaults.validationSize);
  setNumberValue("batchSize", defaults.batchSize);
  setNumberValue("generateSamples", defaults.generateSamples);
  setNumberValue("modelTemperature", defaults.modelTemperature);
  setNumberValue("modelMaxTokens", defaults.modelMaxTokens);
}

function renderMetricOptions() {
  const options = state.task === "classification" ? classificationMetrics : generationMetrics;
  const select = $("metricSelect");
  const current = select.value;
  select.innerHTML = "";
  options.forEach((metric) => {
    const option = document.createElement("option");
    option.value = metric;
    option.textContent = metric;
    select.appendChild(option);
  });
  select.value = options.includes(current) ? current : taskMetricDefaults[state.task] || options[0];
}

function renderParams() {
  const method = methodById($("methodSelect").value);
  const panel = $("paramsPanel");
  panel.innerHTML = "";
  if (!method || method.params.length === 0) {
    panel.innerHTML = `<div class="hint">У метода нет отдельных параметров.</div>`;
    return;
  }
  method.params.forEach((param) => {
    const label = document.createElement("label");
    label.dataset.param = param.name;
    let input = "";
    if (param.type === "bool") {
      input = `<select data-param-input="${param.name}"><option value="false">Нет</option><option value="true">Да</option></select>`;
    } else {
      const step = param.type === "float" ? (param.step || 0.01) : 1;
      input = `<input data-param-input="${param.name}" type="number" min="${param.min ?? ""}" max="${param.max ?? ""}" step="${step}" value="${param.default ?? ""}" />`;
    }
    label.innerHTML = `${param.label}${input}`;
    panel.appendChild(label);
    const control = label.querySelector("[data-param-input]");
    if (param.type === "bool") control.value = String(Boolean(param.default));
  });
}

function renderMethodHint() {
  const method = methodById($("methodSelect").value);
  $("methodHint").textContent = method ? `${capitalizeFirst(method.family)}. ${method.description}` : "";
}

function capitalizeFirst(text) {
  if (!text) return "";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function setTask(task, options = {}) {
  state.task = task;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.task === task);
  });
  renderMetricOptions();
  if (options.applyDefaults !== false) {
    applyMethodRuntimeDefaults($("methodSelect").value);
  }
}

function addDatasetRow(input = "", target = "") {
  const row = document.createElement("div");
  row.className = "dataset-row";
  row.innerHTML = `
    <input class="dataset-input" type="text" placeholder="Входной пример" value="${escapeHtml(input)}" />
    <input class="dataset-target" type="text" placeholder="Эталон / метка" value="${escapeHtml(target)}" />
    <button type="button" class="icon-btn remove-row" title="Удалить строку">×</button>
  `;
  row.querySelector(".remove-row").addEventListener("click", () => row.remove());
  $("datasetRows").appendChild(row);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setExample(name) {
  const example = examples[name];
  if (!example) return;
  setTask(example.task, { applyDefaults: false });
  $("startPrompt").value = example.prompt;
  $("problemDescription").value = example.description;
  $("metricSelect").value = example.metric;
  $("datasetRows").innerHTML = "";
  example.rows.forEach(([input, target]) => addDatasetRow(input, target));
  applyMethodRuntimeDefaults($("methodSelect").value);
}

function collectDataset() {
  const rows = [...document.querySelectorAll(".dataset-row")];
  const pairs = rows
    .map((row) => [
      row.querySelector(".dataset-input").value.trim(),
      row.querySelector(".dataset-target").value.trim(),
    ])
    .filter(([input, target]) => input || target);
  if (pairs.length === 0) return { dataset: null, target: null };
  return {
    dataset: pairs.map(([input]) => input),
    target: pairs.map(([, target]) => target),
  };
}

function collectParams() {
  const params = {};
  document.querySelectorAll("[data-param-input]").forEach((input) => {
    const key = input.dataset.paramInput;
    if (input.value === "") return;
    if (input.tagName === "SELECT") {
      params[key] = input.value === "true";
    } else {
      params[key] = Number(input.value);
    }
  });
  return params;
}

function buildBaseRequest() {
  const { dataset, target } = collectDataset();
  const modelName = currentModelName() || null;
  return {
    start_prompt: $("startPrompt").value.trim(),
    task: state.task,
    method: $("methodSelect").value,
    metric: $("metricSelect").value,
    problem_description: $("problemDescription").value.trim() || null,
    dataset,
    target,
    validation_size: Number($("validationSize").value),
    train_as_test: false,
    generate_num_samples: Number($("generateSamples").value),
    batch_size: Number($("batchSize").value),
    model_name: modelName,
    model_temperature: Number($("modelTemperature").value),
    model_max_tokens: Number($("modelMaxTokens").value),
    method_params: collectParams(),
    mock: $("mockMode").checked,
  };
}

async function createJob() {
  const compareMode = $("compareMode").checked;
  const base = buildBaseRequest();
  if (!base.mock && (!base.dataset || base.dataset.length < 2)) {
    throw new Error("Для реального запуска добавьте минимум 2 строки данных или включите тестовый запуск.");
  }
  let payload;
  if (compareMode) {
    const methods = [...document.querySelectorAll("#compareMethods input:checked")].map((input) => input.value);
    payload = {
      mode: "compare",
      compare: {
        base,
        methods,
        method_params_by_method: { [base.method]: base.method_params },
      },
    };
  } else {
    payload = { mode: "single", request: base };
  }

  setBusy(true);
  setStatus("queued", "Задача поставлена в очередь");
  renderProgress({
    progress_stage: "queued",
    progress_percent: 5,
    progress_message: "Отправляем задачу на сервер",
    status: "queued",
  });
  renderLiveDetails({
    job_id: "pending",
    status: "queued",
    progress_stage: "queued",
    progress_percent: 5,
    progress_message: "Отправляем задачу на сервер",
    updated_at: Date.now() / 1000,
  });
  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text);
  }
  const job = await response.json();
  renderProgress(job);
  renderLiveDetails(job);
  pollJob(job.job_id);
}

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const job = await response.json();
  setStatus(job.status, `Задача ${job.job_id.slice(0, 8)} · ${statusText(job.status)}`);
  renderProgress(job);
  renderLiveDetails(job);
  if (job.status === "queued" || job.status === "running") {
    window.setTimeout(() => pollJob(jobId), 1200);
    return;
  }
  setBusy(false);
  if (job.status === "failed") {
    $("runDetails").textContent = job.error || "Неизвестная ошибка";
    return;
  }
  renderResult(job.result);
}

function setBusy(busy) {
  $("runButton").disabled = busy;
  $("runButton").textContent = busy ? "Оптимизация..." : "Запустить оптимизацию";
  setControlsDisabled(busy);
}

function setControlsDisabled(disabled) {
  document.querySelector(".sidebar").classList.toggle("is-busy", disabled);
  document
    .querySelectorAll(".sidebar input, .sidebar select, .sidebar textarea, .sidebar button")
    .forEach((control) => {
      control.disabled = disabled;
    });
  $("runButton").disabled = disabled;
  if (!disabled) {
    $("mockMode").disabled = !state.config.allowMock && !state.config.forceMock;
    toggleCustomModel();
  }
}

function setStatus(status, line) {
  $("statusPill").textContent = statusText(status);
  $("statusPill").className = `status-pill ${status}`;
  $("jobLine").textContent = line;
}

function statusText(status) {
  return {
    idle: "готово",
    queued: "в очереди",
    running: "в работе",
    completed: "готово",
    failed: "ошибка",
  }[status] || status;
}

function renderProgress(job = null) {
  const stage = job?.progress_stage || "idle";
  const percent = Number(job?.progress_percent || 0);
  const status = job?.status || "idle";
  const rawIndex = progressSteps.findIndex((step) => step.id === stage);
  const activeIndex = rawIndex >= 0 ? rawIndex : (status === "failed" ? progressSteps.length - 1 : -1);
  const stepFill = activeIndex <= 0 ? 0 : (activeIndex / (progressSteps.length - 1)) * 100;
  $("progressMessage").textContent = job?.progress_message || "Задача ещё не запускалась";
  $("progressPercent").textContent = `${Math.max(0, Math.min(100, percent))}%`;
  $("progressBar").style.width = `${Math.max(0, Math.min(100, stepFill))}%`;
  $("progressSteps").style.setProperty("--step-fill-scale", String(Math.max(0, Math.min(1, stepFill / 100))));
  $("progressSteps").innerHTML = progressSteps
    .map((step, index) => {
      let cls = "pending";
      if (status === "failed") {
        cls = index <= activeIndex ? "failed" : "pending";
      } else if (status === "completed" || step.id === "completed" && stage === "completed") {
        cls = index <= progressSteps.length - 1 ? "done" : "pending";
      } else if (index < activeIndex) {
        cls = "done";
      } else if (index === activeIndex) {
        cls = "active";
      }
      return `
        <div class="progress-step ${cls}">
          <span></span>
          <strong>${step.label}</strong>
        </div>
      `;
    })
    .join("");
}

function renderLiveDetails(job) {
  if (!job || !["queued", "running"].includes(job.status)) return;
  $("runDetails").textContent = JSON.stringify(
    {
      job_id: job.job_id,
      status: job.status,
      stage: job.progress_stage,
      progress_percent: job.progress_percent,
      message: job.progress_message,
      updated_at: new Date((job.updated_at || Date.now() / 1000) * 1000).toISOString(),
    },
    null,
    2,
  );
}

function renderResult(result) {
  if (Array.isArray(result)) {
    renderComparison(result);
    const best = [...result].sort((a, b) => (b.final_metric ?? 0) - (a.final_metric ?? 0))[0];
    renderSingle(best);
    return;
  }
  $("comparison").classList.add("hidden");
  renderSingle(result);
}

function renderSingle(result) {
  $("initialPrompt").textContent = result.initial_prompt || "";
  $("finalPrompt").textContent = result.final_prompt || "";
  $("initMetric").textContent = fmtMetric(result.init_metric);
  $("finalMetric").textContent = fmtMetric(result.final_metric);
  $("deltaMetric").textContent = fmtMetric(result.metric_delta);
  $("elapsedMetric").textContent = `${Number(result.elapsed_seconds || 0).toFixed(1)}s`;
  $("runDetails").textContent = JSON.stringify(result, null, 2);
}

function renderComparison(results) {
  const box = $("comparison");
  box.classList.remove("hidden");
  const max = Math.max(...results.map((item) => item.final_metric || 0), 0.001);
  const best = Math.max(...results.map((item) => item.final_metric || 0), 0);
  box.innerHTML = results
    .map((item) => {
      const width = Math.max(2, ((item.final_metric || 0) / max) * 100);
      const label = methodById(item.method)?.label || item.method;
      const isBest = (item.final_metric || 0) === best;
      return `
        <div class="comparison-row">
          <strong>${label}${isBest ? " · лучший" : ""}</strong>
          <div class="scorebar"><span style="width:${width}%"></span></div>
          <span>${fmtMetric(item.final_metric)}</span>
        </div>
      `;
    })
    .join("");
}

document.addEventListener("click", (event) => {
  const copy = event.target.closest("[data-copy]");
  if (copy) {
    navigator.clipboard.writeText($(copy.dataset.copy).textContent);
  }
});

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => {
    const exampleName = defaultExampleByTask[button.dataset.task];
    if (exampleName) {
      setExample(exampleName);
    } else {
      setTask(button.dataset.task);
    }
  });
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => setExample(button.dataset.example));
});

$("methodSelect").addEventListener("change", () => {
  applyMethodRuntimeDefaults($("methodSelect").value);
  renderParams();
  renderMethodHint();
});

$("compareMode").addEventListener("change", () => {
  $("compareMethods").classList.toggle("hidden", !$("compareMode").checked);
});

$("modelSelect").addEventListener("change", toggleCustomModel);

$("modelName").addEventListener("input", updateRuntimeStatus);

$("addRow").addEventListener("click", () => addDatasetRow());

$("runButton").addEventListener("click", () => {
  createJob().catch((error) => {
    setBusy(false);
    setStatus("failed", "Запрос не выполнен");
    renderProgress({
      progress_stage: "failed",
      progress_percent: 100,
      progress_message: "Запрос не выполнен",
      status: "failed",
    });
    $("runDetails").textContent = String(error);
  });
});

loadConfig().catch((error) => {
  $("runtimeStatus").textContent = "Ошибка конфигурации";
  $("runDetails").textContent = String(error);
});
