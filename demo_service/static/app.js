const state = {
  methods: [],
  config: {},
  task: "generation",
};

const examples = {
  support: {
    task: "classification",
    prompt: "Classify each customer support request by intent.",
    description: "Classify support messages into one of the known labels.",
    metric: "f1",
    rows: [
      ["My refund still has not arrived.", "billing"],
      ["The app crashes when I open settings.", "technical"],
      ["Can I change the delivery address?", "shipping"],
      ["I was charged twice for the same order.", "billing"],
      ["The login code never arrives.", "technical"],
      ["Where is my package now?", "shipping"],
    ],
  },
  summary: {
    task: "generation",
    prompt: "Summarize the text in one concise paragraph.",
    description: "Produce a factual one-paragraph summary that preserves the main claim.",
    metric: "rouge",
    rows: [
      [
        "CoolPrompt lets teams compare prompt optimization methods from a single interface.",
        "CoolPrompt provides one interface for comparing prompt optimization methods.",
      ],
      [
        "The library supports data-driven optimization when examples and targets are available.",
        "The library can optimize prompts using datasets with expected targets.",
      ],
      [
        "A Railway demo makes the system available without local installation.",
        "Railway deployment lets users try the system without installing it locally.",
      ],
    ],
  },
  translation: {
    task: "generation",
    prompt: "Translate the text into polished Russian while preserving terminology.",
    description: "Translate English source text into natural Russian and keep technical terms consistent.",
    metric: "meteor",
    rows: [
      ["The model returned a structured response.", "Модель вернула структурированный ответ."],
      ["The pipeline validates every prompt candidate.", "Пайплайн проверяет каждого кандидата промпта."],
      ["The user can compare optimization methods.", "Пользователь может сравнивать методы оптимизации."],
    ],
  },
};

const generationMetrics = ["bertscore", "rouge", "meteor", "bleu", "em", "llm_as_judge", "geval"];
const classificationMetrics = ["f1", "accuracy"];

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
  $("runtimeStatus").textContent = config.hasOpenAIKey
    ? `${config.defaultModel} ready`
    : config.forceMock
      ? "Mock mode"
      : "OPENAI_API_KEY is missing";
  $("modelName").placeholder = config.defaultModel;
  $("mockMode").checked = Boolean(config.forceMock);
  $("mockMode").disabled = !config.allowMock && !config.forceMock;
  renderMethods();
  setExample("support");
}

function renderMethods() {
  const select = $("methodSelect");
  select.innerHTML = "";
  state.methods.forEach((method) => {
    const option = document.createElement("option");
    option.value = method.id;
    option.textContent = `${method.label}${method.legacy ? " · legacy" : ""}`;
    select.appendChild(option);
  });
  select.value = "rider";

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
  select.value = options.includes(current) ? current : options[0];
}

function renderParams() {
  const method = methodById($("methodSelect").value);
  const panel = $("paramsPanel");
  panel.innerHTML = "";
  if (!method || method.params.length === 0) {
    panel.innerHTML = `<div class="hint">No method-specific parameters.</div>`;
    return;
  }
  method.params.forEach((param) => {
    const label = document.createElement("label");
    label.dataset.param = param.name;
    let input = "";
    if (param.type === "bool") {
      input = `<select data-param-input="${param.name}"><option value="false">False</option><option value="true">True</option></select>`;
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
  $("methodHint").textContent = method ? `${method.family}: ${method.description}` : "";
}

function setTask(task) {
  state.task = task;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.task === task);
  });
  renderMetricOptions();
}

function addDatasetRow(input = "", target = "") {
  const row = document.createElement("div");
  row.className = "dataset-row";
  row.innerHTML = `
    <input class="dataset-input" type="text" placeholder="Input example" value="${escapeHtml(input)}" />
    <input class="dataset-target" type="text" placeholder="Target / label" value="${escapeHtml(target)}" />
    <button type="button" class="icon-btn remove-row" title="Remove row">×</button>
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
  setTask(example.task);
  $("startPrompt").value = example.prompt;
  $("problemDescription").value = example.description;
  $("metricSelect").value = example.metric;
  $("datasetRows").innerHTML = "";
  example.rows.forEach(([input, target]) => addDatasetRow(input, target));
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
    model_name: $("modelName").value.trim() || null,
    model_temperature: Number($("modelTemperature").value),
    model_max_tokens: Number($("modelMaxTokens").value),
    method_params: collectParams(),
    mock: $("mockMode").checked,
  };
}

async function createJob() {
  const compareMode = $("compareMode").checked;
  const base = buildBaseRequest();
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
  setStatus("queued", "Job queued");
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
  pollJob(job.job_id);
}

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const job = await response.json();
  setStatus(job.status, `Job ${job.job_id.slice(0, 8)} · ${job.status}`);
  if (job.status === "queued" || job.status === "running") {
    window.setTimeout(() => pollJob(jobId), 1200);
    return;
  }
  setBusy(false);
  if (job.status === "failed") {
    $("runDetails").textContent = job.error || "Unknown error";
    return;
  }
  renderResult(job.result);
}

function setBusy(busy) {
  $("runButton").disabled = busy;
  $("runButton").textContent = busy ? "Running..." : "Run optimization";
}

function setStatus(status, line) {
  $("statusPill").textContent = status;
  $("statusPill").className = `status-pill ${status}`;
  $("jobLine").textContent = line;
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
          <strong>${label}${isBest ? " · best" : ""}</strong>
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
  button.addEventListener("click", () => setTask(button.dataset.task));
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => setExample(button.dataset.example));
});

$("methodSelect").addEventListener("change", () => {
  renderParams();
  renderMethodHint();
});

$("compareMode").addEventListener("change", () => {
  $("compareMethods").classList.toggle("hidden", !$("compareMode").checked);
});

$("addRow").addEventListener("click", () => addDatasetRow());

$("runButton").addEventListener("click", () => {
  createJob().catch((error) => {
    setBusy(false);
    setStatus("failed", "Request failed");
    $("runDetails").textContent = String(error);
  });
});

loadConfig().catch((error) => {
  $("runtimeStatus").textContent = "Config failed";
  $("runDetails").textContent = String(error);
});
