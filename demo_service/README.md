# CoolPrompt Interface Demo

One-page FastAPI demo for trying CoolPrompt methods without installing the
library locally. It exposes:

- method selection across CoolPrompt optimizers;
- comparison mode for several methods on the same dataset;
- method-specific hyperparameters;
- dataset/target inputs for train/validation-aware methods;
- Railway-ready Docker deployment.

## Local Run

```bash
pip install -e .
pip install -r demo_service/requirements.txt
set OPENAI_API_KEY=...
python -m uvicorn demo_service.app:app --host 0.0.0.0 --port 8000
```

For a no-cost UI smoke run:

```bash
set COOLPROMPT_DEMO_MOCK=1
python -m uvicorn demo_service.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Railway

Railway detects the root `Dockerfile` automatically. Required variable:

- `OPENAI_API_KEY`

Useful optional variables:

- `COOLPROMPT_DEMO_MODEL=gpt-4o-mini`
- `COOLPROMPT_DEMO_WORKERS=2`
- `COOLPROMPT_MAX_COMPARE_METHODS=4`
- `COOLPROMPT_DEMO_MOCK=1` for a no-cost mock deployment
