FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY coolprompt ./coolprompt
COPY demo_service ./demo_service

RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install -r demo_service/requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn demo_service.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
