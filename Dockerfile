FROM ghcr.io/astral-sh/uv:0.9.11-python3.13-bookworm-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/app/.venv"
ENV PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
