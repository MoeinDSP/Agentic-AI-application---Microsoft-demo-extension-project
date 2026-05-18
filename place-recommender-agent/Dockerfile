FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /place-recommender-agent

ENV PATH="/root/.local/bin:/place-recommender-agent/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
# uv run uses project root as cwd; app code lives under app/ with flat imports (main, agent, …)
ENV PYTHONPATH=/place-recommender-agent/app

# Install deps first (better layer cache when only app/ changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY app ./app

WORKDIR /place-recommender-agent/app

EXPOSE 8000

# Railway (and others) set PORT; default matches local docker-compose
CMD ["sh", "-c", "exec uv run --directory /place-recommender-agent uvicorn main:agent_factory.web --host 0.0.0.0 --port ${PORT:-8000}"]
