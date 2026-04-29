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

# COPY pyproject.toml uv.lock ./

# RUN uv sync --frozen

# COPY app/ ./app/

# EXPOSE 8000

# CMD ["uvicorn", "main:agent", "--app-dir", "app", "--host", "0.0.0.0", "--port", "8000"]

COPY pyproject.toml ./

RUN uv sync