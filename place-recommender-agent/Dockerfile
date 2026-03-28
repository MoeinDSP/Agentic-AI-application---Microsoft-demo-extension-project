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

COPY pyproject.toml ./

RUN uv sync