FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8070

CMD ["sh", "-c", "uvicorn agent4.main:app --factory --host 0.0.0.0 --port ${PORT:-8070}"]
