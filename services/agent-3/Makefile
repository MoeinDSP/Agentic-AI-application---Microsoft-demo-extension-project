install:
	uv sync --group dev

run:
	uv run python -m uvicorn agent3.main:create_app --factory --reload --host 127.0.0.1 --port 8080

test:
	uv run pytest

lint:
	uv run ruff check .
