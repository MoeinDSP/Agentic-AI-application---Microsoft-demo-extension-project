install:
	uv sync --group dev

run:
	uv run uvicorn agent3_mcp.main:app --factory --reload --host 127.0.0.1 --port 8090

test:
	uv run pytest

lint:
	uv run ruff check .
