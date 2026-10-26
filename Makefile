.PHONY: sync dev run rollback test

sync:
	uv sync

dev:
	uv run uvicorn src.jarvis_backend.server:app --host 0.0.0.0 --port 8000 --reload

run:
	uv run python -m src.jarvis_backend.main

rollback:
	python3 -m src.jarvis_backend.git_guard

test:
	uv run --with pytest --with pytest-asyncio --with httpx pytest -q
