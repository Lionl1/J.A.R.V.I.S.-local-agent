.PHONY: sync dev run

sync:
	uv sync

dev:
	uv run uvicorn src.jarvis_backend.server:app --host 0.0.0.0 --port 8000 --reload

run:
	uv run python -m src.jarvis_backend.main
