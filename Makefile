.PHONY: sync dev start run rollback test

sync:
	uv sync

# Разработка: перезапуск при любой правке файлов.
dev:
	uv run uvicorn src.jarvis_backend.server:app --host 127.0.0.1 --port 8000 --reload

# Повседневный запуск фонового агента: после самоправки ядра Джарвис перезапускается сам.
start:
	uv run uvicorn src.jarvis_backend.server:app --host 127.0.0.1 --port 8000

run:
	uv run python -m src.jarvis_backend.main

rollback:
	python3 -m src.jarvis_backend.git_guard

test:
	uv run pytest -q
