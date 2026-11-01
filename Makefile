.PHONY: sync dev start run rollback test autostart-install autostart-uninstall autostart-status on off stop

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

# Автозапуск при входе в систему (macOS LaunchAgent / Linux systemd --user).
autostart-install:
	uv run python -m src.jarvis_backend.autostart install

autostart-uninstall:
	python3 -m src.jarvis_backend.autostart uninstall

autostart-status:
	python3 -m src.jarvis_backend.autostart status

# Аварийный выключатель: остановить всё (и запущенное вручную), убрать автозапуск, выгрузить модель.
# Работает системным python3, даже если правка агента сломала проект.
off:
	python3 -m src.jarvis_backend.autostart off

# Включить всё обратно: LM Studio + модель + автозапуск Джарвиса.
on:
	python3 -m src.jarvis_backend.autostart on

# Остановить до следующего входа в систему.
stop:
	python3 -m src.jarvis_backend.autostart stop
