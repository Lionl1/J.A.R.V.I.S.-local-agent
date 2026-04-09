# J.A.R.V.I.S. Local AI Agent

Полностью локальный, асинхронный, проактивный голосовой ассистент с доступом к ОС, браузерным веб-интерфейсом и поддержкой Model Context Protocol (MCP).

## Что это
J.A.R.V.I.S. запускается на вашем ПК, подключается к локальной OpenAI-совместимой LLM (например, LM Studio или vLLM), умеет выполнять системные действия через инструменты, принимать команды из браузера по WebSocket и работать в режиме проактивных уведомлений.

## Ключевые фичи
- Self-modifying code: безопасное чтение и обновление собственного кода (`read_source_code`, `update_source_code`) с проверкой синтаксиса до сохранения.
- MCP Integration: подключение внешних MCP-серверов из `mcp_config.json` и автоматическая маршрутизация tool-calls.
- Fast TTS/STT: быстрый TTS через `edge-tts` и локальная STT-транскрибация через `faster-whisper`.
- Tool Calling: локальные инструменты ОС, web-search, анализ экрана (vision), таймеры и напоминания.
- Proactive Agent: фоновые триггеры (таймеры/папки) с самостоятельной отправкой сообщений и аудио в Web UI.

## Архитектура
- Backend: FastAPI + WebSocket (`src/jarvis_backend/server.py`)
- Agent Core: AsyncOpenAI + tool loop (`src/jarvis_backend/agent.py`)
- Tools: локальные и MCP-инструменты (`src/jarvis_backend/tools.py`, `src/jarvis_backend/mcp_manager.py`)
- Audio: STT/TTS (`src/jarvis_backend/audio.py`)
- UI: `static/index.html`

## Быстрый старт
### 1. Установите `uv`
macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Настройте окружение
```bash
cp .env.example .env
```

Заполните `.env` минимум этими значениями:
- `JARVIS_API_KEY`
- `JARVIS_MODEL`
- `JARVIS_BASE_URL` (обычно `http://localhost:1234/v1`)

### 3. Установите зависимости
```bash
make sync
```

### 4. Запустите сервер разработки
```bash
make dev
```

Откройте в браузере: `http://localhost:8000/`

## MCP настройка
1. Создайте рабочий конфиг:
```bash
cp mcp_config.example.json mcp_config.json
```
2. Замените `<INSERT_YOUR_PATH_HERE>` на ваши реальные пути.
3. Для `sqlite` передавайте путь через флаг `--db-path` (как в примере), иначе сервер MCP не запустится.
4. Перезапустите сервер.

## Полезные команды
- `make sync` — установка/синхронизация зависимостей через `uv`.
- `make dev` — FastAPI + autoreload для веб-интерфейса.
- `make run` — CLI-режим ассистента (без веба).

