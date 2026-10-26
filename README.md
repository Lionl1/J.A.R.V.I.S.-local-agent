# J.A.R.V.I.S. Local AI Agent

A fully local, asynchronous, proactive voice assistant with OS integration, web-based terminal interface, and Model Context Protocol (MCP) support.

## What is this?
J.A.R.V.I.S. is a background agent for your computer: voice and web interface, system control through tools,
and a pluggable brain — any LLM. Its key feature is recursive self-improvement (RSI): Jarvis reads and edits
its own code, rules and UI, verifies every change in a sandbox, asks you to approve the diff, commits it to git
and picks it up immediately.

## Pluggable brain
Set `JARVIS_PROVIDER` in `.env`:

| Provider | Default endpoint | Key |
|---|---|---|
| `lmstudio` (default) | `http://localhost:1234/v1` | — |
| `ollama` | `http://localhost:11434/v1` | — |
| `vllm` | `http://localhost:8000/v1` | — |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `groq`, `deepseek`, `gemini` | their OpenAI-compatible endpoints | `GROQ_API_KEY` / `DEEPSEEK_API_KEY` / `GEMINI_API_KEY` |
| `anthropic` | native Claude API (`claude-opus-5-5` by default) | `ANTHROPIC_API_KEY` |

`JARVIS_MODEL`, `JARVIS_BASE_URL` and `JARVIS_API_KEY` override the preset. The conversation history is stored in one
canonical format, so the brain can be swapped without losing context. With `anthropic`, server-side refusal
fallbacks (`fallbacks: "default"`) are enabled for models that support them.

## Self-improvement (RSI) loop
1. Jarvis reads its files (`list_source_files`, `read_source_code`).
2. It changes them with `edit_source_code` (exact fragment replacement) or `update_source_code` (new/whole file).
3. The change is applied to a temporary copy of the project: compile → import every module and skill → run tests.
   If anything fails, Jarvis gets the error output and fixes the code itself.
4. Only a verified change is shown to you as a diff; after "Allow" it is committed to git by `J.A.R.V.I.S.`.
5. It is picked up immediately: rules and skills reload in-process; a core change restarts the process
   (`make start`/`make run`) while the conversation history (`.jarvis/history.json`) survives.
6. Undo: `rollback_source_change` or `make rollback`.

New abilities go into `src/jarvis_backend/skills/<name>.py` (see `skills/__init__.py` and `skills/system_info.py`).

## Key Features
- **Self-Modifying Code**: see the RSI loop above.
- **System control**: `run_shell_command`, `open_application`, screen analysis, web search, reminders, MCP servers and skills.
- **Confirmation for sensitive tools**: `edit_source_code`, `update_source_code`, `rollback_source_change`, `run_shell_command` and `open_application` run only after the user clicks "Allow" in the web UI (or answers `y` in the console). No answer within `JARVIS_CONFIRM_TIMEOUT` seconds (default 120) or a disconnect means "deny".
- **MCP Integration**: Connect external MCP servers using `mcp_config.json` and automatically route tool calls.
- **Fast Local TTS/STT**: Fast speech synthesis via `edge-tts` and high-speed local audio transcription via `faster-whisper`.
- **Flexible Tool Calling**: Desktop application control, web search, visual screen analysis (vision model), timers, and reminders.
- **Proactive Agent Actions**: Filesystem watchers and timers running in the background with proactive status, speech, and audio broadcasts directly to the Web UI.

## Architecture
- **Backend**: FastAPI + WebSocket (`src/jarvis_backend/server.py`)
- **Brain**: pluggable LLM adapters (`src/jarvis_backend/brain.py`)
- **Agent Core**: history + tool loop shared by web and console modes (`src/jarvis_backend/agent.py`)
- **RSI**: sandbox verification (`src/jarvis_backend/verifier.py`), git snapshots (`src/jarvis_backend/git_guard.py`), skills (`src/jarvis_backend/skills/`)
- **Tools**: Local utilities and MCP server orchestration (`src/jarvis_backend/tools.py`, `src/jarvis_backend/mcp_manager.py`)
- **Audio**: Audio file playback, transcription, and TTS (`src/jarvis_backend/audio.py`)
- **UI**: Premium glassmorphic web console with dynamic Siri-like state orb and Canvas audio visualizer (`static/index.html`)

## Getting Started

### 1. Install `uv`
macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` and configure at least the following:
- `JARVIS_PROVIDER` (and the provider's API key, if it is a cloud one)
- `JARVIS_SERVER_API_KEY` — the token you enter in the web UI

### 3. Install Dependencies
```bash
make sync
```

### 4. Run
```bash
make start
```
(`make dev` restarts on every file change — for hacking on Jarvis by hand.)

Open in your browser: `http://localhost:8000/`

## MCP Configuration
1. Create a working config file:
```bash
cp mcp_config.example.json mcp_config.json
```
2. Replace `<INSERT_YOUR_PATH_HERE>` with your actual directories.
3. For the `sqlite` server, pass the database file path via the `--db-path` argument, otherwise the server will fail to start.
4. Restart your FastAPI server.

## Useful Commands
- `make sync` — Install or synchronize dependencies using `uv`.
- `make start` — Run Jarvis as a background agent (restarts itself after a core self-edit).
- `make dev` — Run the FastAPI server with autoreload enabled for web development.
- `make run` — Launch J.A.R.V.I.S. in CLI console mode (without the Web UI).
- `make rollback` — Revert the last agent-made code change (works even if the server no longer starts).
- `make test` — Run the test suite.
