# J.A.R.V.I.S. Local AI Agent

A fully local, asynchronous, proactive voice assistant with OS integration, web-based terminal interface, and Model Context Protocol (MCP) support.

## What is this?
J.A.R.V.I.S. runs on your local machine, connects to a local OpenAI-compatible LLM endpoint (like LM Studio or vLLM), performs system operations through local tools, receives commands from your browser via WebSocket, and supports proactive background triggers.

## Key Features
- **Self-Modifying Code**: Safe reading and modification of its own codebase (`read_source_code`, `update_source_code`) with Python syntax validation before saving.
- **MCP Integration**: Connect external MCP servers using `mcp_config.json` and automatically route tool calls.
- **Fast Local TTS/STT**: Fast speech synthesis via `edge-tts` and high-speed local audio transcription via `faster-whisper`.
- **Flexible Tool Calling**: Desktop application control, web search, visual screen analysis (vision model), timers, and reminders.
- **Proactive Agent Actions**: Filesystem watchers and timers running in the background with proactive status, speech, and audio broadcasts directly to the Web UI.

## Architecture
- **Backend**: FastAPI + WebSocket (`src/jarvis_backend/server.py`)
- **Agent Core**: AsyncOpenAI + tool loop orchestration (`src/jarvis_backend/agent.py`)
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
- `JARVIS_API_KEY`
- `JARVIS_MODEL`
- `JARVIS_BASE_URL` (typically `http://localhost:1234/v1` for LM Studio)

### 3. Install Dependencies
```bash
make sync
```

### 4. Run Development Server
```bash
make dev
```

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
- `make dev` — Run the FastAPI server with autoreload enabled for web development.
- `make run` — Launch J.A.R.V.I.S. in CLI console mode (without the Web UI).
