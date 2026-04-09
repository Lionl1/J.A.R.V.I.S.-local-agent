from __future__ import annotations

import asyncio
import json
import secrets
from base64 import b64decode
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_401_UNAUTHORIZED

from .agent import AgentCore
from .audio import synthesize_speech_base64, transcribe_audio_bytes
from .config import settings
from .mcp_manager import MCPManager
from .triggers import DirectoryWatcher
from .tools import LOCAL_TOOL_NAMES, TOOLS_SCHEMAS, execute_tool, set_proactive_callback

API_KEY = settings.api_key.strip()
MODEL_NAME = settings.model
BASE_URL = settings.base_url
MAX_TOOL_STEPS = settings.max_tool_steps
TRIGGER_WATCH_DIR = (settings.trigger_watch_dir or "").strip()

if not API_KEY:
    raise RuntimeError(
        "JARVIS_API_KEY не задан. Укажите переменную окружения (или .env), "
        "иначе сервер не будет запущен по соображениям безопасности."
    )


# Глобальный singleton агента (однопользовательский режим).
agent = AgentCore(model=MODEL_NAME, base_url=BASE_URL)
# Lock защищает историю сообщений от гонок при параллельных запросах/сокетах.
agent_lock: asyncio.Lock = asyncio.Lock()
mcp_manager = MCPManager()
active_websocket: WebSocket | None = None
websocket_send_lock: asyncio.Lock = asyncio.Lock()
directory_watcher: DirectoryWatcher | None = None
directory_watcher_task: asyncio.Task[Any] | None = None

app = FastAPI(title="J.A.R.V.I.S. Local Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Раздача фронтенда.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _extract_bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _validate_token(token: str | None) -> bool:
    if not token:
        return False
    return secrets.compare_digest(token, API_KEY)


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """HTTP dependency для защиты REST-роутов."""
    candidate = x_api_key or _extract_bearer_token(authorization) or token
    if not _validate_token(candidate):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


async def require_ws_api_key(websocket: WebSocket) -> bool:
    """Проверка ключа для WebSocket.

    Если токен невалидный, соединение закрывается с кодом 4003.
    """
    token = (
        websocket.headers.get("x-api-key")
        or _extract_bearer_token(websocket.headers.get("authorization"))
        or websocket.query_params.get("token")
    )
    if not _validate_token(token):
        await websocket.accept()
        await websocket.close(code=4003, reason="Invalid API key")
        return False
    return True


@app.get("/", response_model=None)
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        {
            "message": "Frontend not found. Place static/index.html.",
            "static_dir": str(STATIC_DIR),
        }
    )


@app.get("/health", dependencies=[Depends(require_api_key)])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    set_proactive_callback(trigger_proactive_message)
    await mcp_manager.start()

    global directory_watcher, directory_watcher_task
    if TRIGGER_WATCH_DIR:
        directory_watcher = DirectoryWatcher(
            TRIGGER_WATCH_DIR,
            _handle_trigger_prompt,
        )
        directory_watcher_task = asyncio.create_task(
            directory_watcher.run(),
            name="jarvis_directory_watcher",
        )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    set_proactive_callback(None)
    global directory_watcher, directory_watcher_task
    if directory_watcher is not None:
        await directory_watcher.stop()
    if directory_watcher_task is not None:
        try:
            await directory_watcher_task
        except Exception:  # noqa: BLE001
            pass
    directory_watcher = None
    directory_watcher_task = None
    await mcp_manager.close()


async def _send_status(websocket: WebSocket, status: str) -> None:
    await websocket.send_json({"type": "status", "payload": status})


async def trigger_proactive_message(message: str) -> None:
    """Send proactive speech_text/audio to currently active websocket."""
    global active_websocket

    text = (message or "").strip()
    if not text or active_websocket is None:
        return

    ws = active_websocket
    try:
        async with websocket_send_lock:
            await ws.send_json({"type": "speech_text", "payload": text})
            try:
                audio_b64 = await synthesize_speech_base64(text)
            except Exception:
                audio_b64 = ""
            if audio_b64:
                await ws.send_json({"type": "audio", "payload": audio_b64})
    except Exception:
        # Соединение уже закрыто/ошибка отправки — silently ignore.
        if active_websocket is ws:
            active_websocket = None


async def _handle_trigger_prompt(prompt: str) -> None:
    """Push filesystem trigger prompt through AgentCore and proactively notify user."""
    try:
        async with agent_lock:
            final_speech = await _run_agent_turn(prompt)
    except Exception as exc:  # noqa: BLE001
        await trigger_proactive_message(f"Ошибка триггера: {exc}")
        return

    await trigger_proactive_message(final_speech)


def _parse_tool_arguments(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        payload = json.loads(arguments)
        if isinstance(payload, dict):
            return payload
    except Exception:  # noqa: BLE001
        return {}
    return {}


async def _route_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name in LOCAL_TOOL_NAMES:
        result = await execute_tool(tool_name, args)
        return json.dumps(result, ensure_ascii=False)
    return await mcp_manager.execute_tool(tool_name, args)


async def _run_agent_turn(user_text: str) -> str:
    """Один turn агента с OpenAI function-calling (local + MCP tools)."""
    agent.messages.append({"role": "user", "content": user_text})

    for _step in range(1, MAX_TOOL_STEPS + 1):
        all_tools = TOOLS_SCHEMAS + await mcp_manager.get_all_tools()
        llm_message = await agent._ask_llm(
            agent.messages,
            tools=all_tools,
            return_message=True,
        )
        content = str(llm_message.get("content", "")).strip()
        tool_calls = llm_message.get("tool_calls", []) or []

        if not tool_calls:
            # Совместимость: если модель по-прежнему вернула JSON с speech/action.
            decision = agent._parse_decision(content)
            final_speech = decision.speech or content or "Готово."
            agent.messages.append({"role": "assistant", "content": content or final_speech})
            return final_speech

        assistant_tool_call_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        agent.messages.append(assistant_tool_call_message)

        for tool_call in tool_calls:
            function_payload = tool_call.get("function", {})
            tool_name = str(function_payload.get("name", "")).strip()
            tool_args = _parse_tool_arguments(function_payload.get("arguments"))
            tool_call_id = str(tool_call.get("id", "")).strip() or f"call_{_step}"

            observation_text = await _route_tool_call(tool_name, tool_args)
            agent.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": observation_text,
                }
            )

    fallback = "Превышен лимит шагов инструментов. Уточни задачу."
    agent.messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "thought": "Stopped due to MAX_TOOL_STEPS",
                    "action": None,
                    "speech": fallback,
                },
                ensure_ascii=False,
            ),
        }
    )
    return fallback


async def _handle_text_message(websocket: WebSocket, user_text: str) -> None:
    """Единый обработчик текстового запроса (из чата или из STT)."""
    await _send_status(websocket, "thinking")

    try:
        async with agent_lock:
            final_speech = await _run_agent_turn(user_text)
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json(
            {"type": "error", "payload": f"Agent error: {exc}"}
        )
        await _send_status(websocket, "listening")
        return

    await websocket.send_json({"type": "speech_text", "payload": final_speech})

    await _send_status(websocket, "executing")
    try:
        audio_b64 = await synthesize_speech_base64(final_speech)
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"type": "error", "payload": f"TTS error: {exc}"})
        await _send_status(websocket, "listening")
        return

    if audio_b64:
        await websocket.send_json({"type": "audio", "payload": audio_b64})

    await _send_status(websocket, "listening")


@app.websocket("/ws/jarvis")
async def ws_jarvis(websocket: WebSocket) -> None:
    global active_websocket
    is_authorized = await require_ws_api_key(websocket)
    if not is_authorized:
        return
    await websocket.accept()
    active_websocket = websocket

    await _send_status(websocket, "listening")

    while True:
        try:
            message = await websocket.receive_json()
        except Exception:  # noqa: BLE001
            await websocket.close()
            if active_websocket is websocket:
                active_websocket = None
            return

        msg_type = message.get("type")
        payload = message.get("payload")

        if msg_type == "text":
            user_text = (payload or "").strip()
            if not user_text:
                await websocket.send_json(
                    {"type": "error", "payload": "Empty text payload"}
                )
                await _send_status(websocket, "listening")
                continue

            await _handle_text_message(websocket, user_text)
            continue

        if msg_type == "audio":
            await _send_status(websocket, "listening_mic")

            try:
                if isinstance(payload, str):
                    audio_bytes = b64decode(payload, validate=True)
                else:
                    raise ValueError("Payload must be base64 string")
            except Exception:  # noqa: BLE001
                await websocket.send_json(
                    {"type": "error", "payload": "Invalid base64 audio payload"}
                )
                await _send_status(websocket, "listening")
                continue

            try:
                recognized_text = (await transcribe_audio_bytes(audio_bytes)).strip()
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    {"type": "error", "payload": f"STT error: {exc}"}
                )
                await _send_status(websocket, "listening")
                continue

            if not recognized_text:
                await websocket.send_json(
                    {"type": "error", "payload": "Не удалось распознать речь"}
                )
                await _send_status(websocket, "listening")
                continue

            await websocket.send_json({"type": "text", "payload": recognized_text})
            await _handle_text_message(websocket, recognized_text)
            continue

        await websocket.send_json(
            {
                "type": "error",
                "payload": "Unsupported message type. Use 'text' or 'audio'.",
            }
        )
        await _send_status(websocket, "listening")
