from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from base64 import b64decode
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_401_UNAUTHORIZED

from . import tools
from .agent import AgentCore
from .audio import speak, synthesize_speech_base64, transcribe_audio_bytes
from .brain import BrainError
from .config import settings
from .mcp_manager import MCPManager
from .triggers import DirectoryWatcher
from .wake import VoiceListener
from .tools import (
    confirm_with_user,
    execute_tool,
    get_tool_schemas,
    is_local_tool,
    set_confirmation_callback,
    set_proactive_callback,
    set_rules_changed_callback,
)

API_KEY = settings.server_api_key.strip()
TRIGGER_WATCH_DIR = (settings.trigger_watch_dir or "").strip()
CONFIRM_TIMEOUT = settings.confirm_timeout

if not API_KEY:
    raise RuntimeError(
        "JARVIS_SERVER_API_KEY / JARVIS_API_KEY не задан. Укажите переменную окружения (или .env), "
        "иначе сервер не будет запущен по соображениям безопасности."
    )


async def _all_tools() -> list[dict[str, Any]]:
    return get_tool_schemas() + await mcp_manager.get_all_tools()


async def _route_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    if is_local_tool(tool_name):
        result = await execute_tool(tool_name, args)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    return await mcp_manager.execute_tool(tool_name, args)


async def _broadcast_tools() -> None:
    await send_to_ws_safe(None, {"type": "tools_list", "payload": await _all_tools()})


# Глобальный singleton агента (однопользовательский режим).
agent = AgentCore(
    tools_provider=_all_tools,
    tool_executor=_route_tool_call,
    history_path=settings.state_dir / "history.json",
)
# Lock защищает историю сообщений от гонок при параллельных запросах/сокетах.
agent_lock: asyncio.Lock = asyncio.Lock()
mcp_manager = MCPManager(on_tools_changed=_broadcast_tools, confirm=confirm_with_user)
active_websocket: WebSocket | None = None
websocket_send_lock: asyncio.Lock = asyncio.Lock()
directory_watcher: DirectoryWatcher | None = None
voice_listener: VoiceListener | None = None
directory_watcher_task: asyncio.Task[Any] | None = None
# Ожидающие ответа пользователя подтверждения: id -> (future, сокет, которому отправлен запрос).
pending_confirmations: dict[str, tuple[asyncio.Future[bool], WebSocket]] = {}
# Turn'ы агента выполняются фоном, чтобы цикл приёма мог получить ответ на подтверждение.
background_turns: set[asyncio.Task[Any]] = set()



@asynccontextmanager
async def lifespan(_app: FastAPI):
    await on_startup()
    try:
        yield
    finally:
        await on_shutdown()


app = FastAPI(title="J.A.R.V.I.S. Local Server", version="1.1.0", lifespan=lifespan)
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


async def on_startup() -> None:
    print(f"[JARVIS] brain: {agent.brain.label()}")
    set_proactive_callback(trigger_proactive_message)
    set_confirmation_callback(request_confirmation)
    set_rules_changed_callback(agent.reload_system_prompt)
    # MCP поднимается в фоне: npx/uvx могут качать пакеты десятки секунд.
    await mcp_manager.start()

    global voice_listener
    if settings.wake_word:
        voice_listener = VoiceListener(_handle_voice_command, on_wake=_on_voice_wake)
        await voice_listener.start()

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


async def on_shutdown() -> None:
    set_proactive_callback(None)
    set_confirmation_callback(None)
    set_rules_changed_callback(None)
    global voice_listener
    if voice_listener is not None:
        await voice_listener.stop()
        voice_listener = None
    _deny_pending_confirmations(None)
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


async def send_to_ws_safe(websocket: WebSocket | None, payload: dict[str, Any]) -> bool:
    """Send JSON message to the given websocket or active_websocket safely under a lock."""
    global active_websocket
    target_ws = websocket or active_websocket
    if target_ws is None:
        return False
    async with websocket_send_lock:
        try:
            await target_ws.send_json(payload)
            return True
        except Exception:
            if active_websocket is target_ws:
                active_websocket = None
            return False


async def _send_status(websocket: WebSocket, status: str) -> None:
    await send_to_ws_safe(websocket, {"type": "status", "payload": status})


async def _speak_locally(text: str) -> None:
    """Произнести через динамики компьютера (фоновый режим без открытого UI)."""
    try:
        if voice_listener is not None:
            async with voice_listener.muted():
                await speak(text)
        else:
            await speak(text)
    except Exception as exc:  # noqa: BLE001
        print(f"[VOICE] TTS error: {exc}", flush=True)


def _reply_locally() -> bool:
    return active_websocket is None or (voice_listener is not None and settings.voice_reply_local)


async def trigger_proactive_message(message: str) -> None:
    """Проактивное сообщение (напоминание, триггер): в UI и/или голосом из динамиков."""
    text = (message or "").strip()
    if not text:
        return

    await send_to_ws_safe(None, {"type": "speech_text", "payload": text})
    if _reply_locally():
        await _speak_locally(text)
        return

    try:
        audio_b64 = await synthesize_speech_base64(text)
    except Exception:
        audio_b64 = ""

    if audio_b64:
        await send_to_ws_safe(None, {"type": "audio", "payload": audio_b64})


async def request_confirmation(
    tool_name: str,
    args: dict[str, Any],
    details: dict[str, Any],
) -> bool:
    """Ask the connected user to approve a sensitive tool call; deny on timeout/disconnect."""
    target_ws = active_websocket
    if target_ws is None:
        return False

    confirmation_id = secrets.token_urlsafe(12)
    future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    pending_confirmations[confirmation_id] = (future, target_ws)

    # new_content не отправляем целиком: пользователь видит diff из details.
    visible_args = {k: v for k, v in args.items() if k != "new_content"}
    sent = await send_to_ws_safe(target_ws, {
        "type": "confirm_request",
        "payload": {
            "id": confirmation_id,
            "tool": tool_name,
            "args": visible_args,
            "timeout": CONFIRM_TIMEOUT,
            **details,
        },
    })
    try:
        if not sent:
            return False
        return await asyncio.wait_for(future, timeout=CONFIRM_TIMEOUT)
    except asyncio.TimeoutError:
        await send_to_ws_safe(target_ws, {
            "type": "confirm_closed",
            "payload": {"id": confirmation_id, "reason": "timeout"},
        })
        return False
    finally:
        pending_confirmations.pop(confirmation_id, None)


def _resolve_confirmation(websocket: WebSocket, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    entry = pending_confirmations.get(str(payload.get("id", "")))
    if entry is None:
        return
    future, target_ws = entry
    # Ответ принимается только от того сокета, которому был отправлен запрос.
    if target_ws is not websocket or future.done():
        return
    future.set_result(payload.get("approved") is True)


def _deny_pending_confirmations(websocket: WebSocket | None) -> None:
    """Deny confirmations of a closed socket (or all of them when websocket is None)."""
    for future, target_ws in list(pending_confirmations.values()):
        if (websocket is None or target_ws is websocket) and not future.done():
            future.set_result(False)


def _start_background_turn(coro: Any) -> None:
    task = asyncio.create_task(coro, name="jarvis_turn")
    background_turns.add(task)
    task.add_done_callback(background_turns.discard)


async def _handle_trigger_prompt(prompt: str) -> None:
    """Push filesystem trigger prompt through AgentCore and proactively notify user."""
    try:
        async with agent_lock:
            final_speech = await _run_agent_turn(prompt)
    except Exception as exc:  # noqa: BLE001
        await trigger_proactive_message(f"Ошибка триггера: {exc}")
        return

    await trigger_proactive_message(final_speech)


async def _emit_agent_event(kind: str, payload: dict[str, Any]) -> None:
    """tool_call / tool_result -> живая лента инструментов в UI."""
    await send_to_ws_safe(None, {"type": kind, "payload": payload})


async def _run_agent_turn(user_text: str | list[dict[str, Any]]) -> str:
    """Один turn агента с function-calling (local + skills + MCP tools)."""
    try:
        return await agent.run_turn(user_text, on_event=_emit_agent_event)
    except BrainError as exc:
        return str(exc)


def _schedule_restart_if_needed() -> None:
    """После правки ядра перезапускаем процесс (история сохранена на диске).

    Под `uvicorn --reload` процесс к этому моменту уже перезапущен наблюдателем файлов,
    поэтому сюда попадаем только в режиме `make start`.
    """
    if not tools.restart_required:
        return
    tools.restart_required = False

    async def _restart() -> None:
        await asyncio.sleep(1.5)  # даём договорить и отправить аудио
        print("[JARVIS] core code changed, restarting process…", flush=True)
        # orig_argv — команда ровно в том виде, как её запустили (`-m uvicorn`, `uv run uvicorn`…).
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])

    _start_background_turn(_restart())


async def _on_voice_wake() -> None:
    await send_to_ws_safe(None, {"type": "status", "payload": "listening_mic"})


async def _handle_voice_command(command: str) -> None:
    """Команда, сказанная в микрофон компьютера («Джарвис, …»)."""
    await send_to_ws_safe(None, {"type": "text", "payload": command})
    await send_to_ws_safe(None, {"type": "status", "payload": "thinking"})
    try:
        async with agent_lock:
            final_speech = await _run_agent_turn(command)
    except Exception as exc:  # noqa: BLE001
        final_speech = f"Ошибка агента: {exc}"
    await send_to_ws_safe(None, {"type": "speech_text", "payload": final_speech})

    if _reply_locally():
        # Отвечаем из динамиков компьютера; микрофон в это время не слушаем.
        await send_to_ws_safe(None, {"type": "status", "payload": "executing"})
        await _speak_locally(final_speech)
    else:
        try:
            audio_b64 = await synthesize_speech_base64(final_speech)
        except Exception:  # noqa: BLE001
            audio_b64 = ""
        if audio_b64:
            await send_to_ws_safe(None, {"type": "audio", "payload": audio_b64})
    await send_to_ws_safe(None, {"type": "status", "payload": "listening"})
    _schedule_restart_if_needed()


async def _resume_interrupted_turn() -> None:
    """Если процесс перезапустился посреди хода (самоправка ядра), доводим ход до конца."""
    if not agent.has_interrupted_turn():
        return
    try:
        async with agent_lock:
            final_speech = await agent.resume_turn(on_event=_emit_agent_event)
    except Exception as exc:  # noqa: BLE001
        final_speech = f"Не удалось продолжить прерванную задачу: {exc}"
    await trigger_proactive_message(final_speech)


async def _handle_text_message(websocket: WebSocket, user_text: str | list[dict[str, Any]], voice: str) -> None:
    """Единый обработчик текстового запроса (из чата или из STT)."""
    await _send_status(websocket, "thinking")
    skills_before = tools.skills_version

    try:
        async with agent_lock:
            final_speech = await _run_agent_turn(user_text)
        if tools.skills_version != skills_before:
            await _broadcast_tools()
    except Exception as exc:  # noqa: BLE001
        await send_to_ws_safe(websocket, {"type": "error", "payload": f"Agent error: {exc}"})
        await _send_status(websocket, "listening")
        return

    await send_to_ws_safe(websocket, {"type": "speech_text", "payload": final_speech})

    await _send_status(websocket, "executing")
    try:
        audio_b64 = await synthesize_speech_base64(final_speech, voice=voice)
    except Exception as exc:  # noqa: BLE001
        await send_to_ws_safe(websocket, {"type": "error", "payload": f"TTS error: {exc}"})
        audio_b64 = ""

    if audio_b64:
        await send_to_ws_safe(websocket, {"type": "audio", "payload": audio_b64})

    await _send_status(websocket, "listening")
    _schedule_restart_if_needed()


@app.websocket("/ws/jarvis")
async def ws_jarvis(websocket: WebSocket) -> None:
    global active_websocket
    is_authorized = await require_ws_api_key(websocket)
    if not is_authorized:
        return
    await websocket.accept()
    active_websocket = websocket

    # Send tools list to client
    try:
        await send_to_ws_safe(websocket, {"type": "tools_list", "payload": await _all_tools()})
    except Exception:
        pass

    await _send_status(websocket, "listening")
    _start_background_turn(_resume_interrupted_turn())

    client_voice = settings.tts_voice

    while True:
        try:
            message = await websocket.receive_json()
        except Exception:  # noqa: BLE001
            _deny_pending_confirmations(websocket)
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass
            if active_websocket is websocket:
                active_websocket = None
            return

        msg_type = message.get("type")
        payload = message.get("payload")

        if msg_type == "confirm_response":
            _resolve_confirmation(websocket, payload)
            continue

        if msg_type == "set_voice":
            client_voice = str(payload).strip()
            continue

        if msg_type == "reset_history":
            async with agent_lock:
                agent.reset()
            await send_to_ws_safe(websocket, {"type": "speech_text", "payload": "История диалога очищена."})
            continue

        if msg_type == "get_mcp_config":
            try:
                config_path = mcp_manager.config_path
                if config_path.exists():
                    cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
                else:
                    cfg_data = {"mcpServers": {}}
                await send_to_ws_safe(websocket, {"type": "mcp_config", "payload": cfg_data})
            except Exception as e:
                await send_to_ws_safe(websocket, {"type": "error", "payload": f"Failed to get MCP config: {e}"})
            continue

        if msg_type == "update_mcp_config":
            try:
                mcp_manager.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                await mcp_manager.restart()
                await send_to_ws_safe(websocket, {"type": "tools_list", "payload": await _all_tools()})
                await send_to_ws_safe(websocket, {"type": "status", "payload": "listening"})
            except Exception as e:
                await send_to_ws_safe(websocket, {"type": "error", "payload": f"Failed to update MCP config: {e}"})
            continue

        if msg_type == "image":
            try:
                image_b64 = payload.get("image_b64", "")
                text_prompt = payload.get("text", "").strip() or "Analyze this image."
                if not image_b64:
                    raise ValueError("Empty image_b64 payload")

                user_content = [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
                await send_to_ws_safe(websocket, {"type": "text", "payload": text_prompt})
                _start_background_turn(
                    _handle_text_message(websocket, user_content, voice=client_voice)
                )
            except Exception as e:
                await send_to_ws_safe(websocket, {"type": "error", "payload": f"Failed to process image: {e}"})
            continue

        if msg_type == "text":
            user_text = (payload or "").strip()
            if not user_text:
                await send_to_ws_safe(websocket, {"type": "error", "payload": "Empty text payload"})
                await _send_status(websocket, "listening")
                continue

            _start_background_turn(_handle_text_message(websocket, user_text, voice=client_voice))
            continue

        if msg_type == "audio":
            await _send_status(websocket, "listening_mic")

            try:
                if isinstance(payload, str):
                    audio_bytes = b64decode(payload, validate=True)
                else:
                    raise ValueError("Payload must be base64 string")
            except Exception:  # noqa: BLE001
                await send_to_ws_safe(websocket, {"type": "error", "payload": "Invalid base64 audio payload"})
                await _send_status(websocket, "listening")
                continue

            try:
                recognized_text = (await transcribe_audio_bytes(audio_bytes)).strip()
            except Exception as exc:  # noqa: BLE001
                await send_to_ws_safe(websocket, {"type": "error", "payload": f"STT error: {exc}"})
                await _send_status(websocket, "listening")
                continue

            if not recognized_text:
                await send_to_ws_safe(websocket, {"type": "error", "payload": "Не удалось распознать речь"})
                await _send_status(websocket, "listening")
                continue

            await send_to_ws_safe(websocket, {"type": "text", "payload": recognized_text})
            _start_background_turn(
                _handle_text_message(websocket, recognized_text, voice=client_voice)
            )
            continue

        await send_to_ws_safe(websocket, {
            "type": "error",
            "payload": "Unsupported message type. Use 'text' or 'audio'.",
        })
        await _send_status(websocket, "listening")
