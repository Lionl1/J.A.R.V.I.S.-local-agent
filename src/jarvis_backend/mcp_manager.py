from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from .config import settings

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # noqa: BLE001
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]


# Фоновый агент (launchd, systemd, IDE) часто стартует с урезанным PATH,
# где нет npx/uvx — ищем их и в стандартных местах установки.
_EXTRA_BIN_DIRS = (
    Path.home() / ".local" / "bin",
    Path.home() / ".cargo" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


def _resolve_command(command: str) -> str:
    if os.sep in command or shutil.which(command):
        return command
    for directory in _EXTRA_BIN_DIRS:
        candidate = directory / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(
        f"команда '{command}' не найдена ни в PATH, ни в {', '.join(map(str, _EXTRA_BIN_DIRS))}"
    )


@dataclass(slots=True)
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


class MCPManager:
    """MCP-серверы живут в одной фоновой задаче-владельце.

    stdio-контексты MCP (anyio) нужно открывать и закрывать в одной и той же задаче,
    поэтому start() только запускает задачу, а close() просит её завершиться.
    Сервер Джарвиса стартует сразу, не дожидаясь npx/uvx.
    """

    INIT_TIMEOUT = 60.0

    def __init__(
        self,
        config_path: str | Path | None = None,
        on_tools_changed: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.config_path = Path(config_path) if config_path else settings.mcp_config_path
        self.on_tools_changed = on_tools_changed
        self._configs: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, Any] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}
        self._tools_cache: list[dict[str, Any]] = []
        self.errors: dict[str, str] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._ready: asyncio.Event | None = None

    async def start(self, wait: bool = False) -> None:
        """Start all MCP stdio sessions from mcp_config.json in the background."""
        if self._task is not None:
            return
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="jarvis_mcp")
        if wait:
            await self._ready.wait()

    async def wait_ready(self) -> None:
        if self._ready is not None:
            await self._ready.wait()

    async def _run(self) -> None:
        assert self._stop is not None and self._ready is not None
        try:
            self._configs = self._load_configs()
            if not self._configs or stdio_client is None:
                return
            async with AsyncExitStack() as stack:
                for server_name, cfg in self._configs.items():
                    session = await self._connect(stack, server_name, cfg)
                    if session is not None:
                        self._sessions[server_name] = session
                await self._refresh_tools()
                self._ready.set()
                if self._sessions:
                    print(f"[MCP] ready: {', '.join(self._sessions)} ({len(self._tools_cache)} tools)", flush=True)
                if self.on_tools_changed is not None:
                    try:
                        await self.on_tools_changed()
                    except Exception:  # noqa: BLE001
                        pass
                await self._stop.wait()
        except Exception as exc:  # noqa: BLE001
            print(f"[MCP] manager crashed: {exc}", flush=True)
        finally:
            self._sessions.clear()
            self._tool_map.clear()
            self._tools_cache = []
            self._ready.set()

    async def _connect(self, stack: AsyncExitStack, server_name: str, cfg: MCPServerConfig) -> Any:
        """Подключить один сервер; его падение не мешает остальным."""
        server_stack = AsyncExitStack()
        try:
            command = _resolve_command(cfg.command)
            env = dict(cfg.env)
            bin_dir = os.path.dirname(command)
            if bin_dir:
                # npx/uvx сами запускают node/python через PATH — добавляем их папку.
                env["PATH"] = os.pathsep.join([bin_dir, env.get("PATH") or os.environ.get("PATH", "")])
            params = StdioServerParameters(command=command, args=cfg.args, env=env or None)
            read_stream, write_stream = await server_stack.enter_async_context(stdio_client(params))
            session = await server_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await asyncio.wait_for(session.initialize(), timeout=self.INIT_TIMEOUT)
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, asyncio.CancelledError):
                await server_stack.aclose()
                raise
            self.errors[server_name] = f"{type(exc).__name__}: {exc}"
            print(f"[MCP] server '{server_name}' failed to start: {self.errors[server_name]}", flush=True)
            try:
                await server_stack.aclose()
            except BaseException:  # noqa: BLE001
                pass
            return None
        stack.push_async_callback(server_stack.aclose)
        return session

    async def close(self) -> None:
        if self._task is None:
            return
        if self._stop is not None:
            self._stop.set()
        try:
            await self._task
        except BaseException:  # noqa: BLE001
            pass
        self._task = None

    async def restart(self) -> None:
        await self.close()
        self.errors.clear()
        await self.start(wait=True)

    async def _refresh_tools(self) -> None:
        tools: list[dict[str, Any]] = []
        tool_map: dict[str, tuple[str, str]] = {}

        for server_name, session in self._sessions.items():
            try:
                result = await session.list_tools()
                mcp_tools = getattr(result, "tools", []) or []
            except Exception:
                continue

            for tool in mcp_tools:
                original_name = str(getattr(tool, "name", "")).strip()
                if not original_name:
                    continue

                prefixed_name = f"{server_name}__{original_name}"
                tool_map[prefixed_name] = (server_name, original_name)

                parameters = getattr(tool, "inputSchema", None) or {
                    "type": "object",
                    "properties": {},
                }
                if not isinstance(parameters, dict):
                    parameters = {"type": "object", "properties": {}}

                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": prefixed_name,
                            "description": str(getattr(tool, "description", ""))[:1024],
                            "parameters": parameters,
                        },
                    }
                )

        self._tool_map = tool_map
        self._tools_cache = tools

    async def get_all_tools(self) -> list[dict[str, Any]]:
        """Return all MCP tools translated to OpenAI function-calling schema (cached)."""
        return list(self._tools_cache)

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute MCP tool by prefixed name: <server>__<tool>."""
        mapping = self._tool_map.get(tool_name)
        if not mapping:
            return f"Ошибка при обращении к серверу: неизвестный MCP инструмент '{tool_name}'"

        server_name, original_name = mapping
        session = self._sessions.get(server_name)
        if session is None:
            return f"Ошибка при обращении к серверу: MCP сервер '{server_name}' неактивен"

        try:
            result = await session.call_tool(original_name, arguments=args or {})
            return self._mcp_result_to_text(result)
        except Exception as exc:  # noqa: BLE001
            return f"Ошибка при обращении к серверу: {exc}"

    def _load_configs(self) -> dict[str, MCPServerConfig]:
        if not self.config_path.exists():
            return {}

        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        raw_servers = payload.get("mcpServers", {})
        if not isinstance(raw_servers, dict):
            return {}

        configs: dict[str, MCPServerConfig] = {}
        for server_name, raw_cfg in raw_servers.items():
            if not isinstance(raw_cfg, dict):
                continue

            command = str(raw_cfg.get("command", "")).strip()
            if not command:
                continue

            args = raw_cfg.get("args", [])
            env = raw_cfg.get("env", {})

            if not isinstance(args, list):
                args = []
            if not isinstance(env, dict):
                env = {}

            configs[str(server_name)] = MCPServerConfig(
                command=command,
                args=[str(x) for x in args],
                env={str(k): str(v) for k, v in env.items()},
            )

        return configs

    def _mcp_result_to_text(self, result: Any) -> str:
        if result is None:
            return ""

        content = getattr(result, "content", None)
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text is not None:
                    chunks.append(str(text))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks).strip()

        if hasattr(result, "model_dump"):
            try:
                dumped = result.model_dump()
                return json.dumps(dumped, ensure_ascii=False)
            except Exception:
                pass

        return str(result)
