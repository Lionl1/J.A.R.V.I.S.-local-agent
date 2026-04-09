from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .config import settings

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # noqa: BLE001
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]


@dataclass(slots=True)
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


class MCPManager:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else settings.mcp_config_path
        self._configs: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, Any] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._started = False

    async def start(self) -> None:
        """Start all MCP stdio sessions from mcp_config.json."""
        if self._started:
            return

        self._started = True
        self._configs = self._load_configs()

        if not self._configs or stdio_client is None:
            return

        self._exit_stack = AsyncExitStack()

        for server_name, cfg in self._configs.items():
            try:
                params = StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args,
                    env=cfg.env or None,
                )
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                self._sessions[server_name] = session
            except Exception:
                # Не падаем при проблеме одного сервера.
                continue

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._sessions.clear()
        self._tool_map.clear()
        self._started = False

    async def get_all_tools(self) -> list[dict[str, Any]]:
        """Return all MCP tools translated to OpenAI function-calling schema."""
        tools: list[dict[str, Any]] = []
        self._tool_map.clear()

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
                self._tool_map[prefixed_name] = (server_name, original_name)

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

        return tools

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
