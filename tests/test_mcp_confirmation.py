from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jarvis_backend.mcp_manager import DENIED_TEXT, MCPManager


class FakeSession:
    """One MCP server: a tool that writes, and a tool the server declares read-only."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(name="write_file", description="", inputSchema=None, annotations=None),
                SimpleNamespace(
                    name="read_file",
                    description="",
                    inputSchema=None,
                    annotations=SimpleNamespace(readOnlyHint=True),
                ),
                SimpleNamespace(
                    name="list_dir",
                    description="",
                    inputSchema=None,
                    annotations=SimpleNamespace(read_only_hint=True),  # mcp 2.x spelling
                ),
                SimpleNamespace(
                    name="delete_file",
                    description="",
                    inputSchema=None,
                    annotations=SimpleNamespace(readOnlyHint=False, destructiveHint=True),
                ),
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        self.calls.append((name, arguments))
        return SimpleNamespace(content=[SimpleNamespace(text="done")], isError=False)


async def _manager(confirm=None, auto_approve: tuple[str, ...] = ()) -> tuple[MCPManager, FakeSession]:
    manager = MCPManager(config_path="unused.json", confirm=confirm, auto_approve=auto_approve)
    session = FakeSession()
    manager._sessions = {"fs": session}
    await manager._refresh_tools()
    return manager, session


def _answer(value: bool, asked: list[tuple[str, dict[str, Any], dict[str, Any]]]):
    async def _confirm(tool_name: str, args: dict[str, Any], details: dict[str, Any]) -> bool:
        asked.append((tool_name, args, details))
        return value

    return _confirm


@pytest.mark.asyncio
async def test_mcp_tool_denied_without_a_confirmation_handler() -> None:
    manager, session = await _manager()

    result = await manager.execute_tool("fs__write_file", {"path": "a.txt", "content": "x"})

    assert result == DENIED_TEXT
    assert session.calls == []


@pytest.mark.asyncio
async def test_mcp_tool_denied_by_user_is_not_called() -> None:
    asked: list = []
    manager, session = await _manager(confirm=_answer(False, asked))

    result = await manager.execute_tool("fs__delete_file", {"path": "a.txt"})

    assert result == DENIED_TEXT
    assert session.calls == []
    assert asked[0][0] == "fs__delete_file"
    assert '"path": "a.txt"' in asked[0][2]["diff"]


@pytest.mark.asyncio
async def test_mcp_tool_runs_after_approval() -> None:
    asked: list = []
    manager, session = await _manager(confirm=_answer(True, asked))

    await manager.execute_tool("fs__write_file", {"path": "a.txt", "content": "x"})

    assert len(asked) == 1
    assert session.calls == [("write_file", {"path": "a.txt", "content": "x"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["fs__read_file", "fs__list_dir"])
async def test_read_only_tools_run_without_asking(tool: str) -> None:
    asked: list = []
    manager, session = await _manager(confirm=_answer(False, asked))

    await manager.execute_tool(tool, {"path": "."})

    assert asked == []
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_auto_approve_patterns() -> None:
    asked: list = []
    manager, session = await _manager(
        confirm=_answer(False, asked), auto_approve=("fs__write_*", "other__*")
    )

    await manager.execute_tool("fs__write_file", {"path": "a.txt", "content": "x"})
    denied = await manager.execute_tool("fs__delete_file", {"path": "a.txt"})

    assert session.calls == [("write_file", {"path": "a.txt", "content": "x"})]
    assert denied == DENIED_TEXT
    assert [call[0] for call in asked] == ["fs__delete_file"]
