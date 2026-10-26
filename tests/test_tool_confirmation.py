from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from .conftest import git

NEW_CODE = "VALUE = 2\n"


def _make_callback(answer: bool, calls: list[tuple[str, dict[str, Any], dict[str, Any]]]):
    async def _callback(tool_name: str, args: dict[str, Any], details: dict[str, Any]) -> bool:
        calls.append((tool_name, args, details))
        return answer

    return _callback


@pytest.mark.asyncio
async def test_sensitive_tool_denied_without_callback(tools_in_repo, repo: Path) -> None:
    result = await tools_in_repo.execute_tool(
        "update_source_code", {"file_name": "tools.py", "new_content": NEW_CODE}
    )

    assert result["ok"] is False
    assert "не подтвердил" in result["error"]
    assert (repo / "src/jarvis_backend/tools.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git(repo, "log", "-1", "--format=%s") == "initial"


@pytest.mark.asyncio
async def test_update_source_code_denied_by_user(tools_in_repo, repo: Path) -> None:
    calls: list = []
    tools_in_repo.set_confirmation_callback(_make_callback(False, calls))

    result = await tools_in_repo.execute_tool(
        "update_source_code", {"file_name": "tools.py", "new_content": NEW_CODE}
    )

    assert result["ok"] is False
    assert len(calls) == 1
    assert (repo / "src/jarvis_backend/tools.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git(repo, "log", "-1", "--format=%s") == "initial"


@pytest.mark.asyncio
async def test_update_source_code_approved_commits_and_writes(tools_in_repo, repo: Path) -> None:
    calls: list = []
    tools_in_repo.set_confirmation_callback(_make_callback(True, calls))

    result = await tools_in_repo.execute_tool(
        "update_source_code", {"file_name": "tools.py", "new_content": NEW_CODE}
    )

    assert result["ok"] is True
    tool_name, _args, details = calls[0]
    assert tool_name == "update_source_code"
    assert "src/jarvis_backend/tools.py" in details["summary"]
    assert "-VALUE = 1" in details["diff"] and "+VALUE = 2" in details["diff"]

    assert (repo / "src/jarvis_backend/tools.py").read_text(encoding="utf-8") == NEW_CODE
    assert git(repo, "rev-parse", "HEAD") == result["commit"]
    assert git(repo, "status", "--porcelain") == ""


@pytest.mark.asyncio
async def test_invalid_update_is_rejected_before_asking(tools_in_repo, repo: Path) -> None:
    calls: list = []
    tools_in_repo.set_confirmation_callback(_make_callback(True, calls))

    syntax_error = await tools_in_repo.execute_tool(
        "update_source_code", {"file_name": "tools.py", "new_content": "def broken(:\n"}
    )
    outside = await tools_in_repo.execute_tool(
        "update_source_code", {"file_name": "../escape.py", "new_content": NEW_CODE}
    )
    (repo / "src/jarvis_backend/tools.py").write_text("VALUE = 99\n", encoding="utf-8")
    dirty = await tools_in_repo.execute_tool(
        "update_source_code", {"file_name": "tools.py", "new_content": NEW_CODE}
    )

    assert syntax_error["ok"] is False and "синтаксиса" in syntax_error["error"]
    assert outside["ok"] is False
    assert dirty["ok"] is False and "незакоммиченные" in dirty["error"]
    assert calls == []
    # Правка пользователя не потеряна.
    assert (repo / "src/jarvis_backend/tools.py").read_text(encoding="utf-8") == "VALUE = 99\n"


@pytest.mark.asyncio
async def test_open_application_requires_confirmation(tools_in_repo, monkeypatch) -> None:
    launched: list[str] = []

    async def fake_open(app_name: str) -> dict[str, Any]:
        launched.append(app_name)
        return {"ok": True}

    monkeypatch.setitem(tools_in_repo.TOOL_REGISTRY, "open_application", fake_open)
    calls: list = []
    tools_in_repo.set_confirmation_callback(_make_callback(False, calls))

    denied = await tools_in_repo.execute_tool("open_application", {"app_name": "Terminal"})
    assert denied["ok"] is False
    assert launched == []
    assert calls[0][2]["summary"] == "Запустить приложение: Terminal"

    tools_in_repo.set_confirmation_callback(_make_callback(True, calls))
    approved = await tools_in_repo.execute_tool("open_application", {"app_name": "Terminal"})
    assert approved["ok"] is True
    assert launched == ["Terminal"]


@pytest.mark.asyncio
async def test_rollback_tool_reverts_agent_change(tools_in_repo, repo: Path) -> None:
    calls: list = []
    tools_in_repo.set_confirmation_callback(_make_callback(True, calls))
    await tools_in_repo.execute_tool(
        "update_source_code", {"file_name": "tools.py", "new_content": NEW_CODE}
    )

    result = await tools_in_repo.execute_tool("rollback_source_change", {})

    assert result["ok"] is True
    assert calls[-1][2]["files"] == ["src/jarvis_backend/tools.py"]
    assert (repo / "src/jarvis_backend/tools.py").read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.asyncio
async def test_safe_tools_do_not_ask(tools_in_repo) -> None:
    calls: list = []
    tools_in_repo.set_confirmation_callback(_make_callback(False, calls))

    result = await tools_in_repo.execute_tool("get_current_time", {})

    assert result["ok"] is True
    assert calls == []
