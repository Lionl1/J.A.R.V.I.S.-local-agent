from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis_backend import verifier

from .conftest import git

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _approve(calls: list):
    async def _callback(tool_name: str, args: dict[str, Any], details: dict[str, Any]) -> bool:
        calls.append((tool_name, args, details))
        return True

    return _callback


@pytest.mark.asyncio
async def test_edit_source_code_replaces_unique_fragment(tools_in_repo, repo: Path) -> None:
    calls: list = []
    tools_in_repo.set_confirmation_callback(_approve(calls))

    result = await tools_in_repo.execute_tool(
        "edit_source_code", {"file_name": "tools.py", "old_text": "VALUE = 1", "new_text": "VALUE = 3"}
    )

    assert result["ok"] is True, result
    assert "+VALUE = 3" in calls[0][2]["diff"]
    assert (repo / "src/jarvis_backend/tools.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    assert git(repo, "rev-parse", "HEAD") == result["commit"]
    assert tools_in_repo.restart_required is True


@pytest.mark.asyncio
async def test_edit_source_code_rejects_missing_or_ambiguous_fragment(tools_in_repo, repo: Path) -> None:
    calls: list = []
    tools_in_repo.set_confirmation_callback(_approve(calls))
    (repo / "jarvis_rules.md").write_text("a\na\n", encoding="utf-8")
    git(repo, "commit", "-qam", "dup")

    missing = await tools_in_repo.execute_tool(
        "edit_source_code", {"file_name": "tools.py", "old_text": "NOPE", "new_text": "x"}
    )
    ambiguous = await tools_in_repo.execute_tool(
        "edit_source_code", {"file_name": "jarvis_rules.md", "old_text": "a", "new_text": "b"}
    )

    assert missing["ok"] is False and "не найден" in missing["error"]
    assert ambiguous["ok"] is False and "2 раз" in ambiguous["error"]
    assert calls == []


@pytest.mark.asyncio
async def test_secret_and_git_files_are_off_limits(tools_in_repo, repo: Path) -> None:
    (repo / ".env").write_text("SECRET=1\n", encoding="utf-8")

    env = await tools_in_repo.execute_tool("read_source_code", {"file_name": ".env"})
    git_cfg = await tools_in_repo.execute_tool("read_source_code", {"file_name": ".git/config"})

    assert env["ok"] is False
    assert git_cfg["ok"] is False


@pytest.mark.asyncio
async def test_shell_command_requires_confirmation(tools_in_repo) -> None:
    denied = await tools_in_repo.execute_tool("run_shell_command", {"command": "echo hi"})
    assert denied["ok"] is False and "не подтвердил" in denied["error"]

    calls: list = []
    tools_in_repo.set_confirmation_callback(_approve(calls))
    approved = await tools_in_repo.execute_tool("run_shell_command", {"command": "echo hi"})
    assert approved["ok"] is True
    assert approved["result"].strip() == "hi"
    assert calls[0][2]["summary"] == "Выполнить команду: echo hi"


def test_verifier_catches_import_time_error() -> None:
    broken = (PROJECT_ROOT / "src/jarvis_backend/skills/system_info.py").read_text(encoding="utf-8")
    broken += "\nundefined_name_at_import_time\n"

    result = verifier.verify_change(
        PROJECT_ROOT, {"src/jarvis_backend/skills/system_info.py": broken}, run_tests=False
    )

    assert result.ok is False
    assert result.stage == "import"
    assert "undefined_name_at_import_time" in result.output


def test_verifier_accepts_valid_new_skill() -> None:
    skill = (
        "TOOLS = [{'type': 'function', 'function': {'name': 'ping', 'description': 'ping',"
        " 'parameters': {'type': 'object', 'properties': {}}}}]\n\n"
        "def ping():\n    return 'pong'\n"
    )

    result = verifier.verify_change(
        PROJECT_ROOT, {"src/jarvis_backend/skills/ping.py": skill}, run_tests=False
    )

    assert result.ok is True, result.output


def test_verifier_rejects_broken_skill_contract() -> None:
    skill = (
        "TOOLS = [{'type': 'function', 'function': {'name': 'ping', 'description': 'ping',"
        " 'parameters': {'type': 'object', 'properties': {}}}}]\n"
    )

    result = verifier.verify_change(
        PROJECT_ROOT, {"src/jarvis_backend/skills/ping.py": skill}, run_tests=False
    )

    assert result.ok is False
    assert "ping" in result.output


@pytest.mark.asyncio
async def test_skills_are_loaded_and_executable() -> None:
    from jarvis_backend import tools

    tools.reload_skills()
    names = {schema["function"]["name"] for schema in tools.get_tool_schemas()}

    assert "get_system_info" in names
    assert tools.is_local_tool("get_system_info")
    result = await tools.execute_tool("get_system_info", {})
    assert result["ok"] is True and "os" in result["result"]
