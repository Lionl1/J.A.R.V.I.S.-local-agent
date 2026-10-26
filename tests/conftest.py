from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Test User", "-c", "user.email=user@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Minimal project layout: <root>/src/jarvis_backend/*.py under git with one user commit."""
    root = tmp_path / "project"
    source = root / "src" / "jarvis_backend"
    source.mkdir(parents=True)
    (source / "tools.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "jarvis_rules.md").write_text("# rules\n", encoding="utf-8")
    (root / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "initial")
    return root


@pytest.fixture()
def tools_in_repo(repo: Path, monkeypatch: pytest.MonkeyPatch):
    from jarvis_backend import tools

    monkeypatch.setattr(tools, "PROJECT_ROOT", repo)
    monkeypatch.setattr(tools, "SOURCE_ROOT", repo / "src" / "jarvis_backend")
    monkeypatch.setattr(tools, "_confirmation_callback", None)
    return tools
