from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_backend import git_guard

from .conftest import git


def test_commit_file_snapshots_without_touching_worktree_or_user_index(repo: Path) -> None:
    target = repo / "src" / "jarvis_backend" / "tools.py"
    # Пользователь застейджил правку в другом файле — она не должна попасть в коммит агента.
    (repo / "other.py").write_text("OTHER = 2\n", encoding="utf-8")
    git(repo, "add", "other.py")

    sha = git_guard.commit_file(repo, target, "VALUE = 2\n", "update src/jarvis_backend/tools.py")

    assert git(repo, "rev-parse", "HEAD") == sha
    assert git(repo, "log", "-1", "--format=%s|%ae") == (
        f"{git_guard.COMMIT_PREFIX}update src/jarvis_backend/tools.py|{git_guard.AUTHOR_EMAIL}"
    )
    assert git(repo, "show", "--name-only", "--format=", "HEAD") == "src/jarvis_backend/tools.py"
    assert git(repo, "show", "HEAD:src/jarvis_backend/tools.py") == "VALUE = 2"
    # Рабочая копия ещё не тронута.
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"

    git_guard.sync_index_with_head(repo, target)
    # Индекс совпадает с HEAD для файла агента, застейдженная правка пользователя на месте.
    assert git(repo, "diff", "--cached", "--name-only") == "other.py"


def test_ensure_file_clean_rejects_user_changes(repo: Path) -> None:
    target = repo / "src" / "jarvis_backend" / "tools.py"
    target.write_text("VALUE = 42\n", encoding="utf-8")

    with pytest.raises(git_guard.GitGuardError, match="незакоммиченные"):
        git_guard.ensure_file_clean(repo, target)


def test_ensure_repo_requires_git(tmp_path: Path) -> None:
    with pytest.raises(git_guard.GitGuardError, match="git"):
        git_guard.ensure_repo(tmp_path)


def test_revert_last_agent_commit_restores_file(repo: Path) -> None:
    target = repo / "src" / "jarvis_backend" / "tools.py"
    git_guard.commit_file(repo, target, "VALUE = 2\n", "update tools.py")
    git_guard.sync_index_with_head(repo, target)
    target.write_text("VALUE = 2\n", encoding="utf-8")

    info = git_guard.revert_last_agent_commit(repo)

    assert info.files == ["src/jarvis_backend/tools.py"]
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert git(repo, "status", "--porcelain") == ""


def test_revert_refuses_user_commit(repo: Path) -> None:
    with pytest.raises(git_guard.GitGuardError, match="не агентом"):
        git_guard.revert_last_agent_commit(repo)
    assert git(repo, "log", "-1", "--format=%s") == "initial"


def test_revert_is_not_chained_over_previous_revert(repo: Path) -> None:
    target = repo / "src" / "jarvis_backend" / "tools.py"
    git_guard.commit_file(repo, target, "VALUE = 2\n", "update tools.py")
    git_guard.sync_index_with_head(repo, target)
    target.write_text("VALUE = 2\n", encoding="utf-8")
    git_guard.revert_last_agent_commit(repo)

    # HEAD теперь revert-коммит, а не правка агента: второй откат запрещён.
    with pytest.raises(git_guard.GitGuardError):
        git_guard.revert_last_agent_commit(repo)
