from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Все правки агента коммитятся от отдельного автора с этим префиксом,
# чтобы откат мог отличить их от коммитов пользователя.
COMMIT_PREFIX = "jarvis: "
AUTHOR_NAME = "J.A.R.V.I.S."
AUTHOR_EMAIL = "jarvis@localhost"


class GitGuardError(Exception):
    """Raised when a git snapshot or rollback cannot be performed safely."""


@dataclass(slots=True)
class CommitInfo:
    sha: str
    subject: str
    files: list[str]


def _identity_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
        }
    )
    if extra:
        env.update(extra)
    return env


def _git(
    repo: Path,
    *args: str,
    input_data: bytes | None = None,
    env: dict[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            input=input_data,
            capture_output=True,
            env=env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitGuardError("git не установлен") from exc
    if result.returncode != 0:
        error_text = result.stderr.decode("utf-8", errors="ignore").strip()
        raise GitGuardError(f"git {args[0]} failed: {error_text}")
    return result.stdout.decode("utf-8", errors="ignore").strip()


def _relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def ensure_repo(repo: Path) -> None:
    """Fail closed: self-modification is allowed only inside a git repo with commits."""
    try:
        top = Path(_git(repo, "rev-parse", "--show-toplevel"))
        _git(repo, "rev-parse", "--verify", "HEAD")
    except GitGuardError as exc:
        raise GitGuardError(
            "Проект не является git-репозиторием с коммитами — правка кода запрещена"
        ) from exc
    if top.resolve() != repo.resolve():
        raise GitGuardError(f"Корень git ({top}) не совпадает с корнем проекта ({repo})")


def ensure_file_clean(repo: Path, path: Path) -> None:
    """Refuse to touch a file with user's uncommitted (or staged) changes."""
    status = _git(repo, "status", "--porcelain", "--", _relative(repo, path))
    if status:
        raise GitGuardError(
            f"В файле {_relative(repo, path)} есть незакоммиченные изменения. "
            "Закоммитьте или отмените их, прежде чем агент будет править этот файл."
        )


def commit_file(repo: Path, path: Path, content: str, message: str) -> str:
    """Commit new file content on top of HEAD without touching the user's index.

    The commit is created *before* the file is written to the working tree:
    the dev server reloads as soon as a .py file changes, and the snapshot must
    already exist by then. Uses a temporary index so staged user changes in
    other files are neither included nor disturbed.
    """
    rel = _relative(repo, path)
    blob = _git(repo, "hash-object", "-w", "--stdin", input_data=content.encode("utf-8"))

    fd, index_path = tempfile.mkstemp(prefix="jarvis-index-")
    os.close(fd)
    try:
        env = _identity_env({"GIT_INDEX_FILE": index_path})
        _git(repo, "read-tree", "HEAD", env=env)
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{rel}", env=env)
        tree = _git(repo, "write-tree", env=env)
    finally:
        Path(index_path).unlink(missing_ok=True)

    parent = _git(repo, "rev-parse", "HEAD")
    sha = _git(
        repo,
        "commit-tree",
        tree,
        "-p",
        parent,
        "-m",
        f"{COMMIT_PREFIX}{message}",
        env=_identity_env(),
    )
    # Сдвигаем HEAD (текущую ветку) только если за это время его никто не поменял.
    _git(repo, "update-ref", "-m", f"{COMMIT_PREFIX}{message}", "HEAD", sha, parent)
    return sha


def sync_index_with_head(repo: Path, path: Path) -> None:
    """After writing the committed content, make the real index match HEAD for this path."""
    _git(repo, "reset", "-q", "HEAD", "--", _relative(repo, path))


def last_agent_commit(repo: Path) -> CommitInfo:
    """Return HEAD if it is an agent commit; rollback never touches user commits."""
    subject = _git(repo, "log", "-1", "--format=%s", "HEAD")
    author = _git(repo, "log", "-1", "--format=%ae", "HEAD")
    if not subject.startswith(COMMIT_PREFIX) or author != AUTHOR_EMAIL:
        raise GitGuardError(
            "Последний коммит сделан не агентом — откатывать нечего "
            f"(HEAD: «{subject}»)"
        )
    sha = _git(repo, "rev-parse", "HEAD")
    files = _git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()
    return CommitInfo(sha=sha, subject=subject, files=[f for f in files if f])


def revert_last_agent_commit(repo: Path) -> CommitInfo:
    info = last_agent_commit(repo)
    for rel in info.files:
        ensure_file_clean(repo, repo / rel)
    _git(repo, "revert", "--no-edit", info.sha, env=_identity_env())
    return info


if __name__ == "__main__":
    # `make rollback`: работает без запуска сервера и без зависимостей проекта —
    # на случай, если правка агента сломала импорт и сервер не поднимается.
    import sys

    project_root = Path(__file__).resolve().parents[2]
    try:
        reverted = revert_last_agent_commit(project_root)
    except GitGuardError as exc:
        print(f"Откат не выполнен: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Откачен {reverted.sha[:8]} «{reverted.subject}»: {', '.join(reverted.files)}")
