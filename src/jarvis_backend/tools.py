from __future__ import annotations

import asyncio
import base64
import datetime as dt
import difflib
import hashlib
import inspect
import io
import json
import platform
import shlex
from pathlib import Path
from typing import Any, Awaitable, Callable

from ddgs import DDGS

from . import git_guard
from . import skills as skills_pkg
from .config import settings
from .verifier import verify_change

try:
    import pyautogui
except Exception:  # noqa: BLE001
    pyautogui = None  # type: ignore[assignment]


class ToolExecutionError(Exception):
    """Raised when a tool cannot be executed successfully."""


PROJECT_ROOT = settings.project_root
SOURCE_ROOT = settings.source_root
_proactive_callback: Callable[[str], Awaitable[None]] | None = None
_reminder_tasks: set[asyncio.Task[Any]] = set()

# Инструменты, которые выполняются только после явного "да" пользователя.
# Аргументы для них может подсунуть prompt injection (web_search, экран, MCP, триггеры),
# поэтому решение принимает человек, а не модель.
CONFIRMATION_REQUIRED_TOOLS = {
    "open_application",
    "update_source_code",
    "edit_source_code",
    "rollback_source_change",
    "run_shell_command",
}
MAX_DIFF_CHARS = 20000
MAX_SHELL_OUTPUT = 8000

# Что агенту разрешено читать и править в собственном проекте.
EDITABLE_SUFFIXES = {".py", ".md", ".html", ".css", ".js", ".json"}
# Никогда: секреты, git, окружение, lock-файлы зависимостей.
FORBIDDEN_PARTS = {".git", ".venv", "venv", "__pycache__", ".jarvis", "node_modules"}

# Правка проверяется в песочнице (компиляция, импорт, тесты) ДО вопроса пользователю.
VERIFY_CHANGES = settings.verify_changes
VERIFY_TIMEOUT = settings.verify_timeout
# (путь, sha256 содержимого) уже прошедших проверку правок — чтобы не гонять тесты дважды.
_verified_changes: set[tuple[str, str]] = set()

# Навыки (skills/*.py) — инструменты, которые агент дописывает себе сам.
_skills = skills_pkg.SkillSet()
# Колбэк «правила изменились» (агент перечитывает jarvis_rules.md без перезапуска).
_rules_changed_callback: Callable[[], None] | None = None
# Изменено ядро (не навык и не правила): нужен перезапуск процесса.
restart_required = False
# Растёт при каждой перезагрузке навыков — сервер по нему рассылает новый список инструментов.
skills_version = 0

ConfirmationCallback = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[bool]]
_confirmation_callback: ConfirmationCallback | None = None


def set_proactive_callback(callback: Callable[[str], Awaitable[None]] | None) -> None:
    """Register callback used by background reminder tasks to push proactive messages."""
    global _proactive_callback
    _proactive_callback = callback


def set_confirmation_callback(callback: ConfirmationCallback | None) -> None:
    """Register callback that asks the user to approve a sensitive tool call.

    Called as callback(tool_name, args, details) and must return True only on explicit approval.
    Without a callback sensitive tools are always denied (fail closed).
    """
    global _confirmation_callback
    _confirmation_callback = callback


def set_rules_changed_callback(callback: Callable[[], None] | None) -> None:
    global _rules_changed_callback
    _rules_changed_callback = callback


def _resolve_source_path(file_name: str) -> Path:
    name = (file_name or "").strip().replace("\\", "/")
    if not name:
        raise ToolExecutionError("Parameter 'file_name' is required")
    if ".." in name.split("/") or name.startswith("/"):
        raise PermissionError("Доступ запрещен: выход за пределы песочницы проекта")

    # Удобный режим:
    # - tools.py -> src/jarvis_backend/tools.py
    # - skills/weather.py -> src/jarvis_backend/skills/weather.py (в т.ч. новый файл)
    # - jarvis_rules.md, static/index.html -> от корня проекта
    source_candidate = SOURCE_ROOT / name
    root_candidate = PROJECT_ROOT / name
    first_part = name.split("/", 1)[0]
    if source_candidate.exists():
        candidate = source_candidate
    elif root_candidate.exists():
        candidate = root_candidate
    elif "/" in name and (SOURCE_ROOT / first_part).is_dir():
        candidate = source_candidate
    elif "/" not in name and name.endswith(".py"):
        candidate = source_candidate
    else:
        candidate = root_candidate
    resolved = candidate.resolve()

    # Защита от Path Traversal: доступ только внутри корня проекта.
    # Требование безопасности: выбрасываем PermissionError с фиксированным сообщением.
    try:
        is_inside_project = resolved.is_relative_to(PROJECT_ROOT)
    except AttributeError:
        # Fallback для старых версий Python (на случай запуска ниже 3.9).
        is_inside_project = PROJECT_ROOT == resolved or PROJECT_ROOT in resolved.parents
    if not is_inside_project:
        raise PermissionError("Доступ запрещен: выход за пределы песочницы проекта")

    rel_parts = resolved.relative_to(PROJECT_ROOT.resolve()).parts
    if any(part in FORBIDDEN_PARTS for part in rel_parts) or resolved.name.startswith(".env"):
        raise PermissionError("Доступ запрещен: служебный или секретный файл")
    if resolved.suffix not in EDITABLE_SUFFIXES:
        raise ToolExecutionError(
            f"Разрешены только файлы {', '.join(sorted(EDITABLE_SUFFIXES))}"
        )

    return resolved


async def open_application(app_name: str) -> dict[str, Any]:
    """Open a local application by name.

    Cross-platform behavior:
    - macOS: `open -a <app_name>`
    - Windows: `cmd /c start "" <app_name>`
    - Linux/Other: try to run executable directly
    """
    if not app_name or not app_name.strip():
        raise ToolExecutionError("Parameter 'app_name' is required")

    app_name = app_name.strip()
    system = platform.system().lower()

    try:
        if "darwin" in system:
            process = await asyncio.create_subprocess_exec(
                "open",
                "-a",
                app_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        elif "windows" in system:
            process = await asyncio.create_subprocess_exec(
                "cmd",
                "/c",
                "start",
                "",
                app_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            cmd = shlex.split(app_name)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        _, stderr = await process.communicate()
        if process.returncode != 0:
            error_text = stderr.decode("utf-8", errors="ignore").strip()
            raise ToolExecutionError(
                f"Failed to open application '{app_name}'. {error_text}"
            )

        return {
            "ok": True,
            "tool": "open_application",
            "result": f"Application launch requested: {app_name}",
        }
    except FileNotFoundError as exc:
        raise ToolExecutionError(
            f"Application or command not found: {app_name}"
        ) from exc


async def get_current_time(fmt: str = "%Y-%m-%d %H:%M:%S") -> dict[str, Any]:
    """Return current local time in the requested format."""
    try:
        now = dt.datetime.now().strftime(fmt)
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"Invalid datetime format '{fmt}': {exc}") from exc

    return {
        "ok": True,
        "tool": "get_current_time",
        "result": now,
    }


async def web_search(query: str, max_results: int = 3) -> str:
    """Search the web via DuckDuckGo and return compact JSON string."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        raise ToolExecutionError("Parameter 'query' is required")

    if max_results < 1:
        max_results = 1
    if max_results > 10:
        max_results = 10

    def _search_blocking() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            results = list(ddgs.text(normalized_query, max_results=max_results))

        compact: list[dict[str, Any]] = []
        for item in results:
            compact.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )
        return compact

    data = await asyncio.to_thread(_search_blocking)
    return json.dumps(data, ensure_ascii=False)


async def analyze_screen(query: str) -> str:
    """Capture screen, send compressed screenshot to local vision model and return text."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        raise ToolExecutionError("Parameter 'query' is required")

    if pyautogui is None:
        raise ToolExecutionError(
            "Dependency 'pyautogui' is missing. Install requirements and restart."
        )

    def _capture_and_encode() -> str:
        image = pyautogui.screenshot()
        # Ограничиваем размер для экономии токенов и bandwidth.
        image.thumbnail((1920, 1080))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=80, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    image_b64 = await asyncio.to_thread(_capture_and_encode)

    # Vision идёт через тот же подключённый мозг (нужна мультимодальная модель).
    from .brain import make_brain

    text = (await make_brain().describe_image(normalized_query, image_b64)).strip()
    return text or "Не удалось получить ответ vision-модели."


async def read_source_code(file_name: str) -> str:
    """Read project source/rules file by short file name."""
    path = _resolve_source_path(file_name)
    if not path.exists():
        raise ToolExecutionError(f"File not found: {path}")

    def _read_blocking() -> str:
        return path.read_text(encoding="utf-8")

    return await asyncio.to_thread(_read_blocking)


async def list_source_files() -> dict[str, Any]:
    """List files Jarvis may read/edit in its own project."""

    def _list_blocking() -> list[str]:
        root = PROJECT_ROOT.resolve()
        found: list[str] = []
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if not path.is_file() or path.suffix not in EDITABLE_SUFFIXES:
                continue
            if any(part in FORBIDDEN_PARTS or part.startswith(".") for part in rel.parts):
                continue
            found.append(f"{rel.as_posix()} ({path.stat().st_size} B)")
        return found

    files = await asyncio.to_thread(_list_blocking)
    return {
        "ok": True,
        "tool": "list_source_files",
        "result": files,
        "skills": sorted(_skills.functions),
        "skill_errors": dict(_skills.errors),
    }


def _apply_edit(path: Path, old_text: Any, new_text: Any) -> str:
    """Точечная замена: old_text должен встречаться в файле ровно один раз."""
    if not path.exists():
        raise ToolExecutionError(f"File not found: {path.name}. Для нового файла используйте update_source_code")
    old = old_text if isinstance(old_text, str) else str(old_text or "")
    new = new_text if isinstance(new_text, str) else str(new_text or "")
    if not old:
        raise ToolExecutionError("Parameter 'old_text' is required")
    current = path.read_text(encoding="utf-8")
    count = current.count(old)
    if count == 0:
        raise ToolExecutionError(
            "old_text не найден в файле. Прочитайте файл заново (read_source_code) и скопируйте фрагмент дословно."
        )
    if count > 1:
        raise ToolExecutionError(f"old_text встречается {count} раз(а) — добавьте окружающие строки, чтобы фрагмент был уникален")
    return current.replace(old, new, 1)


def _content_key(path: Path, content: str) -> tuple[str, str]:
    return str(path), hashlib.sha256(content.encode("utf-8")).hexdigest()


def _prepare_source_update(file_name: str, new_content: Any) -> tuple[Path, str]:
    """Validate an update before asking the user: path sandbox, syntax, git state, sandbox tests."""
    path = _resolve_source_path(file_name)
    content = new_content if isinstance(new_content, str) else str(new_content)

    if path.suffix == ".py":
        try:
            compile(content, "<string>", "exec")
        except (SyntaxError, IndentationError) as exc:
            line = getattr(exc, "lineno", "?")
            text = getattr(exc, "msg", str(exc))
            raise ToolExecutionError(
                f"Ошибка синтаксиса: {text}. Строка: {line}. Файл НЕ обновлен."
            ) from exc

    git_guard.ensure_repo(PROJECT_ROOT)
    git_guard.ensure_file_clean(PROJECT_ROOT, path)

    key = _content_key(path, content)
    if VERIFY_CHANGES and key not in _verified_changes:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        result = verify_change(PROJECT_ROOT, {rel: content}, timeout=VERIFY_TIMEOUT)
        if not result.ok:
            raise ToolExecutionError(result.summary() + "\nФайл НЕ обновлен. Исправьте код и попробуйте снова.")
        _verified_changes.add(key)
    return path, content


def _prepare_source_edit(file_name: str, old_text: Any, new_text: Any) -> tuple[Path, str]:
    path = _resolve_source_path(file_name)
    return _prepare_source_update(file_name, _apply_edit(path, old_text, new_text))


def _source_diff(path: Path, content: str) -> str:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... (diff обрезан)"
    return diff or "(изменений нет)"


async def _commit_source_change(path: Path, content: str, tool_name: str) -> dict[str, Any]:
    rel = path.relative_to(PROJECT_ROOT).as_posix()

    def _commit_and_write_blocking() -> str:
        # Порядок важен: --reload перезапустит сервер сразу после записи файла,
        # поэтому коммит и индекс должны быть готовы до неё.
        sha = git_guard.commit_file(PROJECT_ROOT, path, content, f"update {rel}")
        git_guard.sync_index_with_head(PROJECT_ROOT, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return sha

    sha = await asyncio.to_thread(_commit_and_write_blocking)
    _verified_changes.discard(_content_key(path, content))
    note = _after_change(path)
    return {
        "ok": True,
        "tool": tool_name,
        "result": (
            f"Изменение {rel} закоммичено ({sha[:8]}). {note} "
            "Откат: инструмент rollback_source_change или `make rollback`."
        ),
        "commit": sha,
    }


def _after_change(path: Path) -> str:
    """Подхватить изменение без перезапуска, где это возможно."""
    global restart_required
    resolved = path.resolve()
    if resolved == settings.rules_path.resolve():
        if _rules_changed_callback is not None:
            _rules_changed_callback()
        return "Правила перечитаны — действуют со следующего ответа."
    if resolved.parent == (SOURCE_ROOT / "skills").resolve():
        reload_skills()
        if _skills.errors:
            return f"Навыки перезагружены, но есть ошибки: {_skills.errors}."
        return f"Навыки перезагружены: {', '.join(sorted(_skills.functions)) or 'нет'}."
    if resolved.suffix == ".py":
        restart_required = True
        return "Ядро изменено — процесс будет перезапущен, история диалога сохранится."
    return "Изменение сохранено."


async def update_source_code(file_name: str, new_content: str) -> dict[str, Any]:
    """Create/replace a whole project file: verify, git snapshot commit, then write to disk."""
    path, content = await asyncio.to_thread(_prepare_source_update, file_name, new_content)
    return await _commit_source_change(path, content, "update_source_code")


async def edit_source_code(file_name: str, old_text: str, new_text: str) -> dict[str, Any]:
    """Replace one unique fragment of a project file (safer than rewriting the whole file)."""
    path, content = await asyncio.to_thread(_prepare_source_edit, file_name, old_text, new_text)
    return await _commit_source_change(path, content, "edit_source_code")


async def rollback_source_change() -> dict[str, Any]:
    """Revert the latest agent commit (only if HEAD was made by the agent)."""
    info = await asyncio.to_thread(git_guard.revert_last_agent_commit, PROJECT_ROOT)
    return {
        "ok": True,
        "tool": "rollback_source_change",
        "result": (
            f"Откачен коммит {info.sha[:8]} «{info.subject}» "
            f"({', '.join(info.files)}). Сервер будет перезагружен."
        ),
    }


def describe_tool_call(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Human-readable details for a confirmation prompt; raises if the call is invalid."""
    if tool_name in {"update_source_code", "edit_source_code"}:
        if tool_name == "update_source_code":
            path, content = _prepare_source_update(
                args.get("file_name", ""), args.get("new_content", "")
            )
        else:
            path, content = _prepare_source_edit(
                args.get("file_name", ""), args.get("old_text", ""), args.get("new_text", "")
            )
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        action = "Изменить" if path.exists() else "Создать"
        checked = " Проверка в песочнице пройдена." if VERIFY_CHANGES and path.suffix == ".py" else ""
        return {
            "summary": f"{action} файл {rel} (изменение будет закоммичено в git).{checked}",
            "diff": _source_diff(path, content),
        }

    if tool_name == "run_shell_command":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ToolExecutionError("Parameter 'command' is required")
        return {"summary": f"Выполнить команду: {command}", "diff": command}

    if tool_name == "rollback_source_change":
        info = git_guard.last_agent_commit(PROJECT_ROOT)
        return {
            "summary": f"Откатить коммит {info.sha[:8]} «{info.subject}»",
            "files": info.files,
        }

    if tool_name == "open_application":
        app_name = str(args.get("app_name", "")).strip()
        if not app_name:
            raise ToolExecutionError("Parameter 'app_name' is required")
        return {"summary": f"Запустить приложение: {app_name}"}

    return {"summary": f"Выполнить {tool_name}"}


async def run_shell_command(command: str, timeout: float | None = None) -> dict[str, Any]:
    """Run a shell command on the user's machine (always confirmed by the user)."""
    command = (command or "").strip()
    if not command:
        raise ToolExecutionError("Parameter 'command' is required")
    limit = min(float(timeout or settings.shell_timeout), 600.0)
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path.home()),
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=limit)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ToolExecutionError(f"Команда не завершилась за {limit:.0f} c и была остановлена")
    output = stdout.decode("utf-8", errors="replace")
    if len(output) > MAX_SHELL_OUTPUT:
        output = output[:MAX_SHELL_OUTPUT] + "\n... (вывод обрезан)"
    return {
        "ok": process.returncode == 0,
        "tool": "run_shell_command",
        "exit_code": process.returncode,
        "result": output,
    }


def reload_skills() -> skills_pkg.SkillSet:
    """(Пере)загрузить навыки из skills/*.py."""
    global _skills, skills_version
    _skills = skills_pkg.load_skills(reserved_names=set(TOOL_REGISTRY))
    skills_version += 1
    for name, error in _skills.errors.items():
        print(f"[WARN] Skill '{name}' failed to load: {error}")
    return _skills


def skill_load_errors() -> dict[str, str]:
    return dict(_skills.errors)


def get_tool_schemas() -> list[dict[str, Any]]:
    """Встроенные инструменты + навыки (без MCP)."""
    return [*TOOLS_SCHEMAS, *_skills.schemas]


def is_local_tool(tool_name: str) -> bool:
    return tool_name in TOOL_REGISTRY or tool_name in _skills.functions


async def set_reminder(minutes: int, reminder_text: str) -> str:
    """Set proactive reminder and trigger message after timeout."""
    if minutes <= 0:
        raise ToolExecutionError("Parameter 'minutes' must be greater than 0")

    text = (reminder_text or "").strip()
    if not text:
        raise ToolExecutionError("Parameter 'reminder_text' is required")

    async def _reminder_worker() -> None:
        try:
            await asyncio.sleep(minutes * 60)
            if _proactive_callback is not None:
                await _proactive_callback(text)
        finally:
            # Cleanup task tracking.
            task = asyncio.current_task()
            if task is not None:
                _reminder_tasks.discard(task)

    task = asyncio.create_task(_reminder_worker(), name=f"jarvis_reminder_{minutes}m")
    _reminder_tasks.add(task)
    return "Таймер установлен"


TOOL_REGISTRY = {
    "open_application": open_application,
    "get_current_time": get_current_time,
    "web_search": web_search,
    "analyze_screen": analyze_screen,
    "read_source_code": read_source_code,
    "list_source_files": list_source_files,
    "update_source_code": update_source_code,
    "edit_source_code": edit_source_code,
    "rollback_source_change": rollback_source_change,
    "run_shell_command": run_shell_command,
    "set_reminder": set_reminder,
}


TOOLS_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Open desktop application by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Application name, e.g. 'Calculator' or 'Google Chrome'.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current local datetime string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fmt": {
                        "type": "string",
                        "description": "Optional strftime format, default '%Y-%m-%d %H:%M:%S'.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results (1-10).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_screen",
            "description": (
                "Используй это, когда пользователь просит посмотреть на экран, "
                "прочитать текст с экрана или найти ошибку в видимом коде"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Что нужно проанализировать на текущем экране.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source_code",
            "description": (
                "Read one of your own project files: source (tools.py, agent.py, skills/<name>.py), "
                "rules (jarvis_rules.md) or UI (static/index.html)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Project file name, e.g. 'tools.py' or 'jarvis_rules.md'.",
                    }
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_source_files",
            "description": "List your own project files you can read and edit, plus loaded skills and skill load errors.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_source_code",
            "description": (
                "Preferred way to change your own code/rules: replace one exact, unique fragment of a file. "
                "The change is compiled, imported and tested in a sandbox first; on failure you get the error "
                "output and should fix it. Then the user approves the diff and it is committed to git."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "e.g. 'tools.py', 'skills/weather.py', 'jarvis_rules.md'."},
                    "old_text": {"type": "string", "description": "Exact fragment copied from the current file (must be unique)."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["file_name", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_source_code",
            "description": (
                "Create a new file or fully rewrite one (use for new skills in skills/<name>.py). "
                "Verified in a sandbox (compile, import, tests), then the user approves the diff; committed to git."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Project file name, e.g. 'tools.py' or 'jarvis_rules.md'.",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Full new content of the file.",
                    },
                },
                "required": ["file_name", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rollback_source_change",
            "description": (
                "Revert the latest change made by update_source_code (git revert). "
                "Works only if the last commit was made by the agent. Requires user confirmation."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": (
                "Run a shell command on the user's computer (cwd = home directory) and get its output. "
                "Use for system control the other tools do not cover. Always requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "number", "description": "Seconds before the command is killed (default 60)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Установить таймер и проактивно напомнить пользователю через заданное число минут.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Через сколько минут отправить напоминание.",
                    },
                    "reminder_text": {
                        "type": "string",
                        "description": "Текст напоминания, который нужно озвучить пользователю.",
                    },
                },
                "required": ["minutes", "reminder_text"],
            },
        },
    },
]

LOCAL_TOOL_NAMES = {
    item["function"]["name"] for item in TOOLS_SCHEMAS if item.get("type") == "function"
}

reload_skills()


async def execute_tool(tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a tool by name with JSON-compatible args."""
    args = args or {}
    tool = TOOL_REGISTRY.get(tool_name) or _skills.functions.get(tool_name)

    if tool is None:
        return {
            "ok": False,
            "tool": tool_name,
            "error": f"Unknown tool '{tool_name}'",
        }

    try:
        if tool_name in CONFIRMATION_REQUIRED_TOOLS or tool_name in _skills.confirm:
            # Валидируем до вопроса пользователю, чтобы не спрашивать о заведомо невыполнимом.
            details = await asyncio.to_thread(describe_tool_call, tool_name, args)
            approved = False
            if _confirmation_callback is not None:
                approved = await _confirmation_callback(tool_name, args, details)
            if not approved:
                return {
                    "ok": False,
                    "tool": tool_name,
                    "error": (
                        "Пользователь не подтвердил выполнение. "
                        "Не повторяй вызов без новой просьбы пользователя."
                    ),
                }
        result = tool(**args)
        if inspect.isawaitable(result):
            result = await result
        return result
    except PermissionError as exc:
        return {
            "ok": False,
            "tool": tool_name,
            "error": str(exc),
        }
    except TypeError as exc:
        return {
            "ok": False,
            "tool": tool_name,
            "error": f"Invalid arguments for '{tool_name}': {exc}",
        }
    except (ToolExecutionError, git_guard.GitGuardError) as exc:
        return {
            "ok": False,
            "tool": tool_name,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "tool": tool_name,
            "error": f"Unexpected tool error: {exc}",
        }
