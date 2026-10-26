from __future__ import annotations

import asyncio
import base64
import datetime as dt
import difflib
import io
import json
import platform
import shlex
from pathlib import Path
from typing import Any, Awaitable, Callable

from duckduckgo_search import DDGS
from openai import AsyncOpenAI
from . import git_guard
from .config import settings

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
    "rollback_source_change",
}
MAX_DIFF_CHARS = 20000

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


def _resolve_source_path(file_name: str) -> Path:
    name = (file_name or "").strip()
    if not name:
        raise ToolExecutionError("Parameter 'file_name' is required")

    # Удобный режим:
    # - tools.py -> src/jarvis_backend/tools.py
    # - jarvis_rules.md -> <project_root>/jarvis_rules.md
    if "/" not in name and "\\" not in name:
        source_candidate = SOURCE_ROOT / name
        root_candidate = PROJECT_ROOT / name
        candidate = source_candidate if source_candidate.exists() else root_candidate
    else:
        candidate = PROJECT_ROOT / name
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

    if resolved.suffix not in {".py", ".md"}:
        raise ToolExecutionError("Only .py and .md files are allowed")

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
    data_url = f"data:image/jpeg;base64,{image_b64}"

    base_url = settings.base_url
    api_key = settings.api_key
    # Vision идет через ту же базовую модель агента.
    model = settings.model.strip()

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=45.0)
    response = await client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты анализируешь изображение экрана пользователя. "
                    "Отвечай кратко, по делу, без markdown."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": normalized_query},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )

    text = (response.choices[0].message.content or "").strip()
    return text or "Не удалось получить ответ vision-модели."


async def read_source_code(file_name: str) -> str:
    """Read project source/rules file by short file name (.py/.md)."""
    path = _resolve_source_path(file_name)
    if not path.exists():
        raise ToolExecutionError(f"File not found: {path}")

    def _read_blocking() -> str:
        return path.read_text(encoding="utf-8")

    return await asyncio.to_thread(_read_blocking)


def _prepare_source_update(file_name: str, new_content: Any) -> tuple[Path, str]:
    """Validate an update before asking the user: path sandbox, syntax, git state."""
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
    return path, content


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


async def update_source_code(file_name: str, new_content: str) -> dict[str, Any]:
    """Update .py/.md file: syntax check, git snapshot commit, then write to disk."""
    path, content = _prepare_source_update(file_name, new_content)
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
    return {
        "ok": True,
        "tool": "update_source_code",
        "result": (
            f"Код обновлен и закоммичен ({sha[:8]}). Сервер сейчас будет перезагружен. "
            "Откат: инструмент rollback_source_change или `make rollback`."
        ),
        "commit": sha,
    }


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
    if tool_name == "update_source_code":
        path, content = _prepare_source_update(
            args.get("file_name", ""), args.get("new_content", "")
        )
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        action = "Изменить" if path.exists() else "Создать"
        return {
            "summary": f"{action} файл {rel} (изменение будет закоммичено в git)",
            "diff": _source_diff(path, content),
        }

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
    "update_source_code": update_source_code,
    "rollback_source_change": rollback_source_change,
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
            "description": "Read project source/rules file by file name (e.g., tools.py or jarvis_rules.md).",
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
            "name": "update_source_code",
            "description": (
                "Update project source/rules file; .py content is syntax-checked before save. "
                "Requires user confirmation; the change is committed to git."
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
                        "description": "Full new Python source code content for the file.",
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


async def execute_tool(tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a tool by name with JSON-compatible args."""
    args = args or {}
    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return {
            "ok": False,
            "tool": tool_name,
            "error": f"Unknown tool '{tool_name}'",
        }

    try:
        if tool_name in CONFIRMATION_REQUIRED_TOOLS:
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
        return await tool(**args)
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
