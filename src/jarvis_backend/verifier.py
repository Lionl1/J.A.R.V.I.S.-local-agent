"""Проверка самоправки до того, как она попадёт на диск.

Правка применяется к временной копии проекта, после чего в отдельном процессе:
1) компилируются все модули;
2) импортируются сервер, консольный режим и все навыки (ловит NameError/ImportError,
   которые синтаксическая проверка не видит);
3) запускаются тесты проекта (если установлен pytest).

Только прошедшая проверку правка показывается пользователю на подтверждение.
Если проверка упала, агент получает вывод ошибки и может исправить код сам.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Что копируем в песочницу: этого достаточно для импорта и тестов.
_COPY_ITEMS = ("src", "tests", "pyproject.toml", "jarvis_rules.md", "static")
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
MAX_OUTPUT_CHARS = 6000

_IMPORT_SMOKE = """
import importlib, pkgutil, sys
import jarvis_backend
mods = [m.name for m in pkgutil.walk_packages(jarvis_backend.__path__, "jarvis_backend.")]
for name in sorted(mods):
    importlib.import_module(name)
from jarvis_backend import tools
tools.reload_skills()
errors = tools.skill_load_errors()
if errors:
    print("Ошибки загрузки навыков:", errors)
    sys.exit(1)
print("imported", len(mods), "modules")
"""


@dataclass(slots=True)
class VerifyResult:
    ok: bool
    stage: str
    output: str

    def summary(self) -> str:
        if self.ok:
            return f"Проверка пройдена ({self.stage})"
        return f"Проверка не пройдена на этапе «{self.stage}»:\n{self.output}"


def _tail(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_OUTPUT_CHARS:
        return "…" + text[-MAX_OUTPUT_CHARS:]
    return text


def _python_for(project_root: Path) -> str:
    """Интерпретатор с зависимостями проекта (в Jarvis.app sys.executable — голый Python)."""
    venv_python = project_root / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def _pytest_available(python: str) -> bool:
    result = subprocess.run([python, "-c", "import pytest"], capture_output=True, check=False)
    return result.returncode == 0


def _run(cmd: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "") if isinstance(exc.stdout, str) else ""
        return 124, f"Превышено время проверки ({timeout:.0f} c)\n{out}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def verify_change(
    project_root: Path,
    changes: dict[str, str],
    *,
    run_tests: bool = True,
    timeout: float = 180.0,
) -> VerifyResult:
    """changes: {относительный путь: новое содержимое}."""
    for rel, content in changes.items():
        if rel.endswith(".json"):
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                return VerifyResult(False, "json", f"{rel}: {exc}")

    if not any(rel.endswith(".py") for rel in changes):
        # Правила (.md), UI (.html) и конфиги не ломают импорт — достаточно проверок выше.
        return VerifyResult(True, "static", "")

    python = _python_for(project_root)
    with tempfile.TemporaryDirectory(prefix="jarvis-verify-") as tmp:
        sandbox = Path(tmp) / "project"
        sandbox.mkdir()
        for item in _COPY_ITEMS:
            src = project_root / item
            if src.is_dir():
                shutil.copytree(src, sandbox / item, ignore=_IGNORE)
            elif src.is_file():
                shutil.copy2(src, sandbox / item)
        for rel, content in changes.items():
            target = sandbox / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        env = {
            k: v
            for k, v in os.environ.items()
            # Песочница не должна ходить в реальный мозг и читать реальные пути.
            if not k.startswith(("JARVIS_", "OPENAI_", "ANTHROPIC_"))
        }
        env.update(
            {
                "PYTHONPATH": str(sandbox / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "JARVIS_PROVIDER": "lmstudio",
                "JARVIS_API_KEY": "verify-key",
                "JARVIS_TRIGGER_DIR": "",
            }
        )

        code, out = _run(
            [python, "-m", "compileall", "-q", "src"], sandbox, env, timeout
        )
        if code != 0:
            return VerifyResult(False, "compile", _tail(out))

        code, out = _run([python, "-c", _IMPORT_SMOKE], sandbox, env, timeout)
        if code != 0:
            return VerifyResult(False, "import", _tail(out))

        if run_tests and (sandbox / "tests").is_dir() and _pytest_available(python):
            code, out = _run(
                [python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
                sandbox,
                env,
                timeout,
            )
            if code != 0:
                return VerifyResult(False, "tests", _tail(out))
            return VerifyResult(True, "compile+import+tests", "")

        return VerifyResult(True, "compile+import", "")
