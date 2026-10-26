"""Автозапуск Джарвиса при входе в систему.

macOS: LaunchAgent (~/Library/LaunchAgents/<label>.plist), Linux: systemd --user unit.
Процесс перезапускается системой, если упал; самоправка ядра перезапускает его сама.

    python -m src.jarvis_backend.autostart install | uninstall | status

Модуль намеренно не импортирует остальной проект: он должен работать, даже если
правка агента сломала импорт сервера.
"""
from __future__ import annotations

import os
import platform
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "com.jarvis.agent"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / ".jarvis" / "logs"
HOST = "127.0.0.1"
PORT = os.getenv("JARVIS_PORT", "8000")
# launchd/systemd стартуют с минимальным PATH — добавляем места, где живут uv, npx, uvx.
EXTRA_PATH = [
    str(Path.home() / ".local" / "bin"),
    str(Path.home() / ".cargo" / "bin"),
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
]


def _server_command() -> list[str]:
    uv = shutil.which("uv") or next(
        (str(Path(d) / "uv") for d in EXTRA_PATH if (Path(d) / "uv").is_file()), None
    )
    if uv:
        return [uv, "run", "--project", str(PROJECT_ROOT), "uvicorn", "src.jarvis_backend.server:app", "--host", HOST, "--port", PORT]
    return [sys.executable, "-m", "uvicorn", "src.jarvis_backend.server:app", "--host", HOST, "--port", PORT]


# ---------------------------------------------------------------- macOS


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def build_plist() -> dict:
    return {
        "Label": LABEL,
        "ProgramArguments": _server_command(),
        "WorkingDirectory": str(PROJECT_ROOT),
        "EnvironmentVariables": {"PATH": ":".join(EXTRA_PATH), "PYTHONUNBUFFERED": "1"},
        "RunAtLoad": True,
        # Перезапуск только после падения; штатный выход (например, выключение) — без рестарта.
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 15,
        # Interactive: доступ к микрофону и звуку как у обычного приложения пользователя.
        "ProcessType": "Interactive",
        "StandardOutPath": str(LOG_DIR / "jarvis.log"),
        "StandardErrorPath": str(LOG_DIR / "jarvis.log"),
    }


def _install_macos() -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    path.write_bytes(plistlib.dumps(build_plist()))
    result = _launchctl("bootstrap", _domain(), str(path))
    if result.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {result.stderr.strip()}")
    return f"Установлен LaunchAgent {path}"


def _uninstall_macos() -> str:
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    path = _plist_path()
    if path.exists():
        path.unlink()
        return f"Удалён {path}"
    return "Автозапуск не был установлен"


def _status_macos() -> str:
    if not _plist_path().exists():
        return "Автозапуск не установлен"
    result = _launchctl("print", f"{_domain()}/{LABEL}")
    if result.returncode != 0:
        return f"Установлен ({_plist_path()}), но не загружен"
    state = next((line.strip() for line in result.stdout.splitlines() if line.strip().startswith("state =")), "state = ?")
    pid = next((line.strip() for line in result.stdout.splitlines() if line.strip().startswith("pid =")), "")
    return f"Установлен: {state} {pid}".strip()


# ---------------------------------------------------------------- Linux


def _unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "jarvis.service"


def build_unit() -> str:
    command = " ".join(f'"{part}"' if " " in part else part for part in _server_command())
    return (
        "[Unit]\n"
        "Description=J.A.R.V.I.S. local agent\n"
        "After=network-online.target sound.target\n\n"
        "[Service]\n"
        f"WorkingDirectory={PROJECT_ROOT}\n"
        f"Environment=PATH={':'.join(EXTRA_PATH)}\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        f"ExecStart={command}\n"
        "Restart=on-failure\n"
        "RestartSec=15\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, check=False)


def _install_linux() -> str:
    path = _unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_unit(), encoding="utf-8")
    _systemctl("daemon-reload")
    result = _systemctl("enable", "--now", "jarvis.service")
    if result.returncode != 0:
        raise RuntimeError(f"systemctl enable failed: {result.stderr.strip()}")
    return f"Установлен systemd unit {path}"


def _uninstall_linux() -> str:
    _systemctl("disable", "--now", "jarvis.service")
    path = _unit_path()
    if path.exists():
        path.unlink()
        _systemctl("daemon-reload")
        return f"Удалён {path}"
    return "Автозапуск не был установлен"


def _status_linux() -> str:
    if not _unit_path().exists():
        return "Автозапуск не установлен"
    return "Установлен: " + _systemctl("is-active", "jarvis.service").stdout.strip()


# ---------------------------------------------------------------- CLI


def main(argv: list[str]) -> int:
    action = argv[0] if argv else "status"
    system = platform.system()
    handlers = {
        "Darwin": {"install": _install_macos, "uninstall": _uninstall_macos, "status": _status_macos},
        "Linux": {"install": _install_linux, "uninstall": _uninstall_linux, "status": _status_linux},
    }.get(system)
    if handlers is None:
        print(f"Автозапуск для {system} пока не поддерживается (используйте Планировщик заданий).", file=sys.stderr)
        return 1
    if action not in handlers:
        print("Использование: python -m src.jarvis_backend.autostart install|uninstall|status", file=sys.stderr)
        return 2
    try:
        print(handlers[action]())
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    if action == "install":
        print(f"Логи: {LOG_DIR / 'jarvis.log'}. Веб-интерфейс: http://{HOST}:{PORT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
