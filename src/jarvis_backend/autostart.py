"""Автозапуск Джарвиса при входе в систему.

macOS: LaunchAgent (~/Library/LaunchAgents/<label>.plist), Linux: systemd --user unit.
Процесс перезапускается системой, если упал; самоправка ядра перезапускает его сама.

    python -m src.jarvis_backend.autostart on | off | stop | install | uninstall | status

    off — аварийный выключатель: остановить Джарвиса (и запущенного вручную тоже),
          убрать автозапуск, выгрузить модель LM Studio. on — включить всё обратно.

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
import time
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


APP_BUNDLE = PROJECT_ROOT / ".jarvis" / "Jarvis.app"
MIC_USAGE = (
    "Jarvis listens for the wake word «Джарвис» and your voice commands. "
    "Speech is recognised locally on this computer."
)


def _venv_site_packages() -> Path | None:
    found = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python3.*/site-packages"))
    return found[-1] if found else None


def _base_interpreter() -> Path | None:
    """Настоящий бинарник Python, на который ссылается .venv (framework-сборка macOS)."""
    cfg = PROJECT_ROOT / ".venv" / "pyvenv.cfg"
    if not cfg.exists():
        return None
    home = next(
        (line.split("=", 1)[1].strip() for line in cfg.read_text().splitlines() if line.startswith("home")),
        "",
    )
    candidate = Path(home).parent / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    return candidate if candidate.exists() else None


def build_app_bundle() -> Path | None:
    """Jarvis.app: копия интерпретатора со своим bundle id и описанием доступа к микрофону.

    macOS выдаёт доступ к микрофону приложению, а не голому python, запущенному launchd в фоне:
    без бандла фоновый процесс молча получает тишину. Бандл подписывается локально (ad-hoc).
    """
    interpreter = _base_interpreter()
    if interpreter is None or shutil.which("codesign") is None:
        return None
    macos_dir = APP_BUNDLE / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True, exist_ok=True)
    executable = macos_dir / "Jarvis"
    info = {
        "CFBundleIdentifier": LABEL,
        "CFBundleName": "Jarvis",
        "CFBundleDisplayName": "Jarvis",
        "CFBundleExecutable": "Jarvis",
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1",
        "CFBundleShortVersionString": "1.0",
        "LSUIElement": True,  # без иконки в Dock
        "NSMicrophoneUsageDescription": MIC_USAGE,
    }
    info_path = APP_BUNDLE / "Contents" / "Info.plist"
    unchanged = (
        executable.exists()
        and executable.stat().st_size == interpreter.stat().st_size
        and int(executable.stat().st_mtime) == int(interpreter.stat().st_mtime)
        and info_path.exists()
        and plistlib.loads(info_path.read_bytes()) == info
    )
    if unchanged:
        # Не пересобираем: новая подпись может заставить macOS снова спросить доступ к микрофону.
        return executable
    shutil.copy2(interpreter, executable)
    info_path.write_bytes(plistlib.dumps(info))
    result = subprocess.run(
        ["codesign", "--force", "--sign", "-", "--identifier", LABEL, str(APP_BUNDLE)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"codesign failed: {result.stderr.strip()}")
    return executable


def _bundle_command(executable: Path) -> list[str] | None:
    site_packages = _venv_site_packages()
    if site_packages is None:
        return None
    # -S: без глобальных site-packages системного Python — только пакеты .venv проекта.
    launcher = (
        "import os, runpy, site, sys; "
        f"os.chdir({str(PROJECT_ROOT)!r}); sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        f"site.addsitedir({str(site_packages)!r}); "
        f"sys.argv = ['uvicorn', 'src.jarvis_backend.server:app', '--host', {HOST!r}, '--port', {PORT!r}]; "
        "runpy.run_module('uvicorn', run_name='__main__', alter_sys=True)"
    )
    return [str(executable), "-S", "-c", launcher]


def _server_command(use_bundle: bool = False) -> list[str]:
    if use_bundle:
        executable = APP_BUNDLE / "Contents" / "MacOS" / "Jarvis"
        command = _bundle_command(executable) if executable.exists() else None
        if command:
            return command
    server_args = ["-m", "uvicorn", "src.jarvis_backend.server:app", "--host", HOST, "--port", PORT]
    # Запускаем Python окружения напрямую, а не через `uv run`: macOS выдаёт доступ к микрофону
    # «ответственному» процессу — тому, кого запустил launchd. Через uv им был бы uv,
    # и разрешение, выданное Python, не действовало бы.
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python), *server_args]
    uv = shutil.which("uv") or next(
        (str(Path(d) / "uv") for d in EXTRA_PATH if (Path(d) / "uv").is_file()), None
    )
    if uv:
        return [uv, "run", "--project", str(PROJECT_ROOT), "python", *server_args]
    return [sys.executable, *server_args]


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
        "ProgramArguments": _server_command(use_bundle=True),
        "AssociatedBundleIdentifiers": [LABEL],
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


def _loaded_macos() -> bool:
    return _launchctl("print", f"{_domain()}/{LABEL}").returncode == 0


def _bootout_macos() -> None:
    """Выгрузить службу и дождаться этого: bootout асинхронный, а bootstrap сразу после него падает."""
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    deadline = time.monotonic() + 15
    while _loaded_macos() and time.monotonic() < deadline:
        time.sleep(0.3)


def _install_macos() -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _bootout_macos()
    try:
        build_app_bundle()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Jarvis.app не собран ({exc}); голосовая активация в фоне может не получить микрофон")
    path.write_bytes(plistlib.dumps(build_plist()))
    for attempt in range(3):
        result = _launchctl("bootstrap", _domain(), str(path))
        if result.returncode == 0:
            break
        time.sleep(1.0 + attempt)
    else:
        raise RuntimeError(f"launchctl bootstrap failed: {result.stderr.strip()}")
    return f"Установлен LaunchAgent {path}"


def _stop_macos() -> str:
    """Остановить сейчас; при следующем входе в систему Джарвис запустится снова."""
    if not _loaded_macos():
        return "Джарвис не запущен"
    _bootout_macos()
    return "Джарвис остановлен (автозапуск при входе в систему сохранён)"


def _uninstall_macos() -> str:
    _bootout_macos()
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


def _stop_linux() -> str:
    _systemctl("stop", "jarvis.service")
    return "Джарвис остановлен (автозапуск при входе в систему сохранён)"


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


# ---------------------------------------------------------------- LM Studio и остальное


def _env_value(name: str) -> str:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return ""
    value = ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            value = line.split("=", 1)[1].strip().strip("'\"")
    return value


def _lms() -> str | None:
    candidate = Path.home() / ".lmstudio" / "bin" / "lms"
    return str(candidate) if candidate.exists() else shutil.which("lms")


def _uses_lmstudio() -> bool:
    return (_env_value("JARVIS_PROVIDER") or "lmstudio").lower() == "lmstudio" and _lms() is not None


def _run_quiet(cmd: list[str], timeout: float = 120) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _kill_stray_processes() -> int:
    """Джарвис, запущенный вручную (make start/dev/run), а не через автозапуск."""
    pids: set[int] = set()
    for pattern in ("src.jarvis_backend.server:app", "src.jarvis_backend.main"):
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
        pids.update(int(pid) for pid in result.stdout.split())
    pids.discard(os.getpid())
    killed = 0
    for pid in pids:
        try:
            os.kill(pid, 15)
            killed += 1
        except OSError:
            pass
    return killed


def turn_off() -> list[str]:
    """Аварийный выключатель: остановить всё и убрать автозапуск."""
    # Jarvis.app не удаляем: с ним связано разрешение на микрофон. Без службы он не запускается.
    report = [_uninstall_macos() if platform.system() == "Darwin" else _uninstall_linux()]
    killed = _kill_stray_processes()
    if killed:
        report.append(f"Остановлено процессов, запущенных вручную: {killed}")
    if _uses_lmstudio():
        lms = _lms()
        assert lms is not None
        _run_quiet([lms, "unload", "--all"])
        _run_quiet([lms, "server", "stop"])
        report.append("LM Studio: модель выгружена из памяти, сервер остановлен")
    report.append("Микрофон больше не слушается. Включить обратно: make on")
    return report


def turn_on() -> list[str]:
    report = []
    if _uses_lmstudio():
        lms = _lms()
        assert lms is not None
        _run_quiet([lms, "server", "start"])
        model = _env_value("JARVIS_MODEL")
        if model and model != "local-model":
            loaded = subprocess.run([lms, "ps"], capture_output=True, text=True, check=False).stdout
            if model in loaded:
                report.append(f"LM Studio: сервер запущен, модель {model} уже загружена")
            else:
                ok = _run_quiet([lms, "load", model, "--context-length", "16384", "-y"], timeout=300)
                report.append(f"LM Studio: сервер запущен, модель {model} {'загружена' if ok else 'НЕ загрузилась'}")
        else:
            report.append("LM Studio: сервер запущен")
    report.append(_install_macos() if platform.system() == "Darwin" else _install_linux())
    return report


# ---------------------------------------------------------------- CLI


def main(argv: list[str]) -> int:
    action = argv[0] if argv else "status"
    system = platform.system()
    handlers = {
        "Darwin": {"install": _install_macos, "uninstall": _uninstall_macos, "status": _status_macos, "stop": _stop_macos},
        "Linux": {"install": _install_linux, "uninstall": _uninstall_linux, "status": _status_linux, "stop": _stop_linux},
    }.get(system)
    if handlers is None:
        print(f"Автозапуск для {system} пока не поддерживается (используйте Планировщик заданий).", file=sys.stderr)
        return 1
    if action not in handlers and action not in {"off", "on"}:
        print("Использование: python -m src.jarvis_backend.autostart on|off|stop|install|uninstall|status", file=sys.stderr)
        return 2
    try:
        if action == "off":
            print("\n".join(turn_off()))
            return 0
        if action == "on":
            print("\n".join(turn_on()))
            action = "install"
        else:
            print(handlers[action]())
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    if action == "install":
        print(f"Логи: {LOG_DIR / 'jarvis.log'}. Веб-интерфейс: http://{HOST}:{PORT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
