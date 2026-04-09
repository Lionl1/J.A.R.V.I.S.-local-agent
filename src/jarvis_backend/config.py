from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
        return
    except Exception:  # noqa: BLE001
        pass

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(name: str, default: int) -> int:
    value = _env_optional_str(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env_optional_str(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_optional_str(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    project_root: Path
    source_root: Path
    static_dir: Path
    mcp_config_path: Path
    rules_path: Path

    model: str
    base_url: str
    api_key: str

    llm_temperature: float
    llm_timeout: float
    max_tool_steps: int

    tts_voice: str
    stt_language: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str

    hf_token: str | None
    trigger_watch_dir: str | None
    debug_text_mode: bool

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env()

        project_root = Path(__file__).resolve().parents[2]
        source_root = project_root / "src" / "jarvis_backend"

        rules_from_env = _env_optional_str("JARVIS_RULES_PATH")
        rules_path = (
            Path(rules_from_env).expanduser().resolve()
            if rules_from_env
            else (project_root / "jarvis_rules.md").resolve()
        )

        base_url = _env_str("JARVIS_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        api_key = _env_str("JARVIS_API_KEY", _env_str("OPENAI_API_KEY", "local-key"))

        cfg = cls(
            project_root=project_root,
            source_root=source_root,
            static_dir=project_root / "static",
            mcp_config_path=project_root / "mcp_config.json",
            rules_path=rules_path,
            model=_env_str("JARVIS_MODEL", "local-model"),
            base_url=base_url,
            api_key=api_key,
            llm_temperature=_env_float("JARVIS_LLM_TEMPERATURE", 0.2),
            llm_timeout=_env_float("JARVIS_LLM_TIMEOUT", 30.0),
            max_tool_steps=max(1, _env_int("JARVIS_MAX_TOOL_STEPS", 5)),
            tts_voice=_env_str("JARVIS_TTS_VOICE", "ru-RU-SvetlanaNeural"),
            stt_language=_env_str("JARVIS_STT_LANGUAGE", "ru"),
            whisper_model=_env_str("JARVIS_WHISPER_MODEL", "base"),
            whisper_device=_env_str("JARVIS_WHISPER_DEVICE", "auto"),
            whisper_compute_type=_env_str("JARVIS_WHISPER_COMPUTE_TYPE", "int8"),
            hf_token=_env_optional_str("HF_TOKEN"),
            trigger_watch_dir=_env_optional_str("JARVIS_TRIGGER_DIR"),
            debug_text_mode=_env_bool("JARVIS_DEBUG_TEXT_MODE", False),
        )

        # Совместимость для huggingface_hub.
        if cfg.hf_token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = cfg.hf_token

        return cfg


settings = Settings.from_env()
