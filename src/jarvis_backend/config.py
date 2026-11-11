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


# Пресеты «мозгов»: любой OpenAI-совместимый сервер + нативный Anthropic.
# JARVIS_BASE_URL / JARVIS_MODEL / JARVIS_API_KEY переопределяют значения пресета.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": "local-model", "key_env": ""},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "key_env": ""},
    "vllm": {"base_url": "http://localhost:8000/v1", "model": "local-model", "key_env": ""},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini", "key_env": "OPENROUTER_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash", "key_env": "GEMINI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com", "model": "claude-opus-5-5", "key_env": "ANTHROPIC_API_KEY"},
}


@dataclass(slots=True)
class Settings:
    project_root: Path
    source_root: Path
    static_dir: Path
    mcp_config_path: Path
    rules_path: Path

    provider: str
    model: str
    base_url: str
    api_key: str
    server_api_key: str

    llm_temperature: float
    llm_timeout: float
    llm_max_retries: int
    llm_max_tokens: int
    llm_effort: str | None
    max_tool_steps: int
    max_history_messages: int

    tts_voice: str
    tts_engine: str
    system_tts_voice: str | None
    stt_language: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str

    hf_token: str | None
    trigger_watch_dir: str | None
    debug_text_mode: bool
    confirm_timeout: float
    mcp_auto_approve: tuple[str, ...]

    wake_word: bool
    wake_engine: str
    wake_threshold: float
    wake_device: str | None
    wake_model: str
    voice_reply_local: bool
    voice_debug: bool

    state_dir: Path
    skills_dir: Path
    verify_changes: bool
    verify_timeout: float
    shell_timeout: float

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env()

        project_root = Path(__file__).resolve().parents[2]
        source_root = project_root / "src" / "jarvis_backend"

        default_rules = (project_root / "jarvis_rules.md").resolve()
        rules_from_env = _env_optional_str("JARVIS_RULES_PATH")
        rules_path = default_rules
        if rules_from_env:
            candidate = Path(rules_from_env).expanduser().resolve()
            if candidate.exists():
                rules_path = candidate
            else:
                # Битый путь в .env не должен лишать агента личности и правил RSI.
                print(f"[WARN] JARVIS_RULES_PATH not found: {candidate}. Using {default_rules}.")

        provider = _env_str("JARVIS_PROVIDER", "lmstudio").lower()
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["lmstudio"])
        base_url = _env_str("JARVIS_BASE_URL", preset["base_url"]).rstrip("/")
        api_key = _env_str(
            "JARVIS_API_KEY",
            _env_str(preset["key_env"], "local-key") if preset["key_env"] else "local-key",
        )

        cfg = cls(
            project_root=project_root,
            source_root=source_root,
            static_dir=project_root / "static",
            mcp_config_path=project_root / "mcp_config.json",
            rules_path=rules_path,
            provider=provider,
            model=_env_str("JARVIS_MODEL", preset["model"]),
            base_url=base_url,
            api_key=api_key,
            # Ключ веб-интерфейса отдельно от ключа LLM: у облачных провайдеров
            # это секрет, который не должен уходить в браузер.
            server_api_key=_env_str("JARVIS_SERVER_API_KEY", api_key),
            llm_temperature=_env_float("JARVIS_LLM_TEMPERATURE", 0.2),
            llm_timeout=_env_float("JARVIS_LLM_TIMEOUT", 60.0),
            llm_max_retries=max(0, _env_int("JARVIS_LLM_MAX_RETRIES", 1)),
            llm_max_tokens=max(256, _env_int("JARVIS_LLM_MAX_TOKENS", 16000)),
            llm_effort=_env_optional_str("JARVIS_EFFORT"),
            max_tool_steps=max(1, _env_int("JARVIS_MAX_TOOL_STEPS", 12)),
            max_history_messages=max(4, _env_int("JARVIS_MAX_HISTORY", 60)),
            tts_voice=_env_str("JARVIS_TTS_VOICE", "ru-RU-SvetlanaNeural"),
            tts_engine=_env_str("JARVIS_TTS_ENGINE", "auto").lower(),
            system_tts_voice=_env_optional_str("JARVIS_SYSTEM_TTS_VOICE"),
            stt_language=_env_str("JARVIS_STT_LANGUAGE", "ru"),
            whisper_model=_env_str("JARVIS_WHISPER_MODEL", "base"),
            whisper_device=_env_str("JARVIS_WHISPER_DEVICE", "auto"),
            whisper_compute_type=_env_str("JARVIS_WHISPER_COMPUTE_TYPE", "int8"),
            hf_token=_env_optional_str("HF_TOKEN"),
            trigger_watch_dir=_env_optional_str("JARVIS_TRIGGER_DIR"),
            debug_text_mode=_env_bool("JARVIS_DEBUG_TEXT_MODE", False),
            confirm_timeout=max(5.0, _env_float("JARVIS_CONFIRM_TIMEOUT", 120.0)),
            mcp_auto_approve=tuple(
                item.strip()
                for item in _env_str("JARVIS_MCP_AUTO_APPROVE", "").split(",")
                if item.strip()
            ),
            wake_word=_env_bool("JARVIS_WAKE_WORD", False),
            wake_engine=_env_str("JARVIS_WAKE_ENGINE", "whisper").lower(),
            wake_threshold=_env_float("JARVIS_WAKE_THRESHOLD", 0.5),
            wake_device=_env_optional_str("JARVIS_WAKE_DEVICE"),
            wake_model=_env_str("JARVIS_WAKE_MODEL", "hey_jarvis"),
            voice_reply_local=_env_bool("JARVIS_VOICE_REPLY_LOCAL", True),
            voice_debug=_env_bool("JARVIS_VOICE_DEBUG", False),
            state_dir=project_root / ".jarvis",
            skills_dir=source_root / "skills",
            verify_changes=_env_bool("JARVIS_VERIFY_CHANGES", True),
            verify_timeout=max(10.0, _env_float("JARVIS_VERIFY_TIMEOUT", 180.0)),
            shell_timeout=max(1.0, _env_float("JARVIS_SHELL_TIMEOUT", 60.0)),
        )

        # Совместимость для huggingface_hub.
        if cfg.hf_token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = cfg.hf_token

        return cfg


settings = Settings.from_env()
