"""Подключаемый «мозг» Джарвиса.

История диалога хранится в одном каноническом формате (OpenAI chat messages:
system / user / assistant(+tool_calls) / tool). Каждый адаптер сам переводит её
в формат своего API, поэтому мозг можно поменять одной переменной окружения
(JARVIS_PROVIDER), не трогая агента и инструменты.

Ключи сообщений, начинающиеся с "_", — служебные (например, сырые блоки ответа
Anthropic для дословного повтора) и наружу OpenAI-совместимым серверам не уходят.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, settings


CONNECT_TIMEOUT = 5.0


_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Локальные reasoning-модели (Qwen, DeepSeek-R1…) пишут размышления в <think>…</think> —
    их нельзя зачитывать вслух и хранить как ответ."""
    return _THINK_RE.sub("", text).strip()


class BrainError(RuntimeError):
    """LLM недоступна или вернула ошибку; текст пригоден для показа пользователю."""


@dataclass(slots=True)
class BrainReply:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Дополнительные служебные поля для assistant-сообщения в истории.
    extra: dict[str, Any] = field(default_factory=dict)


class Brain:
    provider = "base"

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.base_url = cfg.base_url

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> BrainReply:
        raise NotImplementedError

    async def describe_image(self, prompt: str, image_b64: str, media_type: str = "image/jpeg") -> str:
        reply = await self.chat(
            [
                {"role": "system", "content": "Ты анализируешь изображение экрана пользователя. Отвечай кратко, по делу, без markdown."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                    ],
                },
            ]
        )
        return reply.content

    def label(self) -> str:
        return f"{self.provider}:{self.model} @ {self.base_url}"


# --------------------------------------------------------------------------- OpenAI-совместимые


class OpenAICompatBrain(Brain):
    """OpenAI, LM Studio, Ollama, vLLM, OpenRouter, Groq, DeepSeek, Gemini и т.п."""

    provider = "openai-compatible"

    def __init__(self, cfg: Settings) -> None:
        super().__init__(cfg)
        from openai import AsyncOpenAI, Timeout

        self.provider = cfg.provider
        self.client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or "local-key",
            # Короткий connect: недоступный сервер, молча роняющий пакеты, не должен держать UI минуту.
            timeout=Timeout(cfg.llm_timeout, connect=CONNECT_TIMEOUT),
            # По умолчанию SDK делает 2 повтора — при мёртвом сервере UI висел ~90 секунд.
            max_retries=cfg.llm_max_retries,
        )
        # Если сервер однажды отверг tools, больше не пытаемся.
        self._tools_supported = True

    @staticmethod
    def _clean(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]

    async def chat(self, messages, tools=None) -> BrainReply:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._clean(messages),
            "temperature": self.cfg.llm_temperature,
        }
        if tools and self._tools_supported:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            try:
                response = await self.client.chat.completions.create(**payload)
            except APIStatusError as exc:
                # Некоторые локальные модели/серверы не поддерживают tools — падаем в обычный чат.
                if "tools" in payload and exc.status_code in (400, 422):
                    self._tools_supported = False
                    payload.pop("tools")
                    payload.pop("tool_choice")
                    response = await self.client.chat.completions.create(**payload)
                else:
                    raise
        except APITimeoutError as exc:
            raise BrainError(f"Мозг не ответил вовремя ({self.label()})") from exc
        except APIConnectionError as exc:
            raise BrainError(
                f"Нет связи с мозгом ({self.label()}). Проверьте, что LLM-сервер запущен, "
                "или смените JARVIS_PROVIDER / JARVIS_BASE_URL в .env"
            ) from exc
        except AuthenticationError as exc:
            raise BrainError(f"Мозг отклонил API-ключ ({self.label()})") from exc
        except APIStatusError as exc:
            raise BrainError(f"Мозг вернул HTTP {exc.status_code} ({self.label()}): {exc.message}") from exc

        message = response.choices[0].message
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
            }
            for call in (message.tool_calls or [])
        ]
        return BrainReply(content=strip_reasoning(message.content or ""), tool_calls=tool_calls)


# --------------------------------------------------------------------------- Anthropic (Claude)

# Модели с адаптивным мышлением: для них включаем block_binding, чтобы правка
# системного промпта/набора инструментов (а это и есть RSI) не ломала историю.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
)
# Серверный fallback при отказе классификатора поддерживают не все модели (у Haiku его нет).
_SERVER_FALLBACK_PREFIXES = ("claude-fable-5", "claude-opus-5", "claude-sonnet-5-5")
_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")
_DATA_URL_RE = re.compile(r"^data:(?P<mt>[\w/+.-]+);base64,(?P<data>.+)$", re.DOTALL)


class AnthropicBrain(Brain):
    provider = "anthropic"

    def __init__(self, cfg: Settings) -> None:
        super().__init__(cfg)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise BrainError("Для JARVIS_PROVIDER=anthropic установите пакет 'anthropic' (make sync)") from exc

        key = cfg.api_key if cfg.api_key and cfg.api_key != "local-key" else None
        # key=None: SDK сам найдёт ANTHROPIC_API_KEY / профиль `ant auth login`.
        self.client = anthropic.AsyncAnthropic(
            api_key=key,
            base_url=cfg.base_url,
            timeout=anthropic.Timeout(max(cfg.llm_timeout, 120.0), connect=CONNECT_TIMEOUT),
            max_retries=cfg.llm_max_retries,
        )
        self._anthropic = anthropic
        self.adaptive = self.model.startswith(_ADAPTIVE_THINKING_PREFIXES)
        self.server_fallback = (
            self.model.startswith(_SERVER_FALLBACK_PREFIXES)
            and "api.anthropic.com" in cfg.base_url
        )

    # ---- конвертация истории

    @staticmethod
    def _tool_id(raw: str) -> str:
        return _TOOL_ID_RE.sub("_", raw or "call") or "call"

    @staticmethod
    def _user_blocks(content: Any) -> Any:
        if isinstance(content, str):
            return content
        blocks: list[dict[str, Any]] = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                blocks.append({"type": "text", "text": str(part.get("text", ""))})
            elif part.get("type") == "image_url":
                url = str((part.get("image_url") or {}).get("url", ""))
                match = _DATA_URL_RE.match(url)
                if match:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": match["mt"], "data": match["data"]},
                        }
                    )
                elif url:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        return blocks or "(пусто)"

    def to_anthropic(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            if pending_results:
                # Все tool_result одного шага — в одном user-сообщении.
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_parts.append(str(msg.get("content", "")))
                continue
            if role == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": self._tool_id(str(msg.get("tool_call_id", ""))),
                        "content": str(msg.get("content", "")),
                    }
                )
                continue
            flush_results()
            if role == "user":
                out.append({"role": "user", "content": self._user_blocks(msg.get("content"))})
            elif role == "assistant":
                raw = msg.get("_anthropic")
                if raw and msg.get("_model") == self.model:
                    # Дословный повтор ответа (вместе с thinking-блоками).
                    out.append({"role": "assistant", "content": raw})
                    continue
                blocks: list[dict[str, Any]] = []
                text = str(msg.get("content") or "").strip()
                if text:
                    blocks.append({"type": "text", "text": text})
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": self._tool_id(str(call.get("id", ""))),
                            "name": str(fn.get("name", "")),
                            "input": args if isinstance(args, dict) else {},
                        }
                    )
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "…"}]})
        flush_results()
        return "\n\n".join(p for p in system_parts if p), out

    @staticmethod
    def to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        result = []
        for item in tools or []:
            fn = item.get("function", {})
            schema = fn.get("parameters") or {"type": "object", "properties": {}}
            result.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", "") or "",
                    "input_schema": schema,
                }
            )
        return result

    # ---- запрос

    async def chat(self, messages, tools=None) -> BrainReply:
        a = self._anthropic
        system, converted = self.to_anthropic(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.cfg.llm_max_tokens,
            "messages": converted,
            # Автокэш последнего блока: системный промпт и инструменты повторяются каждый шаг.
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            params["system"] = system
        anthropic_tools = self.to_anthropic_tools(tools)
        if anthropic_tools:
            params["tools"] = anthropic_tools
        if self.cfg.llm_effort:
            params["output_config"] = {"effort": self.cfg.llm_effort}

        betas: list[str] = []
        if self.adaptive:
            # Агент правит свой системный промпт и набор инструментов — без drop_block
            # старые thinking-блоки давали бы 400 «bound to a different conversation».
            params["thinking"] = {
                "type": "adaptive",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
            betas.append("thinking-binding-controls-2026-08-01")
        if self.server_fallback:
            params["fallbacks"] = "default"
            betas.append("server-side-fallback-2026-07-01")

        try:
            if betas:
                response = await self.client.beta.messages.create(betas=betas, **params)
            else:
                response = await self.client.messages.create(**params)
        except a.APITimeoutError as exc:
            raise BrainError(f"Мозг не ответил вовремя ({self.label()})") from exc
        except a.APIConnectionError as exc:
            raise BrainError(f"Нет связи с мозгом ({self.label()})") from exc
        except a.AuthenticationError as exc:
            raise BrainError(f"Мозг отклонил API-ключ ({self.label()}). Задайте ANTHROPIC_API_KEY или JARVIS_API_KEY") from exc
        except a.RateLimitError as exc:
            raise BrainError(f"Превышен лимит запросов к мозгу ({self.label()}), попробуйте позже") from exc
        except a.APIStatusError as exc:
            raise BrainError(f"Мозг вернул HTTP {exc.status_code} ({self.label()}): {exc.message}") from exc

        if response.stop_reason == "refusal":
            return BrainReply(content="Я не могу помочь с этим запросом.")

        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": json.dumps(block.input, ensure_ascii=False)},
                    }
                )
        raw = [block.to_dict() for block in response.content]
        return BrainReply(
            content="\n".join(t for t in texts if t).strip(),
            tool_calls=tool_calls,
            extra={"_anthropic": raw, "_model": self.model},
        )


def make_brain(cfg: Settings | None = None) -> Brain:
    cfg = cfg or settings
    if cfg.provider == "anthropic":
        return AnthropicBrain(cfg)
    return OpenAICompatBrain(cfg)
