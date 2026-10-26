from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .brain import Brain, make_brain
from .config import settings

FALLBACK_SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., a local AI assistant. "
    "Respond concisely and politely. "
    "Use tools only when necessary. "
    "Do not use markdown in your final verbal replies. "
    "Automatically detect the user's language and respond in the same language based on context."
)

ToolsProvider = Callable[[], Awaitable[list[dict[str, Any]]]]
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]
EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


def _get_rules_path() -> Path:
    return settings.rules_path.resolve()


def _load_system_prompt() -> str:
    rules_path = _get_rules_path()
    try:
        text = rules_path.read_text(encoding="utf-8").strip()
        if text:
            return text
        print(f"[WARN] Rules file is empty: {rules_path}. Using fallback prompt.")
    except FileNotFoundError:
        print(f"[WARN] Rules file not found: {rules_path}. Using fallback prompt.")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to load rules from {rules_path}: {exc}. Using fallback prompt.")
    return FALLBACK_SYSTEM_PROMPT


def parse_tool_arguments(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        payload = json.loads(arguments)
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass(slots=True)
class ActionCall:
    tool: str
    args: dict[str, Any]


@dataclass(slots=True)
class AgentDecision:
    thought: str
    speech: str
    action: ActionCall | None
    raw_text: str


class AgentCore:
    """Мозг (любой LLM) + история + цикл вызова инструментов.

    Один и тот же цикл используется веб-сервером и консольным режимом.
    История сохраняется на диск, поэтому переживает перезапуск после самоправки кода.
    """

    def __init__(
        self,
        brain: Brain | None = None,
        *,
        tools_provider: ToolsProvider | None = None,
        tool_executor: ToolExecutor | None = None,
        history_path: Path | None = None,
        max_tool_steps: int | None = None,
        max_history: int | None = None,
    ) -> None:
        self.brain = brain or make_brain()
        self.tools_provider = tools_provider
        self.tool_executor = tool_executor
        self.max_tool_steps = max_tool_steps or settings.max_tool_steps
        self.max_history = max_history or settings.max_history_messages
        self.history_path = history_path
        self.system_prompt = _load_system_prompt()
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        self._load_history()

    @property
    def model(self) -> str:
        return self.brain.model

    # ---- история

    def reload_system_prompt(self) -> None:
        """Подхватить отредактированный jarvis_rules.md без перезапуска."""
        self.system_prompt = _load_system_prompt()
        self.messages[0] = {"role": "system", "content": self.system_prompt}

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.save_history()

    def _load_history(self) -> None:
        if self.history_path is None or not self.history_path.exists():
            return
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load history {self.history_path}: {exc}")
            return
        if isinstance(data, list):
            self.messages.extend(m for m in data if isinstance(m, dict) and m.get("role") != "system")
            self._repair_history()
            self._trim_memory()

    def save_history(self) -> None:
        if self.history_path is None:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.history_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps([_strip_images(m) for m in self.messages[1:]], ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.history_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to save history: {exc}")

    def _trim_memory(self, max_messages: int | None = None) -> None:
        """Удаляет самые старые целые ходы (user + ответы + tool), сохраняя system.

        Режем только с начала и только целыми ходами: цепочки tool_call остаются целыми,
        а у Claude это не ломает подписи thinking-блоков.
        """
        limit = max(2, max_messages or self.max_history)
        if len(self.messages) <= limit:
            return

        user_indexes = [i for i in range(1, len(self.messages)) if self.messages[i].get("role") == "user"]
        cut = None
        for idx in user_indexes[1:]:
            cut = idx
            if 1 + len(self.messages) - idx <= limit:
                break
        if cut is not None:
            del self.messages[1:cut]

        if len(self.messages) > limit:
            # Один гигантский ход: оставляем хвост, начиная с ближайшего user-сообщения.
            tail = self.messages[-(limit - 1):]
            while tail and tail[0].get("role") != "user":
                tail.pop(0)
            self.messages = [self.messages[0], *tail]

    # ---- цикл агента

    async def run_turn(
        self,
        user_content: str | list[dict[str, Any]],
        on_event: EventCallback | None = None,
    ) -> str:
        """Один ход: реплика пользователя -> (инструменты)* -> финальный ответ."""
        self._trim_memory()
        self.messages.append({"role": "user", "content": user_content})
        try:
            return await self._loop(on_event)
        finally:
            self.save_history()

    def has_interrupted_turn(self) -> bool:
        """История заканчивается результатом инструмента — процесс перезапустился посреди хода."""
        return len(self.messages) > 1 and self.messages[-1].get("role") == "tool"

    async def resume_turn(self, on_event: EventCallback | None = None) -> str:
        """Довести прерванный ход до финального ответа (например, после самоправки ядра)."""
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "[Система] Процесс был перезапущен во время выполнения инструментов "
                    "(например, после изменения твоего кода). Продолжи задачу с того места, "
                    "где остановился, и кратко сообщи пользователю результат."
                ),
            }
        )
        try:
            return await self._loop(on_event)
        except Exception as exc:  # noqa: BLE001
            text = f"Не удалось продолжить прерванную задачу: {exc}"
            self.messages.append({"role": "assistant", "content": text})
            return text
        finally:
            self.save_history()

    def _repair_history(self) -> None:
        """Каждый tool_call должен иметь ответ, иначе API отвергнет историю."""
        repaired: list[dict[str, Any]] = []
        pending: list[str] = []
        for msg in self.messages:
            if msg.get("role") == "tool":
                call_id = str(msg.get("tool_call_id", ""))
                if call_id in pending:
                    pending.remove(call_id)
                    repaired.append(msg)
                continue
            for call_id in pending:
                repaired.append({"role": "tool", "tool_call_id": call_id, "content": "Результат утерян: процесс был перезапущен."})
            pending = []
            repaired.append(msg)
            if msg.get("role") == "assistant":
                pending = [str(c.get("id", "")) for c in msg.get("tool_calls") or []]
        for call_id in pending:
            repaired.append({"role": "tool", "tool_call_id": call_id, "content": "Результат утерян: процесс был перезапущен."})
        self.messages = repaired

    async def _loop(self, on_event: EventCallback | None) -> str:
        for step in range(1, self.max_tool_steps + 1):
            tools = await self.tools_provider() if self.tools_provider else []
            reply = await self.brain.chat(self.messages, tools=tools)

            if not reply.tool_calls:
                # Совместимость: модели без function calling иногда отвечают JSON {speech, action}.
                decision = self._parse_decision(reply.content) if reply.content.lstrip().startswith(("{", "```")) else None
                if decision and decision.action is not None and self.tool_executor is not None:
                    call_id = f"json_{step}"
                    reply.tool_calls = [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": decision.action.tool,
                                "arguments": json.dumps(decision.action.args, ensure_ascii=False),
                            },
                        }
                    ]
                else:
                    speech = (decision.speech if decision else reply.content) or "Готово."
                    self.messages.append({"role": "assistant", "content": reply.content or speech, **reply.extra})
                    return speech

            self.messages.append(
                {"role": "assistant", "content": reply.content, "tool_calls": reply.tool_calls, **reply.extra}
            )
            for call in reply.tool_calls:
                fn = call.get("function", {})
                name = str(fn.get("name", "")).strip()
                args = parse_tool_arguments(fn.get("arguments"))
                call_id = str(call.get("id", "")).strip() or f"call_{step}"
                if on_event:
                    await on_event("tool_call", {"name": name, "args": args, "id": call_id})
                if self.tool_executor is None:
                    result = json.dumps({"ok": False, "error": "Инструменты недоступны"}, ensure_ascii=False)
                else:
                    result = await self.tool_executor(name, args)
                if on_event:
                    await on_event("tool_result", {"name": name, "result": result, "id": call_id})
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
                # Сохраняем после каждого инструмента: правка кода может перезапустить процесс.
                self.save_history()

        fallback = "Превышен лимит шагов инструментов. Уточни задачу, и я продолжу."
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback

    def _parse_decision(self, text: str) -> AgentDecision:
        payload = self._safe_extract_json(text)

        if not isinstance(payload, dict):
            return AgentDecision(
                thought="Failed to parse JSON response",                
                speech=text.strip() or "Не удалось распознать ответ модели.",
                action=None,
                raw_text=text,
            )

        thought = str(payload.get("thought", "")).strip()
        speech = str(payload.get("speech", "")).strip()
        action = self._parse_action(payload.get("action"))

        if not speech:
            speech = "Готово."

        return AgentDecision(
            thought=thought,
            speech=speech,
            action=action,
            raw_text=text,
        )

    def _parse_action(self, action_data: Any) -> ActionCall | None:
        if not action_data:
            return None

        if isinstance(action_data, str):
            # Allow shorthand action="get_current_time"
            action_data = {"tool": action_data, "args": {}}

        if not isinstance(action_data, dict):
            return None

        tool_name = str(action_data.get("tool", "")).strip()
        if not tool_name:
            return None

        args = action_data.get("args", {})
        if not isinstance(args, dict):
            args = {}

        return ActionCall(tool=tool_name, args=args)

    def _safe_extract_json(self, text: str) -> Any:
        if not text:
            return None

        candidates = self._extract_json_candidates(text)
        for candidate in candidates:
            parsed = self._try_parse_json(candidate)
            if parsed is not None:
                return parsed

        return None

    def _extract_json_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        # 1) JSON fenced block
        fenced = re.findall(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL)
        candidates.extend(fenced)

        # 2) Balanced braces scan (robust against extra prose around JSON)
        depth = 0
        start = -1
        for idx, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidates.append(text[start : idx + 1])
                        start = -1

        # 3) Full text as last resort
        candidates.append(text)

        # Deduplicate while preserving order
        unique: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            key = item.strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(key)

        return unique

    def _try_parse_json(self, candidate: str) -> Any:
        candidate = candidate.strip()
        if not candidate:
            return None

        # Attempt 1: strict JSON
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Attempt 2: minor cleanup (trailing commas, smart quotes)
        normalized = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
        normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            pass

        # Attempt 3: Python-like dict fallback
        try:
            value = ast.literal_eval(normalized)
            if isinstance(value, (dict, list)):
                return value
        except (SyntaxError, ValueError):
            return None

        return None


def _strip_images(message: dict[str, Any]) -> dict[str, Any]:
    """Не тащим base64-картинки в файл истории."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            parts.append({"type": "text", "text": "[изображение]"})
        else:
            parts.append(part)
    return {**message, "content": parts}
