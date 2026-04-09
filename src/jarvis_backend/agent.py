from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

from .config import settings
from .tools import execute_tool

FALLBACK_SYSTEM_PROMPT = (
    "Ты J.A.R.V.I.S. Локальный ассистент. "
    "Отвечай кратко и вежливо. "
    "Используй инструменты только при необходимости. "
    "Финальные ответы формируй без markdown."
)


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
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = (model or settings.model).strip()
        self.base_url = (base_url or settings.base_url).strip().rstrip("/")
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.timeout = settings.llm_timeout if timeout is None else timeout
        self.system_prompt = _load_system_prompt()
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=(api_key or settings.api_key or "local-key"),
            timeout=self.timeout,
        )
        self.messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

    def _trim_memory(self, max_messages: int = 20) -> None:
        """Trim conversation history while preserving system prompt and turn integrity.

        Strategy:
        - keep system message at index 0 always;
        - remove oldest completed user turns first;
        - each removed turn includes user/assistant (+tool messages if any),
          which preserves tool_call chains.
        """
        if max_messages < 2:
            max_messages = 2

        if len(self.messages) <= max_messages:
            return

        # Build turn ranges from the first non-system message.
        turns: list[tuple[int, int]] = []
        start_idx: int | None = None
        for idx in range(1, len(self.messages)):
            role = str(self.messages[idx].get("role", ""))
            if role == "user":
                if start_idx is not None:
                    turns.append((start_idx, idx))
                start_idx = idx
        if start_idx is not None:
            turns.append((start_idx, len(self.messages)))

        # Remove oldest complete turns until size is under limit.
        removed = 0
        for start, end in turns:
            if len(self.messages) - removed <= max_messages:
                break
            # Shift by already removed count.
            adj_start = max(1, start - removed)
            adj_end = max(adj_start, end - removed)
            if adj_start < adj_end:
                del self.messages[adj_start:adj_end]
                removed += (adj_end - adj_start)

        # Fallback: if no turns were detected but still too large, keep tail.
        if len(self.messages) > max_messages:
            self.messages = [self.messages[0], *self.messages[-(max_messages - 1):]]

    async def run_turn(self, user_text: str) -> dict[str, Any]:
        self.messages.append({"role": "user", "content": user_text})

        raw_output = await self._ask_llm(self.messages)
        decision = self._parse_decision(raw_output)

        result: dict[str, Any] = {
            "thought": decision.thought,
            "speech": decision.speech,
            "action": None,
            "observation": None,
            "raw": decision.raw_text,
        }

        if decision.action:
            observation = await execute_tool(decision.action.tool, decision.action.args)
            result["action"] = {
                "tool": decision.action.tool,
                "args": decision.action.args,
            }
            result["observation"] = observation

            # Keep tool result in history so the next turn has context.
            self.messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "thought": decision.thought,
                            "action": {
                                "tool": decision.action.tool,
                                "args": decision.action.args,
                            },
                            "speech": decision.speech,
                            "observation": observation,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        else:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "thought": decision.thought,
                            "action": None,
                            "speech": decision.speech,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        return result

    async def _ask_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        return_message: bool = False,
    ) -> str | dict[str, Any]:
        # Apply memory trimming before each model request.
        self._trim_memory(max_messages=20)
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            try:
                response = await self.client.chat.completions.create(**payload)
            except APIStatusError as exc:
                # Некоторые OpenAI-compatible провайдеры/модели не поддерживают tools.
                # Делаем безопасный fallback на обычный chat completion.
                if tools and exc.status_code == 400:
                    fallback_payload = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                    }
                    response = await self.client.chat.completions.create(
                        **fallback_payload
                    )
                else:
                    raise
        except APITimeoutError as exc:
            raise RuntimeError(
                f"LLM timeout: endpoint={self.base_url}, model={self.model}"
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                f"LLM connection error: endpoint={self.base_url}. "
                "Проверьте доступность сервера и URL /v1."
            ) from exc
        except AuthenticationError as exc:
            raise RuntimeError(
                f"LLM auth error: endpoint={self.base_url}. "
                "Проверьте API key."
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"LLM HTTP error {exc.status_code}: endpoint={self.base_url}, model={self.model}"
            ) from exc

        message = response.choices[0].message
        if return_message:
            tool_calls: list[dict[str, Any]] = []
            if message.tool_calls:
                for call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": call.id,
                            "type": call.type,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                    )
            return {
                "content": (message.content or "").strip(),
                "tool_calls": tool_calls,
            }

        content = message.content
        if not content:
            return ""
        return content.strip()

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
