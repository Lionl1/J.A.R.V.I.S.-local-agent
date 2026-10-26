from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jarvis_backend.agent import AgentCore
from jarvis_backend.brain import AnthropicBrain, Brain, BrainReply
from jarvis_backend.config import settings


class ScriptedBrain(Brain):
    provider = "fake"

    def __init__(self, replies: list[BrainReply]) -> None:
        self.model = "fake"
        self.base_url = "memory"
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    async def chat(self, messages, tools=None) -> BrainReply:
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


def _call(name: str, args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.mark.asyncio
async def test_turn_runs_tools_then_answers_and_persists(tmp_path: Path) -> None:
    executed: list[tuple[str, dict]] = []
    events: list[str] = []

    async def executor(name: str, args: dict) -> str:
        executed.append((name, args))
        return '{"ok": true, "result": "12:00"}'

    async def on_event(kind: str, payload: dict) -> None:
        events.append(kind)

    brain = ScriptedBrain(
        [
            BrainReply(content="", tool_calls=[_call("get_current_time", {})]),
            BrainReply(content="Сейчас полдень."),
        ]
    )
    history = tmp_path / "history.json"
    agent = AgentCore(brain, tool_executor=executor, history_path=history)

    answer = await agent.run_turn("который час?", on_event=on_event)

    assert answer == "Сейчас полдень."
    assert executed == [("get_current_time", {})]
    assert events == ["tool_call", "tool_result"]
    assert [m["role"] for m in brain.seen[1]] == ["system", "user", "assistant", "tool"]

    restored = AgentCore(ScriptedBrain([]), history_path=history)
    assert [m["role"] for m in restored.messages] == ["system", "user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_interrupted_turn_is_repaired_and_resumed(tmp_path: Path) -> None:
    history = tmp_path / "history.json"
    # Процесс умер после tool_call, до записи результата.
    history.write_text(
        json.dumps(
            [
                {"role": "user", "content": "обнови навык"},
                {"role": "assistant", "content": "", "tool_calls": [_call("edit_source_code", {}, "c1")]},
            ]
        ),
        encoding="utf-8",
    )
    brain = ScriptedBrain([BrainReply(content="Готово, навык обновлён.")])
    agent = AgentCore(brain, history_path=history)

    assert agent.has_interrupted_turn()
    answer = await agent.resume_turn()

    assert answer == "Готово, навык обновлён."
    roles = [m["role"] for m in brain.seen[0]]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assert not agent.has_interrupted_turn()


def test_trim_keeps_whole_turns_from_the_front() -> None:
    agent = AgentCore(ScriptedBrain([]), max_history=6)
    for i in range(4):
        agent.messages += [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": "", "tool_calls": [_call("t", {}, f"c{i}")]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "r"},
            {"role": "assistant", "content": f"a{i}"},
        ]

    agent._trim_memory()

    assert agent.messages[0]["role"] == "system"
    assert agent.messages[1] == {"role": "user", "content": "q3"}
    assert len(agent.messages) == 5


@pytest.mark.asyncio
async def test_json_action_fallback_for_models_without_function_calling() -> None:
    executed: list[str] = []

    async def executor(name: str, args: dict) -> str:
        executed.append(name)
        return "ok"

    brain = ScriptedBrain(
        [
            BrainReply(content='{"thought": "", "speech": "", "action": {"tool": "get_current_time", "args": {}}}'),
            BrainReply(content='{"speech": "Полдень.", "action": null}'),
        ]
    )
    agent = AgentCore(brain, tool_executor=executor)

    assert await agent.run_turn("время?") == "Полдень."
    assert executed == ["get_current_time"]


def test_history_conversion_for_claude() -> None:
    brain = AnthropicBrain(replace(settings, provider="anthropic", model="claude-opus-5-5", api_key="test"))
    system, messages = brain.to_anthropic(
        [
            {"role": "system", "content": "rules"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "что на картинке?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {
                "role": "assistant",
                "content": "смотрю",
                "tool_calls": [_call("a", {"x": 1}, "call:1"), _call("b", {}, "call_2")],
            },
            {"role": "tool", "tool_call_id": "call:1", "content": "ra"},
            {"role": "tool", "tool_call_id": "call_2", "content": "rb"},
            {"role": "assistant", "content": "готово"},
        ]
    )

    assert system == "rules"
    assert messages[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }
    assert messages[1]["content"][1] == {"type": "tool_use", "id": "call_1", "name": "a", "input": {"x": 1}}
    # Оба результата одного шага — в одном user-сообщении, id совпадают с tool_use.
    assert [b["tool_use_id"] for b in messages[2]["content"]] == ["call_1", "call_2"]
    assert messages[3] == {"role": "assistant", "content": [{"type": "text", "text": "готово"}]}

    tools = brain.to_anthropic_tools(
        [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}]
    )
    assert tools == [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
