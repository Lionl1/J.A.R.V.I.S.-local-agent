from __future__ import annotations

import asyncio
import json

from .agent import AgentCore
from .audio import listen, speak
from .config import settings
from .tools import execute_tool

# Режим отладки: True -> только текстовый ввод/вывод без микрофона и TTS.
DEBUG_TEXT_MODE = settings.debug_text_mode

MAX_TOOL_STEPS = settings.max_tool_steps


async def _read_user_command() -> str:
    """Получает команду пользователя в debug или voice-режиме."""
    if DEBUG_TEXT_MODE:
        # input() блокирующий, поэтому выносим в thread через to_thread.
        return (await asyncio.to_thread(input, "Вы> ")).strip()

    try:
        return (await listen()).strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[STT ERROR] {exc}")
        return ""


async def _deliver_speech(text: str) -> None:
    """Отдаёт финальный ответ: print в debug-режиме или TTS в voice-режиме."""
    normalized = (text or "").strip()
    if not normalized:
        return

    if DEBUG_TEXT_MODE:
        print(f"JARVIS> {normalized}")
        return

    try:
        await speak(normalized)
    except Exception as exc:  # noqa: BLE001
        print(f"[TTS ERROR] {exc}")


async def _run_agent_turn(agent: AgentCore, user_text: str) -> str:
    """Запускает один агентский turn с возможными tool-call итерациями.

    Поток:
    1) Добавляем реплику пользователя в контекст.
    2) Запрашиваем LLM и парсим JSON-решение.
    3) Если есть action: выполняем инструмент, отправляем Observation обратно в LLM.
    4) Повторяем до финального ответа без action.
    """
    agent.messages.append({"role": "user", "content": user_text})

    for step in range(1, MAX_TOOL_STEPS + 1):
        raw = await agent._ask_llm(agent.messages)
        decision = agent._parse_decision(raw)

        if decision.action is None:
            # Финальный ответ ассистента: фиксируем в истории и возвращаем наружу.
            agent.messages.append(
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
            return decision.speech

        # Выполняем инструмент вне AgentCore (по требованиям orchestration слоя).
        observation = await execute_tool(decision.action.tool, decision.action.args)

        # Сохраняем шаг ассистента с action и observation в контексте.
        agent.messages.append(
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

        # Отправляем Observation как новый пользовательский сигнал для следующего шага.
        agent.messages.append(
            {
                "role": "user",
                "content": (
                    "Observation from tool execution: "
                    f"{json.dumps(observation, ensure_ascii=False)}\n"
                    "Если задача выполнена, верни финальный ответ пользователю в поле speech "
                    "и action=null."
                ),
            }
        )

        if DEBUG_TEXT_MODE:
            print(
                "[TOOL] "
                f"step={step} tool={decision.action.tool} args={decision.action.args} "
                f"observation={observation}"
            )

    # Если модель зациклилась на tools, принудительно завершаем turn.
    fallback = "Превышен лимит шагов инструментов. Уточни задачу, и я продолжу."
    agent.messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "thought": "Stopped due to MAX_TOOL_STEPS",
                    "action": None,
                    "speech": fallback,
                },
                ensure_ascii=False,
            ),
        }
    )
    return fallback


async def main() -> None:
    """Точка входа в бесконечный Agentic Loop."""
    # Конфиг агента берётся из env внутри AgentCore.
    agent = AgentCore()

    while True:
        print("Ожидание команды...")
        user_text = await _read_user_command()

        if not user_text:
            continue

        # Простая команда выхода для debug-режима и консольного запуска.
        if user_text.lower() in {"exit", "quit", "выход"}:
            print("Завершение работы.")
            return

        try:
            final_speech = await _run_agent_turn(agent, user_text)
        except Exception as exc:  # noqa: BLE001
            print(f"[AGENT ERROR] {type(exc).__name__}: {exc}")
            final_speech = f"Произошла ошибка в агенте: {exc}"

        await _deliver_speech(final_speech)


if __name__ == "__main__":
    asyncio.run(main())
