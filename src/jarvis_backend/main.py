from __future__ import annotations

import asyncio
import json
import os
import sys

from . import tools
from .agent import AgentCore
from .audio import listen, speak
from .brain import BrainError
from .config import settings
from .mcp_manager import MCPManager
from .wake import VoiceListener

# Режим отладки: True -> только текстовый ввод/вывод без микрофона и TTS.
DEBUG_TEXT_MODE = settings.debug_text_mode



async def _confirm_in_terminal(tool_name: str, args: dict, details: dict) -> bool:
    """Подтверждение опасного инструмента в консоли (и в voice-режиме тоже — голосом не разрешаем)."""
    print(f"\n[ПОДТВЕРЖДЕНИЕ] {details.get('summary', tool_name)}")
    if details.get("diff"):
        print(details["diff"])
    if details.get("files"):
        print("Файлы: " + ", ".join(details["files"]))
    answer = await asyncio.to_thread(input, "Разрешить? [y/N] ")
    return answer.strip().lower() in {"y", "yes", "д", "да"}


# Фоновое прослушивание «Джарвис, …» (JARVIS_WAKE_WORD=true) вместо реакции на любую фразу.
_voice_listener: VoiceListener | None = None
_voice_commands: asyncio.Queue[str] = asyncio.Queue()


async def _read_user_command() -> str:
    """Получает команду пользователя в debug или voice-режиме."""
    if DEBUG_TEXT_MODE:
        # input() блокирующий, поэтому выносим в thread через to_thread.
        return (await asyncio.to_thread(input, "Вы> ")).strip()

    if _voice_listener is not None:
        return (await _voice_commands.get()).strip()

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

    print(f"JARVIS> {normalized}")
    try:
        if _voice_listener is not None:
            async with _voice_listener.muted():
                await speak(normalized)
        else:
            await speak(normalized)
    except Exception as exc:  # noqa: BLE001
        print(f"[TTS ERROR] {exc}")


async def _print_tool_event(kind: str, payload: dict) -> None:
    if not DEBUG_TEXT_MODE:
        return
    if kind == "tool_call":
        print(f"[TOOL] {payload['name']} {json.dumps(payload['args'], ensure_ascii=False)[:300]}")
    else:
        print(f"[TOOL RESULT] {str(payload['result'])[:300]}")


def _restart_process() -> None:
    """Ядро изменено агентом: перезапускаемся, история уже сохранена на диске."""
    print("[JARVIS] Код ядра изменён — перезапуск…", flush=True)
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])


async def main() -> None:
    """Точка входа в бесконечный Agentic Loop."""
    mcp_manager = MCPManager()

    async def all_tools() -> list[dict]:
        return tools.get_tool_schemas() + await mcp_manager.get_all_tools()

    async def route(tool_name: str, args: dict) -> str:
        if tools.is_local_tool(tool_name):
            result = await tools.execute_tool(tool_name, args)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return await mcp_manager.execute_tool(tool_name, args)

    agent = AgentCore(
        tools_provider=all_tools,
        tool_executor=route,
        history_path=settings.state_dir / "history.json",
    )
    print(f"Мозг: {agent.brain.label()}")
    tools.set_confirmation_callback(_confirm_in_terminal)
    tools.set_rules_changed_callback(agent.reload_system_prompt)
    tools.set_proactive_callback(_deliver_speech)
    await mcp_manager.start()

    global _voice_listener
    if settings.wake_word and not DEBUG_TEXT_MODE:
        _voice_listener = VoiceListener(_voice_commands.put)
        await _voice_listener.start()
        print("Скажите «Джарвис, …»")

    try:
        if agent.has_interrupted_turn():
            await _deliver_speech(await agent.resume_turn(on_event=_print_tool_event))

        while True:
            print("Ожидание команды...")
            user_text = await _read_user_command()

            if not user_text:
                continue

            # Простая команда выхода для debug-режима и консольного запуска.
            if user_text.lower() in {"exit", "quit", "выход"}:
                print("Завершение работы.")
                return
            if user_text.lower() in {"/reset", "сброс"}:
                agent.reset()
                print("История очищена.")
                continue

            try:
                final_speech = await agent.run_turn(user_text, on_event=_print_tool_event)
            except BrainError as exc:
                final_speech = str(exc)
            except Exception as exc:  # noqa: BLE001
                print(f"[AGENT ERROR] {type(exc).__name__}: {exc}")
                final_speech = f"Произошла ошибка в агенте: {exc}"

            await _deliver_speech(final_speech)

            if tools.restart_required:
                if _voice_listener is not None:
                    await _voice_listener.stop()
                await mcp_manager.close()
                _restart_process()
    finally:
        if _voice_listener is not None:
            await _voice_listener.stop()
        await mcp_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
