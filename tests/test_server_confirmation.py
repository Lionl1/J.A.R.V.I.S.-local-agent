from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jarvis_backend import server


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


@pytest.fixture()
def ws(monkeypatch: pytest.MonkeyPatch) -> FakeWebSocket:
    socket = FakeWebSocket()
    monkeypatch.setattr(server, "active_websocket", socket)
    monkeypatch.setattr(server, "pending_confirmations", {})
    return socket


async def _wait_for_request(ws: FakeWebSocket) -> dict[str, Any]:
    for _ in range(100):
        requests = [m for m in ws.sent if m["type"] == "confirm_request"]
        if requests:
            return requests[-1]["payload"]
        await asyncio.sleep(0.01)
    raise AssertionError("confirm_request was not sent")


@pytest.mark.asyncio
async def test_approval_from_same_socket(ws: FakeWebSocket) -> None:
    task = asyncio.create_task(
        server.request_confirmation(
            "update_source_code",
            {"file_name": "tools.py", "new_content": "x" * 10_000},
            {"summary": "Изменить файл", "diff": "+x"},
        )
    )
    request = await _wait_for_request(ws)
    # Полный new_content не уходит в браузер — только diff.
    assert "new_content" not in request["args"]
    assert request["diff"] == "+x"

    server._resolve_confirmation(ws, {"id": request["id"], "approved": True})

    assert await task is True
    assert server.pending_confirmations == {}


@pytest.mark.asyncio
async def test_answer_from_other_socket_is_ignored(ws: FakeWebSocket, monkeypatch) -> None:
    monkeypatch.setattr(server, "CONFIRM_TIMEOUT", 0.2)
    task = asyncio.create_task(server.request_confirmation("open_application", {}, {"summary": "s"}))
    request = await _wait_for_request(ws)

    server._resolve_confirmation(FakeWebSocket(), {"id": request["id"], "approved": True})

    assert await task is False


@pytest.mark.asyncio
async def test_only_strict_true_approves(ws: FakeWebSocket) -> None:
    task = asyncio.create_task(server.request_confirmation("open_application", {}, {"summary": "s"}))
    request = await _wait_for_request(ws)

    server._resolve_confirmation(ws, {"id": request["id"], "approved": "yes"})

    assert await task is False


@pytest.mark.asyncio
async def test_timeout_denies_and_notifies(ws: FakeWebSocket, monkeypatch) -> None:
    monkeypatch.setattr(server, "CONFIRM_TIMEOUT", 0.05)

    approved = await server.request_confirmation("open_application", {}, {"summary": "s"})

    assert approved is False
    assert ws.sent[-1]["type"] == "confirm_closed"
    assert ws.sent[-1]["payload"]["reason"] == "timeout"


@pytest.mark.asyncio
async def test_disconnect_denies_pending(ws: FakeWebSocket) -> None:
    task = asyncio.create_task(server.request_confirmation("open_application", {}, {"summary": "s"}))
    await _wait_for_request(ws)

    server._deny_pending_confirmations(ws)

    assert await task is False


@pytest.mark.asyncio
async def test_no_client_means_deny(monkeypatch) -> None:
    monkeypatch.setattr(server, "active_websocket", None)

    assert await server.request_confirmation("open_application", {}, {"summary": "s"}) is False


def test_websocket_loop_receives_answer_while_turn_waits(monkeypatch) -> None:
    """Регрессия: turn агента ждёт подтверждения, а цикл приёма должен продолжать читать сокет."""
    from fastapi.testclient import TestClient

    from jarvis_backend import tools

    launched: list[str] = []

    async def fake_open(app_name: str) -> dict[str, Any]:
        launched.append(app_name)
        return {"ok": True, "result": f"opened {app_name}"}

    async def fake_turn(user_text: str) -> str:
        result = await tools.execute_tool("open_application", {"app_name": user_text})
        return "ok" if result["ok"] else "denied"

    async def no_tts(text: str, voice: str | None = None) -> str:
        return ""

    monkeypatch.setitem(tools.TOOL_REGISTRY, "open_application", fake_open)
    monkeypatch.setattr(server, "_run_agent_turn", fake_turn)
    monkeypatch.setattr(server, "synthesize_speech_base64", no_tts)
    monkeypatch.setattr(server, "CONFIRM_TIMEOUT", 5.0)
    monkeypatch.setattr(server, "pending_confirmations", {})
    tools.set_confirmation_callback(server.request_confirmation)
    try:
        client = TestClient(server.app)
        with client.websocket_connect(f"/ws/jarvis?token={server.API_KEY}") as ws:

            def next_of(kind: str) -> dict[str, Any]:
                while True:
                    message = ws.receive_json()
                    if message["type"] == kind:
                        return message["payload"]

            ws.send_json({"type": "text", "payload": "Calculator"})
            request = next_of("confirm_request")
            assert request["tool"] == "open_application"
            assert launched == []

            ws.send_json({"type": "confirm_response", "payload": {"id": request["id"], "approved": True}})
            assert next_of("speech_text") == "ok"
            assert launched == ["Calculator"]

            ws.send_json({"type": "text", "payload": "Terminal"})
            request = next_of("confirm_request")
            ws.send_json({"type": "confirm_response", "payload": {"id": request["id"], "approved": False}})
            assert next_of("speech_text") == "denied"
            assert launched == ["Calculator"]
    finally:
        tools.set_confirmation_callback(None)
