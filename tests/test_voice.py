from __future__ import annotations

import plistlib

import numpy as np
import pytest

from jarvis_backend import audio, autostart, wake
from jarvis_backend.wake import FRAME_SAMPLES, SpeechSegmenter, VoiceListener, match_wake


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Джарвис, который час?", "который час?"),
        ("Эй, Джарвис, открой калькулятор.", "открой калькулятор."),
        ("Hey Jarvis, what's the weather?", "what's the weather?"),
        ("Джарвис!", ""),
        ("джарвис", ""),
        ("Слушай, Джарвис, поставь таймер", "поставь таймер"),
        ("Сегодня хорошая погода", None),
        ("Я вчера смотрел фильм, где был Джарвис", None),
        ("", None),
    ],
)
def test_match_wake(text: str, expected: str | None) -> None:
    assert match_wake(text) == expected


def _tone(seconds: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def _noise(seconds: float, amplitude: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, amplitude, int(16000 * seconds)).astype(np.int16)


def test_segmenter_splits_phrases_and_skips_clicks() -> None:
    stream = np.concatenate(
        [
            _noise(1.0, 40),
            _tone(1.0, 4000),  # фраза 1
            _noise(1.2, 40, 1),
            _tone(0.06, 4000),  # щелчок короче min_speech — не фраза
            _noise(1.2, 40, 2),
            _tone(0.8, 4000),  # фраза 2
            _noise(1.2, 40, 3),
        ]
    )
    segmenter = SpeechSegmenter()
    segments = []
    for k in range(0, len(stream) - FRAME_SAMPLES, FRAME_SAMPLES):
        out = segmenter.feed(stream[k : k + FRAME_SAMPLES])
        if out is not None:
            segments.append(len(out) / 16000)

    assert len(segments) == 2
    assert 1.0 <= segments[0] <= 2.2
    assert 0.8 <= segments[1] <= 2.0


async def _collect(listener_kwargs: dict, phrases: list[str]) -> tuple[list[str], int]:
    commands: list[str] = []
    wakes: list[int] = []

    async def on_command(text: str) -> None:
        commands.append(text)

    async def on_wake() -> None:
        wakes.append(1)

    listener = VoiceListener(on_command, on_wake=on_wake, **listener_kwargs)
    for phrase in phrases:
        await listener.handle_text(phrase)
    return commands, len(wakes)


@pytest.mark.asyncio
async def test_listener_dispatches_only_addressed_phrases(monkeypatch) -> None:
    monkeypatch.setattr(wake, "_play_chime", lambda: None)

    commands, wakes = await _collect(
        {"engine": "whisper"},
        [
            "Сегодня хорошая погода",
            "Джарвис, который час?",
            "Джарвис!",
            "Открой калькулятор",
            "А это уже просто разговор",
        ],
    )

    assert commands == ["который час?", "Открой калькулятор"]
    assert wakes == 1


@pytest.mark.asyncio
async def test_openwakeword_mode_ignores_phrases_without_wake(monkeypatch) -> None:
    monkeypatch.setattr(wake, "_play_chime", lambda: None)

    commands, _ = await _collect({"engine": "openwakeword"}, ["Джарвис, который час?"])

    assert commands == []


def test_second_pass_with_name_hint_only_when_needed() -> None:
    calls: list[bool] = []

    def fake_transcribe(segment: np.ndarray, hint: bool) -> str:
        calls.append(hint)
        return "Джарвис" if hint else "джермес"

    async def noop(_: str) -> None:
        return None

    listener = VoiceListener(noop, engine="whisper", transcribe=fake_transcribe)
    assert listener._recognize(np.zeros(10, np.int16)) == "Джарвис"
    assert calls == [False, True]

    calls.clear()
    listener._transcribe = lambda segment, hint: calls.append(hint) or "Джарвис, время"
    assert listener._recognize(np.zeros(10, np.int16)) == "Джарвис, время"
    assert calls == [False]


@pytest.mark.asyncio
async def test_tts_falls_back_to_system_voice_and_backs_off(monkeypatch) -> None:
    edge_calls: list[str] = []

    async def broken_edge(text, voice, delays, timeout):
        edge_calls.append(text)
        raise RuntimeError("NoAudioReceived")

    async def system(text):
        return b"RIFF....WAVE"

    monkeypatch.setattr(audio, "TTS_ENGINE", "auto")
    monkeypatch.setattr(audio, "_edge_disabled_until", 0.0)
    monkeypatch.setattr(audio, "_edge_tts_bytes", broken_edge)
    monkeypatch.setattr(audio, "_system_tts_bytes", system)

    assert await audio.synthesize_speech_bytes("раз") == b"RIFF....WAVE"
    assert await audio.synthesize_speech_bytes("два") == b"RIFF....WAVE"
    # После первого сбоя edge-tts не дёргаем — вторая фраза сразу системным голосом.
    assert edge_calls == ["раз"]


def test_launch_agent_plist_runs_server_from_project() -> None:
    plist = autostart.build_plist()
    plistlib.dumps(plist)  # сериализуется

    command = " ".join(plist["ProgramArguments"])
    assert plist["Label"] == autostart.LABEL
    assert plist["WorkingDirectory"] == str(autostart.PROJECT_ROOT)
    assert "src.jarvis_backend.server:app" in command
    assert "127.0.0.1" in command
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert "/opt/homebrew/bin" in plist["EnvironmentVariables"]["PATH"]


def test_systemd_unit_restarts_on_failure() -> None:
    unit = autostart.build_unit()

    assert "Restart=on-failure" in unit
    assert "src.jarvis_backend.server:app" in unit
    assert f"WorkingDirectory={autostart.PROJECT_ROOT}" in unit


def test_bundle_command_uses_only_project_packages(tmp_path) -> None:
    command = autostart._bundle_command(tmp_path / "Jarvis")
    if command is None:
        pytest.skip("нет .venv")
    assert command[1] == "-S"  # без глобальных site-packages системного Python
    assert "site.addsitedir" in command[3] and ".venv" in command[3]
    compile(command[3], "<launcher>", "exec")
