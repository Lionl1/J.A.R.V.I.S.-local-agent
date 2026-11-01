from __future__ import annotations

import asyncio
import base64
import os
import platform
import re
import shutil
import threading
import time
import tempfile
from pathlib import Path

import speech_recognition as sr
try:
    import edge_tts
except Exception:  # noqa: BLE001
    edge_tts = None  # type: ignore[assignment]

try:
    from faster_whisper import WhisperModel
except Exception:  # noqa: BLE001
    WhisperModel = None  # type: ignore[assignment,misc]

from .config import settings


DEFAULT_TTS_VOICE = settings.tts_voice
DEFAULT_STT_LANGUAGE = settings.stt_language
DEFAULT_WHISPER_MODEL = settings.whisper_model
DEFAULT_WHISPER_DEVICE = settings.whisper_device
DEFAULT_WHISPER_COMPUTE_TYPE = settings.whisper_compute_type


TTS_ENGINE = settings.tts_engine
SYSTEM_TTS_VOICE = settings.system_tts_voice
TTS_RETRY_DELAYS = (0.0, 0.6, 1.5, 3.0)
# После сбоя edge-tts не трогаем его какое-то время, чтобы каждая фраза не ждала таймаута.
EDGE_COOLDOWN_SECONDS = 300
_edge_disabled_until = 0.0

_whisper_model: WhisperModel | None = None
_whisper_model_lock = threading.Lock()


async def synthesize_speech_bytes(text: str, voice: str = DEFAULT_TTS_VOICE) -> bytes:
    """Синтезирует речь в аудио-байты (MP3 от edge-tts или WAV от системного голоса).

    TTS_ENGINE: auto — edge-tts, а при сбое офлайн-голос ОС; edge — только edge-tts;
    system — только голос ОС (say / espeak-ng / SAPI), работает без интернета.
    """
    normalized = (text or "").strip()
    if not normalized:
        return b""

    global _edge_disabled_until
    if TTS_ENGINE == "edge":
        return await _edge_tts_bytes(normalized, voice, TTS_RETRY_DELAYS, 20.0)
    if TTS_ENGINE != "system" and time.monotonic() >= _edge_disabled_until:
        try:
            # В режиме auto одна быстрая попытка: лучше сразу ответить системным голосом.
            return await _edge_tts_bytes(normalized, voice, (0.0,), 3.5)
        except Exception as exc:  # noqa: BLE001
            _edge_disabled_until = time.monotonic() + EDGE_COOLDOWN_SECONDS
            print(f"[TTS] edge-tts unavailable ({exc}); using system voice for {EDGE_COOLDOWN_SECONDS}s", flush=True)
    return await _system_tts_bytes(normalized)


async def _edge_tts_bytes(
    text: str, voice: str, delays: tuple[float, ...], attempt_timeout: float
) -> bytes:
    if edge_tts is None:
        raise RuntimeError("TTS dependency is missing: install package 'edge-tts'.")
    # Сервис Microsoft периодически отвечает без аудио (NoAudioReceived) — повторяем с паузой.
    last_error: Exception | None = None
    for delay in delays:
        if delay:
            await asyncio.sleep(delay)
        try:
            output = await asyncio.wait_for(_edge_tts_attempt(text, voice), timeout=attempt_timeout)
            if output:
                return output
            last_error = RuntimeError("TTS returned no audio")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"edge-tts failed after {len(delays)} attempts: {last_error}")


async def _edge_tts_attempt(text: str, voice: str) -> bytes:
    communicator = edge_tts.Communicate(text=text, voice=voice)
    output = bytearray()
    async for chunk in communicator.stream():
        if chunk.get("type") == "audio":
            output.extend(chunk["data"])
    return bytes(output)


def _system_voice(text: str) -> str | None:
    if SYSTEM_TTS_VOICE:
        return SYSTEM_TTS_VOICE
    is_russian = bool(re.search("[а-яё]", text, re.IGNORECASE))
    system = platform.system()
    if system == "Darwin":
        return "Milena" if is_russian else None
    if system == "Linux":
        return "ru" if is_russian else None
    return None


async def _system_tts_bytes(text: str) -> bytes:
    """Офлайн-синтез средствами ОС -> WAV-байты."""
    system = platform.system()
    voice = _system_voice(text)
    fd, wav_name = tempfile.mkstemp(prefix="jarvis_tts_", suffix=".wav")
    os.close(fd)
    wav_path = Path(wav_name)
    try:
        if system == "Darwin":
            cmd = ["say", "-o", str(wav_path), "--data-format=LEI16@22050"]
            if voice:
                cmd += ["-v", voice]
            cmd.append(text)
        elif system == "Windows":
            script = (
                "Add-Type -AssemblyName System.Speech;"
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"$s.SetOutputToWaveFile('{wav_path}');"
                "$s.Speak([Console]::In.ReadToEnd()); $s.Dispose();"
            )
            cmd = ["powershell", "-NoProfile", "-Command", script]
        else:
            binary = shutil.which("espeak-ng") or shutil.which("espeak")
            if not binary:
                raise RuntimeError("Нет офлайн-TTS: установите espeak-ng")
            cmd = [binary, "-w", str(wav_path)] + (["-v", voice] if voice else []) + [text]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate(text.encode("utf-8") if system == "Windows" else None)
        if process.returncode != 0:
            raise RuntimeError(f"system TTS failed: {stderr.decode('utf-8', 'ignore').strip()}")
        return wav_path.read_bytes()
    finally:
        wav_path.unlink(missing_ok=True)


async def synthesize_speech_base64(text: str, voice: str = DEFAULT_TTS_VOICE) -> str:
    """Синтезирует речь и возвращает base64-строку mp3."""
    audio_bytes = await synthesize_speech_bytes(text=text, voice=voice)
    if not audio_bytes:
        return ""
    return base64.b64encode(audio_bytes).decode("ascii")


async def speak(text: str, voice: str = DEFAULT_TTS_VOICE) -> None:
    """Синтезирует речь через edge-tts и асинхронно воспроизводит результат.

    Важно: функция полностью async и не блокирует event loop.
    - Генерация mp3 выполняется нативно асинхронным edge-tts.
    - Воспроизведение идёт через асинхронный subprocess.
    """
    normalized = (text or "").strip()
    if not normalized:
        return

    audio_bytes = await synthesize_speech_bytes(text=normalized, voice=voice)
    if not audio_bytes:
        return
    suffix = ".wav" if audio_bytes[:4] == b"RIFF" else ".mp3"
    tmp_path = Path(tempfile.gettempdir()) / f"jarvis_tts_{os.getpid()}_{int(asyncio.get_running_loop().time() * 1000)}{suffix}"

    try:
        tmp_path.write_bytes(audio_bytes)
        await _play_audio_file(tmp_path)
    finally:
        # Удаляем временный файл в любом случае.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


async def _play_audio_file(path: Path) -> None:
    """Кросс-платформенное воспроизведение аудиофайла через системный плеер."""
    cmd = _resolve_playback_command(path)
    if not cmd:
        raise RuntimeError(
            "Не найден системный аудио-плеер. "
            "Установите ffplay/mpg123/aplay (Linux) или используйте macOS/Windows стандартные средства."
        )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Ошибка воспроизведения аудио, код завершения: {process.returncode}")


def _resolve_playback_command(path: Path) -> list[str] | None:
    """Подбирает команду воспроизведения под ОС и доступные утилиты."""
    system = platform.system().lower()

    if "darwin" in system:
        return ["afplay", str(path)]

    if "windows" in system:
        # Воспроизведение через встроенный .NET SoundPlayer (без внешних зависимостей).
        ps_script = (
            "Add-Type -AssemblyName presentationCore;"
            f"$player = New-Object System.Media.SoundPlayer '{path}';"
            "$player.PlaySync();"
        )
        return ["powershell", "-NoProfile", "-Command", ps_script]

    # Linux/Unix fallback: ищем доступный плеер в порядке приоритета.
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
    if shutil.which("mpg123"):
        return ["mpg123", "-q", str(path)]
    if shutil.which("aplay"):
        return ["aplay", str(path)]

    return None


async def listen(
    *,
    timeout: float | None = None,
    phrase_time_limit: float | None = 10.0,
    language: str = DEFAULT_STT_LANGUAGE,
) -> str:
    """Слушает микрофон и возвращает распознанный текст.

    Критично: SpeechRecognition API блокирующий, поэтому весь процесс
    захвата аудио и распознавания уходит в отдельный поток через asyncio.to_thread.
    """
    return await asyncio.to_thread(
        _listen_blocking,
        timeout,
        phrase_time_limit,
        language,
    )


def _listen_blocking(
    timeout: float | None,
    phrase_time_limit: float | None,
    language: str,
) -> str:
    """Блокирующая часть STT (выполняется только в worker thread)."""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        # Небольшая калибровка фона снижает число ложных срабатываний.
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(
            source,
            timeout=timeout,
            phrase_time_limit=phrase_time_limit,
        )

    # Предпочитаем локальное распознавание через whisper интеграцию SR.
    # Этот путь использует локальную модель и не требует облака.
    if hasattr(recognizer, "recognize_whisper"):
        try:
            text = recognizer.recognize_whisper(
                audio,
                language=language,
                model="base",
            )
            return (text or "").strip()
        except ModuleNotFoundError:
            # В некоторых окружениях recognize_whisper требует пакет `whisper`.
            # Если его нет, делаем локальный fallback через faster-whisper.
            return _transcribe_audio_data_with_faster_whisper(audio, language)
        except sr.UnknownValueError:
            return ""

    # Если recognize_whisper отсутствует, тоже используем faster-whisper.
    return _transcribe_audio_data_with_faster_whisper(audio, language)


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    suffix: str = ".webm",
    language: str = DEFAULT_STT_LANGUAGE,
) -> str:
    """Транскрибирует аудио-байты из браузера в текст.

    Поток:
    1) сохраняем байты во временный файл;
    2) запускаем faster-whisper в отдельном потоке;
    3) удаляем временный файл;
    4) возвращаем распознанный текст.
    """
    if not audio_bytes:
        return ""

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(audio_bytes)
        temp_file.flush()
    finally:
        temp_file.close()

    try:
        return await asyncio.to_thread(
            _transcribe_file_blocking,
            str(temp_path),
            language,
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _get_whisper_model() -> WhisperModel:
    if WhisperModel is None:
        raise RuntimeError(
            "STT dependency is missing: install package 'faster-whisper'."
        )
    global _whisper_model
    with _whisper_model_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel(
                DEFAULT_WHISPER_MODEL,
                device=DEFAULT_WHISPER_DEVICE,
                compute_type=DEFAULT_WHISPER_COMPUTE_TYPE,
            )
        return _whisper_model


def _transcribe_file_blocking(file_path: str, language: str) -> str:
    """Блокирующая транскрибация файла через faster-whisper."""
    model = _get_whisper_model()
    segments, _ = model.transcribe(
        file_path,
        language=language or None,
        vad_filter=True,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


def _transcribe_audio_data_with_faster_whisper(
    audio_data: sr.AudioData,
    language: str,
) -> str:
    """Fallback для live-микрофона: AudioData -> временный WAV -> faster-whisper."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(audio_data.get_wav_data())
        temp_file.flush()
    finally:
        temp_file.close()

    try:
        return _transcribe_file_blocking(str(temp_path), language)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
