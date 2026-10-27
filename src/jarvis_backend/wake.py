"""Фоновое голосовое управление: «Джарвис, …» без нажатия кнопок.

Микрофон читается в отдельном потоке. Два движка активации:
- whisper (по умолчанию): фразы выделяются по уровню сигнала и распознаются локальным
  faster-whisper; если фраза начинается с «Джарвис» — остаток и есть команда
  («Джарвис, который час?»). Работает на любом языке; всё распознаётся локально.
- openwakeword: лёгкая нейросеть на фразу «Hey Jarvis» (английское произношение),
  после неё следующая фраза — команда. Экономнее по CPU.

Пока Джарвис говорит, микрофон игнорируется, чтобы он не услышал сам себя.
"""
from __future__ import annotations

import asyncio
import platform
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

import numpy as np

from .config import settings

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480  # 30 мс
OWW_CHUNK = 1280  # 80 мс — размер кадра openWakeWord
COMMAND_WINDOW_SECONDS = 8.0  # сколько ждать команду после «Джарвис!» без продолжения
UNMUTE_GRACE_SECONDS = 0.4

_WAKE_NAMES = r"(?:джарвис\w*|жарвис\w*|джервис\w*|джарвиз\w*|jarvis|jervis|jarvi)"
# До двух слов перед именем («Эй, Джарвис», «Слушай, Джарвис»), дальше — команда.
WAKE_RE = re.compile(rf"^\W*(?:\w+\W+){{0,2}}?{_WAKE_NAMES}\b[\s,.!?:;…—-]*", re.IGNORECASE)


def match_wake(text: str) -> str | None:
    """None — обращения нет; "" — только имя; иначе — текст команды."""
    match = WAKE_RE.match(text or "")
    if not match:
        return None
    return text[match.end():].strip()


class SpeechSegmenter:
    """Выделяет фразы из потока по уровню сигнала с адаптивным порогом шума."""

    def __init__(
        self,
        *,
        frame_samples: int = FRAME_SAMPLES,
        start_factor: float = 3.0,
        # Нижняя граница порога; основной порог — шум × start_factor.
        # Тихие USB-микрофоны дают речь всего 250–700 RMS, поэтому граница низкая.
        min_level: float = 120.0,
        silence_ms: int = 800,
        max_ms: int = 20000,
        preroll_ms: int = 300,
        min_speech_ms: int = 250,
    ) -> None:
        frame_ms = frame_samples * 1000 / SAMPLE_RATE
        self.start_factor = start_factor
        self.min_level = min_level
        self.silence_frames = int(silence_ms / frame_ms)
        self.max_frames = int(max_ms / frame_ms)
        self.preroll_frames = max(1, int(preroll_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.noise_floor: float | None = None
        self.reset()

    def reset(self) -> None:
        self._preroll: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []
        self._in_speech = False
        self._loud_run = 0
        self._silence_run = 0
        self._speech_frames = 0

    def _threshold(self) -> float:
        return max(self.min_level, (self.noise_floor or 0.0) * self.start_factor)

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        """frame: int16. Возвращает законченную фразу (int16) или None."""
        level = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) if frame.size else 0.0
        threshold = self._threshold()

        if not self._in_speech:
            # Порог шума обновляем только в тишине.
            self.noise_floor = level if self.noise_floor is None else 0.95 * self.noise_floor + 0.05 * level
            self._preroll.append(frame)
            if len(self._preroll) > self.preroll_frames:
                self._preroll.pop(0)
            self._loud_run = self._loud_run + 1 if level > threshold else 0
            if self._loud_run >= 2:
                self._in_speech = True
                self._frames = list(self._preroll)
                self._speech_frames = self._loud_run
                self._silence_run = 0
            return None

        self._frames.append(frame)
        if level > threshold * 0.7:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self.silence_frames or len(self._frames) >= self.max_frames:
            segment = np.concatenate(self._frames)
            enough = self._speech_frames >= self.min_speech_frames
            self.reset()
            return segment if enough else None
        return None


def _play_chime() -> None:
    if platform.system() == "Darwin":
        import subprocess

        subprocess.Popen(  # noqa: S603 - фиксированная системная команда
            ["afplay", "/System/Library/Sounds/Tink.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class VoiceListener:
    """Слушает микрофон в фоне и отдаёт распознанные команды в on_command."""

    def __init__(
        self,
        on_command: Callable[[str], Awaitable[None]],
        *,
        engine: str | None = None,
        threshold: float | None = None,
        language: str | None = None,
        device: str | None = None,
        on_wake: Callable[[], Awaitable[None]] | None = None,
        transcribe: Callable[[np.ndarray, bool], str] | None = None,
    ) -> None:
        self.on_command = on_command
        self.on_wake = on_wake
        self.engine = (engine or settings.wake_engine).lower()
        self.threshold = threshold if threshold is not None else settings.wake_threshold
        language = language if language is not None else settings.stt_language
        self.language = None if (language or "").lower() in {"", "auto"} else language
        self.device = device if device is not None else settings.wake_device
        self._transcribe = transcribe or self._whisper_transcribe
        self.debug = settings.voice_debug

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[tuple[str, Any]] | None = None
        self._thread: threading.Thread | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._stop = threading.Event()
        self._muted = 0
        self._muted_until = 0.0
        self._awaiting_command_until = 0.0
        self._frames_read = 0
        self._watchdog: asyncio.Task[None] | None = None
        self.error: str | None = None

    # ---- жизненный цикл

    async def start(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._stop.clear()
        self._consumer = asyncio.create_task(self._consume(), name="jarvis_voice_consumer")
        self._thread = threading.Thread(target=self._capture, name="jarvis_mic", daemon=True)
        self._thread.start()
        self._watchdog = asyncio.create_task(self._check_microphone(), name="jarvis_mic_watchdog")

    async def _check_microphone(self) -> None:
        """Чтение может просто зависнуть, пока macOS ждёт ответа на запрос доступа к микрофону."""
        await asyncio.sleep(6.0)
        if self._frames_read == 0 and self.error is None:
            self.error = (
                "С микрофона не приходят данные. Проверьте запрос доступа на экране или "
                "Системные настройки → Конфиденциальность и безопасность → Микрофон."
            )
            print(f"[VOICE] {self.error}", flush=True)

    async def stop(self) -> None:
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 2.0)
            self._thread = None
        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except BaseException:  # noqa: BLE001
                pass
            self._consumer = None

    @asynccontextmanager
    async def muted(self) -> AsyncIterator[None]:
        """Не слушать микрофон, пока Джарвис говорит сам."""
        self._muted += 1
        try:
            yield
        finally:
            self._muted -= 1
            self._muted_until = time.monotonic() + UNMUTE_GRACE_SECONDS

    def _is_muted(self) -> bool:
        return self._muted > 0 or time.monotonic() < self._muted_until

    def _awaiting_command(self) -> bool:
        return time.monotonic() < self._awaiting_command_until

    def _emit(self, kind: str, payload: Any = None) -> None:
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, payload))

    # ---- поток микрофона

    def _open_stream(self):
        import pyaudio

        pa = pyaudio.PyAudio()
        index = None
        if self.device:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0 and self.device.lower() in str(info.get("name", "")).lower():
                    index = i
                    break
            if index is None:
                print(f"[VOICE] microphone '{self.device}' not found, using default", flush=True)
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=index,
            frames_per_buffer=FRAME_SAMPLES,
        )
        name = pa.get_device_info_by_index(index)["name"] if index is not None else pa.get_default_input_device_info()["name"]
        return pa, stream, name

    def _capture(self) -> None:
        try:
            pa, stream, name = self._open_stream()
        except Exception as exc:  # noqa: BLE001
            self.error = f"Микрофон недоступен: {exc}"
            print(f"[VOICE] {self.error}", flush=True)
            return

        oww = None
        if self.engine == "openwakeword":
            try:
                oww = _load_openwakeword()
            except Exception as exc:  # noqa: BLE001
                self.error = f"openWakeWord недоступен ({exc}), переключаюсь на whisper"
                print(f"[VOICE] {self.error}", flush=True)
                self.engine = "whisper"

        print(f"[VOICE] listening on '{name}' (engine: {self.engine})", flush=True)
        segmenter = SpeechSegmenter()
        oww_buffer = np.zeros(0, dtype=np.int16)
        silent_frames = 0
        warned_silence = False
        debug_peak = 0.0
        try:
            while not self._stop.is_set():
                data = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                frame = np.frombuffer(data, dtype=np.int16)
                self._frames_read += 1
                if self.debug:
                    level = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                    debug_peak = max(debug_peak, level)
                    if self._frames_read % (5 * SAMPLE_RATE // FRAME_SAMPLES) == 0:
                        print(
                            f"[VOICE] level: peak_rms={debug_peak:.0f} noise={segmenter.noise_floor or 0:.0f} "
                            f"threshold={segmenter._threshold():.0f} muted={self._is_muted()}",
                            flush=True,
                        )
                        debug_peak = 0.0

                # macOS без разрешения на микрофон отдаёт сплошные нули, а не ошибку.
                if not warned_silence:
                    silent_frames = silent_frames + 1 if not frame.any() else -10**9
                    if silent_frames > 5 * SAMPLE_RATE // FRAME_SAMPLES:
                        warned_silence = True
                        self.error = (
                            "Микрофон отдаёт тишину. Разрешите доступ: Системные настройки → "
                            "Конфиденциальность и безопасность → Микрофон."
                        )
                        print(f"[VOICE] {self.error}", flush=True)

                if self._is_muted():
                    segmenter.reset()
                    oww_buffer = np.zeros(0, dtype=np.int16)
                    continue

                if oww is not None and not self._awaiting_command():
                    oww_buffer = np.concatenate([oww_buffer, frame])
                    while len(oww_buffer) >= OWW_CHUNK:
                        chunk, oww_buffer = oww_buffer[:OWW_CHUNK], oww_buffer[OWW_CHUNK:]
                        scores = oww.predict(chunk)
                        if max(scores.values(), default=0.0) >= self.threshold:
                            oww.reset()
                            segmenter.reset()
                            self._awaiting_command_until = time.monotonic() + COMMAND_WINDOW_SECONDS
                            self._emit("wake")
                            break
                    continue

                segment = segmenter.feed(frame)
                if segment is not None:
                    self._emit("segment", segment)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Ошибка чтения микрофона: {exc}"
            print(f"[VOICE] {self.error}", flush=True)
        finally:
            try:
                stream.stop_stream()
                stream.close()
                pa.terminate()
            except Exception:  # noqa: BLE001
                pass

    # ---- обработка фраз

    def _whisper_transcribe(self, segment: np.ndarray, hint_name: bool = False) -> str:
        from .audio import _get_whisper_model

        model = _get_whisper_model()
        audio = segment.astype(np.float32) / 32768.0
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0:
            # Тихий микрофон: выравниваем громкость, иначе Whisper возвращает пустоту или мусор.
            audio *= min(0.9 / peak, 30.0)
        # Подсказка помогает расслышать одиночное «Джарвис!», но в длинной фразе модель
        # иногда считает имя уже сказанным и выбрасывает его — поэтому только вторым проходом.
        extra = {"hotwords": "Джарвис"} if hint_name else {}
        segments, _ = model.transcribe(audio, language=self.language, vad_filter=True, beam_size=1, **extra)
        return " ".join(s.text.strip() for s in segments).strip()

    def _recognize(self, segment: np.ndarray) -> str:
        text = self._transcribe(segment, False)
        if self._awaiting_command() or self.engine != "whisper" or match_wake(text) is not None:
            return text
        hinted = self._transcribe(segment, True)
        return hinted if match_wake(hinted) is not None else text

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            kind, payload = await self._queue.get()
            try:
                if kind == "wake":
                    await self._wake()
                    continue
                if self.debug:
                    _save_debug_segment(payload)
                text = await asyncio.to_thread(self._recognize, payload)
                if self.debug:
                    level = float(np.sqrt(np.mean(payload.astype(np.float32) ** 2)))
                    print(f"[VOICE] heard {len(payload) / SAMPLE_RATE:.1f}s rms={level:.0f}: {text!r}", flush=True)
                if not text:
                    continue
                await self.handle_text(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[VOICE] error: {exc}", flush=True)

    async def handle_text(self, text: str) -> None:
        """Решает, обращена ли фраза к Джарвису, и запускает команду."""
        if self._awaiting_command():
            self._awaiting_command_until = 0.0
            command = match_wake(text)
            if command == "":
                await self._wake()  # снова только «Джарвис» — продолжаем ждать команду
                return
            await self._dispatch(command or text)
            return
        if self.engine == "openwakeword":
            return  # в этом режиме команды принимаются только после «Hey Jarvis»
        command = match_wake(text)
        if command is None:
            return
        if not command:
            await self._wake()
            return
        await self._dispatch(command)

    async def _wake(self) -> None:
        self._awaiting_command_until = time.monotonic() + COMMAND_WINDOW_SECONDS
        async with self.muted():
            _play_chime()
            await asyncio.sleep(0.25)
        if self.on_wake is not None:
            await self.on_wake()

    async def _dispatch(self, command: str) -> None:
        print(f"[VOICE] command: {command}", flush=True)
        await self.on_command(command)


def _save_debug_segment(segment: np.ndarray) -> None:
    """JARVIS_VOICE_DEBUG: последние 20 услышанных фраз в .jarvis/voice_debug/*.wav."""
    import wave

    folder = settings.state_dir / "voice_debug"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(segment.astype(np.int16).tobytes())
    for old in sorted(folder.glob("*.wav"))[:-20]:
        old.unlink(missing_ok=True)


def _load_openwakeword():
    from openwakeword import utils
    from openwakeword.model import Model

    name = settings.wake_model
    try:
        return Model(wakeword_models=[name], inference_framework="onnx")
    except Exception:  # noqa: BLE001
        # Модели не входят в пакет — скачиваются один раз при первом запуске.
        utils.download_models(model_names=[name])
        return Model(wakeword_models=[name], inference_framework="onnx")
