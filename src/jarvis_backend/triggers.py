from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class DirectoryWatcher:
    """Async-friendly filesystem watcher based on watchdog."""

    def __init__(
        self,
        directory: str | Path,
        on_prompt_ready: Callable[[str], Awaitable[None]],
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self._on_prompt_ready = on_prompt_ready
        self._observer = Observer()
        self._stop_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> None:
        """Start watching directory until stop() is called."""
        if not self.directory.exists() or not self.directory.is_dir():
            return

        self._loop = asyncio.get_running_loop()
        handler = _CreatedFileHandler(self._loop, self._handle_created)
        self._observer.schedule(handler, str(self.directory), recursive=False)
        self._observer.start()

        try:
            await self._stop_event.wait()
        finally:
            self._observer.stop()
            await asyncio.to_thread(self._observer.join)

    async def stop(self) -> None:
        self._stop_event.set()

    async def _handle_created(self, file_path: Path) -> None:
        # Небольшая пауза, чтобы файл успел полностью записаться.
        await asyncio.sleep(1)

        prompt = (
            f"Пользователь только что добавил новый файл: {file_path.name}. "
            "Проанализируй его суть и кратко сообщи об этом пользователю."
        )
        await self._on_prompt_ready(prompt)


class _CreatedFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        callback: Callable[[Path], Awaitable[None]],
    ) -> None:
        super().__init__()
        self._loop = loop
        self._callback = callback

    def on_created(self, event: FileSystemEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        def _spawn() -> None:
            asyncio.create_task(self._callback(file_path))

        self._loop.call_soon_threadsafe(_spawn)
