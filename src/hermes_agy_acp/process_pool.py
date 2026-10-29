from __future__ import annotations

import collections
import logging
import os
import queue
import re
import subprocess
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from hermes_agy_acp.config import get_settings

logger = logging.getLogger(__name__)

SENSITIVE_PATTERNS = [
    (re.compile(r"(Bearer\s+)[^\s]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"((?:password|token|secret|api_?key)=)[^\s&]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(ya29\.[a-zA-Z0-9_-]+)", re.IGNORECASE), r"[REDACTED]"),
]


class StderrRingBuffer:
    """Thread-safe ring buffer storing sanitized stderr lines."""

    def __init__(self, max_lines: int = 100) -> None:
        self.max_lines = max_lines
        self._deque: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        """Sanitize line and append to ring buffer."""
        clean = line.rstrip("\r\n")
        for pattern, replacement in SENSITIVE_PATTERNS:
            clean = pattern.sub(replacement, clean)
        with self._lock:
            self._deque.append(clean)

    def get_lines(self) -> list[str]:
        """Return snapshot of current lines."""
        with self._lock:
            return list(self._deque)

    def clear(self) -> None:
        """Clear buffer lines."""
        with self._lock:
            self._deque.clear()


class ACPProcessSupervisor:
    """Supervises the stdio process of the ACP server."""

    def __init__(
        self,
        command: list[str] | None = None,
        cwd: Path | str | None = None,
        env: dict[str, str] | None = None,
        auto_respawn: bool = True,
        stderr_maxlen: int = 100,
    ) -> None:
        self.command = command
        self.cwd = str(cwd) if cwd is not None else None
        self.env = env if env is not None else os.environ.copy()
        self.auto_respawn = auto_respawn
        self.stderr_buffer = StderrRingBuffer(max_lines=stderr_maxlen)
        self.process: subprocess.Popen[bytes] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start or restart the subprocess with supervised stdio."""
        if self.is_alive():
            return

        cmd = self.command or get_settings().get_server_command()
        self._stdout_queue = queue.Queue()
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=self.env,
            bufsize=0,
        )

        stdout_thread = threading.Thread(
            target=self._read_stdout_worker,
            args=(self.process.stdout,),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr_worker,
            args=(self.process.stderr,),
            daemon=True,
        )

        stdout_thread.start()
        stderr_thread.start()
        self._threads = [stdout_thread, stderr_thread]

    def _read_stdout_worker(self, stream: Any) -> None:
        try:
            for raw_line in iter(stream.readline, b""):
                text = raw_line.decode("utf-8", errors="replace")
                self._stdout_queue.put(text)
        except (OSError, ValueError) as err:
            logger.debug("Stdout reader terminated: %s", err)
        finally:
            self._stdout_queue.put(None)

    def _read_stderr_worker(self, stream: Any) -> None:
        try:
            for raw_line in iter(stream.readline, b""):
                text = raw_line.decode("utf-8", errors="replace")
                self.stderr_buffer.append(text)
        except (OSError, ValueError) as err:
            logger.debug("Stderr reader terminated: %s", err)

    def is_alive(self) -> bool:
        """Return True if subprocess is actively running."""
        return self.process is not None and self.process.poll() is None

    def ensure_running(self) -> None:
        """Ensure process is running, auto-respawning if enabled."""
        if not self.is_alive():
            if self.process is None or self.auto_respawn:
                self.start()
            else:
                raise RuntimeError("ACP server process has terminated and auto-respawn is disabled")

    def write_line(self, line: str) -> None:
        """Write line to process stdin."""
        self.ensure_running()
        if not self.process or not self.process.stdin:
            raise RuntimeError("Subprocess stdin is unavailable")

        payload = line if line.endswith("\n") else f"{line}\n"
        self.process.stdin.write(payload.encode("utf-8"))
        self.process.stdin.flush()

    def read_line(self, timeout: float | None = None) -> str:
        """Read line from process stdout with optional timeout."""
        self.ensure_running()
        try:
            item = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as err:
            raise TimeoutError(f"Timed out waiting for process output ({timeout}s)") from err

        if item is None:
            raise EOFError("Process stdout closed (EOF)")
        return item


    def terminate(self, timeout: float = 3.0) -> None:
        """Gracefully terminate process (SIGTERM then SIGKILL)."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None

    stop = terminate

    def restart(self) -> None:
        """Terminate existing process and launch a fresh instance."""
        self.terminate()
        self.start()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.terminate()
