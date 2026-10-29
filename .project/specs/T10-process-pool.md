# Specification T10: ACP Server Process Supervisor (process_pool.py)

## 1. Overview
The `process_pool.py` module manages the lifecycle of the underlying `agy_acp_server` stdio process.
It supervises execution, auto-respawns on unexpected termination, buffers and sanitizes stderr logs in a fixed-size ring buffer, and handles graceful signal-based shutdown.

## 2. Public Interface

### 2.1 Classes

#### `StderrRingBuffer`:
- Ring buffer storing up to `max_lines` (default: 100) stderr lines.
- `append(line: str) -> None`:
  - Strips trailing newlines.
  - Sanitizes sensitive patterns (e.g., Bearer tokens, passwords, oauth hashes: replaces matching tokens with `[REDACTED]`).
- `get_lines() -> list[str]`:
  - Returns list of current buffered lines.
- `clear() -> None`:
  - Clears buffer.

#### `ACPProcessSupervisor`:
- Constructor:
  - `__init__(command: list[str] | None = None, cwd: Path | str | None = None, env: dict[str, str] | None = None, auto_respawn: bool = True, stderr_maxlen: int = 100)`
  - If `command` is None, loads command from `get_settings().get_server_command()`.
- Attributes:
  - `command: list[str]`
  - `cwd: str | None`
  - `env: dict[str, str] | None`
  - `auto_respawn: bool`
  - `stderr_buffer: StderrRingBuffer`
  - `process: subprocess.Popen[bytes] | None`
- Methods:
  - `start() -> None`:
    - Spawns subprocess with `stdin=PIPE, stdout=PIPE, stderr=PIPE`.
    - Spawns background daemon thread to consume stderr into `stderr_buffer`.
  - `is_alive() -> bool`:
    - Returns `True` if process exists and `poll() is None`.
  - `ensure_running() -> None`:
    - If `not is_alive()`: if `auto_respawn` is True, restarts process; otherwise raises `RuntimeError("Process terminated")`.
  - `write_line(line: str) -> None`:
    - Ensures process is running. Writes string (ensuring trailing newline) encoded as UTF-8 to `stdin`, flushes.
  - `read_line(timeout: float | None = None) -> str`:
    - Ensures process is running. Reads next line from `stdout`.
    - If timeout occurs or EOF reached, raises `TimeoutError` or `EOFError`.
  - `terminate(timeout: float = 3.0) -> None`:
    - Sends `SIGTERM` (or calls `process.terminate()`).
    - Waits up to `timeout` seconds. If still alive, sends `SIGKILL` (calls `process.kill()`).
  - `restart() -> None`:
    - Terminates current process and starts a fresh one.
  - `__enter__() -> ACPProcessSupervisor` / `__exit__(...)`:
    - Context manager ensuring `terminate()` on exit.
