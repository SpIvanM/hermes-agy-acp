import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_BUS_SYS_MODULES_KEY = "_hermes_live_status_bus"
_BUS_INIT_LOCK = threading.Lock()

EventKind = Literal["thinking", "tool_start", "tool_update", "tool_done", "turn_end"]

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization headers and Bearer tokens
    (re.compile(r"(Authorization\s*:\s*(?:Bearer|Basic|Token)\s+)[^\s\"']+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"\b(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE), r"\1[REDACTED]"),
    # Telegram bot tokens: <digits>:<alnum_or_dash_>=20+
    (re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{16,}\b"), "[REDACTED]"),
    # Well-known API key prefixes
    (re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{8,}\b"), "[REDACTED]"),
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{12,})\b"), "[REDACTED]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{10,}\b"), "[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED]"),
    # CLI flags carrying secrets (--token ..., --api-key=..., etc.)
    (
        re.compile(
            r"(--?(?:bot[-_]?token|access[-_]?token|private[-_]?key|api[-_]?key|token|secret|password|passwd|auth)(?:\s+|=))(?:\"[^\"]*\"|'[^']*'|\S+)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED]",
    ),
    # Env/config assignments (BOT_TOKEN=..., API_KEY: ...)
    (
        re.compile(
            r"\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY|AUTH_KEY)[A-Z0-9_]*\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|\S+)"
        ),
        r"\1[REDACTED]",
    ),
    # JSON key-value secrets
    (
        re.compile(
            r"(\"(?:bot_token|access_token|api_key|token|secret|password|authorization)\"\s*:\s*\")([^\"]+)(\")",
            re.IGNORECASE,
        ),
        r"\1[REDACTED]\3",
    ),
)

_AGY_TAG_ROLE_RE = re.compile(r"\[P:[^\]]*\bR:([A-Za-z0-9_.-]+)\]")
_AGY_FLAG_ROLE_RE = re.compile(r"\bagy\b[^\n]*?(?:--role|-r)\s+[\"']?([A-Za-z0-9_.-]+)[\"']?")
_ACPX_SESSION_ROLE_RE = re.compile(r"\bacpx\b[^\n]*?(?:\s|^)-s\s+[\"']?([A-Za-z0-9_.-]+)[\"']?")


@dataclass(frozen=True)
class LiveStatusEvent:
    turn_id: str
    kind: EventKind
    agent: str
    tool_name: str = ""
    args_summary: str = ""
    call_id: str = ""
    status: str = ""
    timestamp: float = 0.0


def redact_secrets(text: str) -> str:
    """Mask tokens, API keys, passwords, and auth headers in text."""
    if not isinstance(text, str):
        text = str(text)
    masked = text
    for pattern, repl in _SECRET_PATTERNS:
        masked = pattern.sub(repl, masked)
    return masked


def redact_args(text: Any, max_len: int = 120) -> str:
    """Redact secrets, collapse whitespace to one line, and truncate safely."""
    if text is None:
        return ""
    raw = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    single_line = " ".join(redact_secrets(raw).split())
    if max_len > 1 and len(single_line) > max_len:
        return single_line[: max_len - 1].rstrip() + "…"
    return single_line[:max_len] if max_len >= 0 else single_line


def extract_subagent_role_from_command(cmd: str) -> str | None:
    """Extract subagent role from agy [P:... R:<Role>] or acpx -s <role> command lines."""
    if not cmd:
        return None
    for pattern in (_AGY_TAG_ROLE_RE, _ACPX_SESSION_ROLE_RE, _AGY_FLAG_ROLE_RE):
        match = pattern.search(cmd)
        if match:
            role = match.group(1).strip()
            if role:
                return role
    return None


class LiveStatusBus:
    """Thread-safe in-process pub/sub bus registered in sys.modules."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[int, tuple[str, Callable[[LiveStatusEvent], None]]] = {}
        self._next_sub_id = 1
        self.LiveStatusEvent = LiveStatusEvent
        self.redact_secrets = redact_secrets
        self.redact_args = redact_args

    @staticmethod
    def _normalize_filter(turn_id_or_all: str | None) -> str:
        if not turn_id_or_all or turn_id_or_all in ("*", "all", "ALL"):
            return "*"
        return str(turn_id_or_all)

    def subscribe(
        self,
        turn_id_or_all: str,
        callback: Callable[[LiveStatusEvent], None],
    ) -> Callable[[], None]:
        """Register a callback for a specific turn_id or '*'/'all'. Returns idempotent unsubscribe."""
        target = self._normalize_filter(turn_id_or_all)
        with self._lock:
            sub_id = self._next_sub_id
            self._next_sub_id += 1
            self._subscribers[sub_id] = (target, callback)

        def _unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(sub_id, None)

        return _unsubscribe

    def publish(self, event: LiveStatusEvent) -> None:
        """Broadcast an immutable LiveStatusEvent to matching subscribers without holding locks."""
        event_turn_id = getattr(event, "turn_id", "")
        with self._lock:
            targets = [
                cb
                for flt, cb in self._subscribers.values()
                if flt == "*" or flt == event_turn_id
            ]
        for cb in targets:
            try:
                cb(event)
            except Exception as exc:
                logger.debug("LiveStatusBus subscriber %r raised %s", cb, type(exc).__name__)

    def clear(self) -> None:
        """Remove all subscribers (intended for tests)."""
        with self._lock:
            self._subscribers.clear()


def get_live_bus() -> LiveStatusBus:
    """Return the process-wide singleton LiveStatusBus in sys.modules['_hermes_live_status_bus']."""
    bus = sys.modules.get(_BUS_SYS_MODULES_KEY)
    if bus is not None and hasattr(bus, "publish") and hasattr(bus, "subscribe"):
        return bus  # type: ignore[return-value]
    with _BUS_INIT_LOCK:
        bus = sys.modules.get(_BUS_SYS_MODULES_KEY)
        if bus is None or not (hasattr(bus, "publish") and hasattr(bus, "subscribe")):
            bus = LiveStatusBus()
            sys.modules[_BUS_SYS_MODULES_KEY] = bus
        return bus  # type: ignore[return-value]


def publish(event: LiveStatusEvent) -> None:
    """Publish a LiveStatusEvent to the singleton bus."""
    get_live_bus().publish(event)


def subscribe(
    turn_id_or_all: str,
    callback: Callable[[LiveStatusEvent], None],
) -> Callable[[], None]:
    """Subscribe to LiveStatusEvents on the singleton bus."""
    return get_live_bus().subscribe(turn_id_or_all, callback)


def _default_child_session_dirs() -> list[Path]:
    home = Path.home()
    return [
        home / ".acpx" / "sessions",
        home / ".gemini" / "antigravity-cli" / "conversations",
    ]


def _status_to_event_kind(status: str) -> EventKind:
    norm = status.strip().lower()
    if norm in ("completed", "failed", "done", "cancelled", "error"):
        return "tool_done"
    if norm in ("pending", "started", ""):
        return "tool_start"
    return "tool_update"


def collect_child_session_events(
    session_dirs: list[Path] | None = None,
    *,
    since_ts: float = 0.0,
    turn_id: str = "",
    default_agent: str = "Subagent",
) -> list[LiveStatusEvent]:
    """Safely read local agy/acpx session JSON/JSONL records into LiveStatusEvents (read-only)."""
    dirs = session_dirs if session_dirs is not None else _default_child_session_dirs()
    events: list[LiveStatusEvent] = []

    for directory in dirs:
        try:
            if not directory.is_dir() or not os.access(directory, os.R_OK | os.X_OK):
                continue
            entries = sorted(directory.iterdir())
        except OSError:
            continue

        for path in entries:
            try:
                if not path.is_file() or path.suffix not in (".jsonl", ".json"):
                    continue
                stat = path.stat()
                if since_ts and stat.st_mtime + 1.0 < since_ts:
                    continue
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for line in raw_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue

                rec_ts = record.get("timestamp")
                ts = float(rec_ts) if isinstance(rec_ts, (int, float)) else stat.st_mtime
                if since_ts and ts < since_ts:
                    continue

                cmd_text = str(record.get("command") or record.get("cmd") or "")
                agent = str(record.get("agent") or record.get("role") or "").strip()
                if not agent and cmd_text:
                    agent = extract_subagent_role_from_command(cmd_text) or ""
                if not agent:
                    agent = default_agent

                tool_name = str(record.get("tool") or record.get("tool_name") or "").strip()
                if not tool_name and cmd_text:
                    tool_name = "run_command"
                raw_args = record.get("args") or record.get("args_summary") or cmd_text
                status = str(record.get("status") or "in_progress").strip()
                kind: EventKind = (
                    "tool_start"
                    if cmd_text and status.lower() == "in_progress" and not record.get("tool")
                    else _status_to_event_kind(status)
                )
                events.append(
                    LiveStatusEvent(
                        turn_id=turn_id,
                        kind=kind,
                        agent=agent,
                        tool_name=tool_name,
                        args_summary=redact_args(raw_args),
                        call_id=str(record.get("call_id") or record.get("toolCallId") or ""),
                        status=status,
                        timestamp=ts,
                    )
                )
    return events


class ChildSessionTailer:
    """Non-blocking read-only poller forwarding child agy/acpx session steps to live_bus."""

    def __init__(
        self,
        turn_id: str,
        *,
        session_dirs: list[Path] | None = None,
        since_ts: float | None = None,
        poll_interval: float = 0.5,
    ) -> None:
        self._turn_id = turn_id
        self._session_dirs = session_dirs
        self._since_ts = time.time() if since_ts is None else since_ts
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_keys: set[tuple[str, str, str, str, float]] = set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="acp-child-tailer", daemon=True)
        self._thread.start()

    def poll_once(self) -> None:
        events = collect_child_session_events(
            session_dirs=self._session_dirs,
            since_ts=self._since_ts,
            turn_id=self._turn_id,
        )
        for ev in events:
            key = (ev.agent, ev.tool_name, ev.call_id, ev.status, ev.timestamp)
            if key not in self._seen_keys:
                self._seen_keys.add(key)
                publish(ev)

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            try:
                self.poll_once()
            except Exception:
                continue

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


# Register singleton eagerly upon module import
get_live_bus()

__all__ = [
    "ChildSessionTailer",
    "EventKind",
    "LiveStatusBus",
    "LiveStatusEvent",
    "collect_child_session_events",
    "extract_subagent_role_from_command",
    "get_live_bus",
    "publish",
    "redact_args",
    "redact_secrets",
    "subscribe",
]
